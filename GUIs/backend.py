import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from sklearn.cluster import KMeans

# --- Your Neural Network Definition ---
class SimpleResCNN(nn.Module):
    def __init__(self, num_classes=43):  ## Our dataset (GTSRB) has 43 classes
        super(SimpleResCNN, self).__init__()
        
        ## (first convvv)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  ## 64 -> 32
        )
        
        ## (second convvv + skip conn)
        self.conv2_main = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv2_skip = nn.Conv2d(32, 64, kernel_size=1)

        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)  ## 32 -> 16

        ## (another final conv, like a combiner one)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  ## 16 -> 8
        )

        ## Classifier (final step)
        self.gap = nn.AdaptiveAvgPool2d(1)  ## 8x8 -> 1x1
        self.fc_class = nn.Linear(128, num_classes)
        self.fc_bbox = nn.Linear(128, 4)

    def forward(self, x):
        ## step 1
        x = self.conv1(x)  ## (3, 64, 64) -> (32, 32, 32)

        ## step 2
        identity = self.conv2_skip(x)  ## (32, 32, 32) -> (64, 32, 32)
        out = self.conv2_main(x)
        x = self.relu2(out + identity)  ## Residual conn
        x = self.pool2(x)  ## -> (64, 16, 16)

        ## step 3
        x = self.conv3(x)  ## -> (128, 8, 8)
        
        ## step 4
        x = self.gap(x).view(x.size(0), -1)  ## -> (128,)
        class_logits = self.fc_class(x)
        bbox_preds = self.fc_bbox(x)
        
        return class_logits, bbox_preds

# --- Class Names ---
class_names = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)", "Speed limit (70km/h)",
    "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)", "No passing",
    "No passing for vehicles over 3.5 metric tons", "Right-of-way at the next intersection", "Priority road", "Yield",
    "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

# --- Load Model ---
def load_model(model_path="/home/theodoros/projects/Traffic_signs/model_weights.pth", num_classes=43, device='cpu'):
    model = SimpleResCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- Process Single Image ---
def process_single_image(img_path_or_bytes, resize_size=(64, 64)):
    """
    Processes a single image given its path or bytes:
    - Loads the image
    - Resizes it
    - Normalizes it [0, 1]
    - Converts to PyTorch tensor in (C, H, W) format and adds batch dimension
    Returns (image_tensor, original_image)
    """
    if isinstance(img_path_or_bytes, bytes):
        nparr = np.frombuffer(img_path_or_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(img_path_or_bytes)
    if img is None:
        raise FileNotFoundError("Image not found or could not be loaded.")
    img_resize = cv2.resize(img, resize_size)
    img_norm = img_resize.astype(np.float32) / 255.0
    image_tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0)  # shape: (1, 3, 64, 64)
    return image_tensor, img_resize

# --- Denormalize Bounding Box ---
def denormalize_bbox(bbox, img_width, img_height):
    """
    Ensures bounding box coordinates are within image boundaries.
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1 * img_width), img_width - 1))
    y1 = max(0, min(int(y1 * img_height), img_height - 1))
    x2 = max(0, min(int(x2 * img_width), img_width - 1))
    y2 = max(0, min(int(y2 * img_height), img_height - 1))
    return x1, y1, x2, y2

# --- KMeans Segmentation ---
def segment_image_kmeans(image, n_clusters=4):
    h, w, c = image.shape
    flat_img = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    labels = kmeans.fit_predict(flat_img)
    segmented_img = labels.reshape(h, w)
    return segmented_img

# --- Predict and Segment ---
def predict_and_segment(model, img_path_or_bytes, device='cpu'):
    """
    Processes one image, predicts, returns dict with prediction, bbox, confidence, segmented image.
    """
    image_tensor, img = process_single_image(img_path_or_bytes)
    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        logits, bbox_preds = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_label].item()
        pred_bbox = bbox_preds.squeeze().cpu().numpy()

    img_height, img_width = img.shape[:2]
    x1_pred, y1_pred, x2_pred, y2_pred = denormalize_bbox(pred_bbox, img_width, img_height)

    bbox = [x1_pred, y1_pred, x2_pred, y2_pred]
    segmented_img = segment_image_kmeans(img)

    return {
        "image": img,
        "pred_label": pred_label,
        "pred_label_name": class_names[pred_label],
        "confidence": confidence,
        "bbox": bbox,
        "segmented_img": segmented_img
    }