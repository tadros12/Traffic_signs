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

# --- Denormalize Bounding Box ---
def denormalize_bbox(bbox_normalized, img_width, img_height):
    x1_n, y1_n, x2_n, y2_n = bbox_normalized
    x1 = max(0, min(int(x1_n * img_width), img_width - 1))
    y1 = max(0, min(int(y1_n * img_height), img_height - 1))
    x2 = max(0, min(int(x2_n * img_width), img_width - 1))
    y2 = max(0, min(int(y2_n * img_height), img_height - 1))
    return x1, y1, x2, y2

# --- KMeans Segmentation ---
def segment_image_kmeans(image, n_clusters=4):
    h, w, c = image.shape
    if h == 0 or w == 0: ## preventing error on empty image
        return np.array([])
    flat_img = image.reshape(-1, 3)
    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0) 
    except TypeError: 
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(flat_img)
    segmented_img = labels.reshape(h, w)
    return segmented_img

# --- New function to find object regions ---
def find_sign_regions(image_cv2, min_area_threshold=100):
    lower_white = np.array([255, 255, 255], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(image_cv2, lower_white, upper_white)
    object_mask = cv2.bitwise_not(white_mask)
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_bboxes = []
    img_h, img_w = image_cv2.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > min_area_threshold:
            if len(contours) == 1 and x == 0 and y == 0 and w == img_w and h == img_h:
                continue 
            detected_bboxes.append((x, y, w, h))
    return detected_bboxes


# --- Main prediction function for multiple signs ---
def predict_multiple_signs(model, img_path_or_bytes, device='cpu', model_input_resize_size=(64, 64)):
    if isinstance(img_path_or_bytes, bytes):
        nparr = np.frombuffer(img_path_or_bytes, np.uint8)
        original_full_image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        original_full_image_cv2 = cv2.imread(img_path_or_bytes)

    if original_full_image_cv2 is None or original_full_image_cv2.size == 0:
        raise FileNotFoundError("Image not found, could not be loaded, or is empty.")

    orig_h, orig_w = original_full_image_cv2.shape[:2]
    
    sign_crop_rects = find_sign_regions(original_full_image_cv2)

    if not sign_crop_rects:
        if not np.all(original_full_image_cv2 == 255):
             sign_crop_rects = [(0, 0, orig_w, orig_h)]

    all_predictions_data = []
    annotated_image_for_gui = original_full_image_cv2.copy()

    # --- ADJUST DRAWING PARAMETERS HERE ---
    bbox_thickness = 1  # Was 2
    font_scale = 0.5   # Was 0.5
    font_thickness = 1  # Was 1 (can remain 1 for clarity with smaller font)
    # ---

    for (x_crop, y_crop, w_crop, h_crop) in sign_crop_rects:
        if w_crop <= 0 or h_crop <= 0:
            continue
        cropped_sign_cv2 = original_full_image_cv2[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]

        if cropped_sign_cv2.size == 0: 
            continue

        img_resized_for_model = cv2.resize(cropped_sign_cv2, model_input_resize_size)
        img_norm = img_resized_for_model.astype(np.float32) / 255.0
        image_tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.inference_mode():
            class_logits, bbox_preds_normalized_local = model(image_tensor)
            probs = F.softmax(class_logits, dim=1)
            pred_label_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label_idx].item()
            pred_bbox_normalized_for_resized_input = bbox_preds_normalized_local.squeeze().cpu().numpy()

        x1_local_pred, y1_local_pred, x2_local_pred, y2_local_pred = denormalize_bbox(
            pred_bbox_normalized_for_resized_input, model_input_resize_size[0], model_input_resize_size[1]
        )
        
        scale_x_from_model_to_crop = w_crop / model_input_resize_size[0]
        scale_y_from_model_to_crop = h_crop / model_input_resize_size[1]

        x1_on_crop = x1_local_pred * scale_x_from_model_to_crop
        y1_on_crop = y1_local_pred * scale_y_from_model_to_crop
        x2_on_crop = x2_local_pred * scale_x_from_model_to_crop
        y2_on_crop = y2_local_pred * scale_y_from_model_to_crop
        
        final_x1 = int(x_crop + x1_on_crop)
        final_y1 = int(y_crop + y1_on_crop)
        final_x2 = int(x_crop + x2_on_crop)
        final_y2 = int(y_crop + y2_on_crop)

        final_x1 = max(0, min(final_x1, orig_w - 1))
        final_y1 = max(0, min(final_y1, orig_h - 1))
        final_x2 = max(0, min(final_x2, orig_w - 1))
        final_y2 = max(0, min(final_y2, orig_h - 1))
        
        if final_x1 >= final_x2: final_x2 = final_x1 + 1
        if final_y1 >= final_y2: final_y2 = final_y1 + 1
        final_x2 = min(final_x2, orig_w -1) # Ensure it's within bounds
        final_y2 = min(final_y2, orig_h -1) # Ensure it's within bounds

        prediction_details = {
            "pred_label_idx": pred_label_idx,
            "pred_label_name": class_names[pred_label_idx],
            "confidence": confidence,
            "bbox_on_original": [final_x1, final_y1, final_x2, final_y2],
            "crop_region_on_original": [x_crop, y_crop, x_crop + w_crop, y_crop + h_crop]
        }
        all_predictions_data.append(prediction_details)

        if final_x2 > final_x1 and final_y2 > final_y1:
            cv2.rectangle(annotated_image_for_gui, (final_x1, final_y1), (final_x2, final_y2), 
                          (0, 0, 255), bbox_thickness) # Use variable
            label_text = f"{class_names[pred_label_idx]}: {confidence*100:.1f}%"
            
            # Adjust text Y position slightly for smaller font
            text_y_pos = final_y1 # Was final_y1 - 10
            if text_y_pos < 5: # If too close to top
                text_y_pos = final_y1 + int(font_scale * 25) # Position inside, adjust based on new font_scale
                if text_y_pos >= final_y2: # Ensure it doesn't go past the bottom of the box
                     text_y_pos = final_y1 + (final_y2 - final_y1) // 2


            cv2.putText(annotated_image_for_gui, label_text, (final_x1, text_y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 
                        font_thickness, cv2.LINE_AA) # Use variables

    segmented_full_img = None
    if original_full_image_cv2.ndim == 3 and original_full_image_cv2.shape[2] == 3:
        segmented_full_img = segment_image_kmeans(original_full_image_cv2.copy())

    return {
        "annotated_image_cv2": annotated_image_for_gui, 
        "predictions_list": all_predictions_data,      
        "segmented_original_img_cv2": segmented_full_img 
    }