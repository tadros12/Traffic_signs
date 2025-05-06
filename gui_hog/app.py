import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image
import os

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide")

# --- Configuration & Model Loading ---
MODEL_PATH = '/home/theodoros/projects/Traffic_signs/svm_hog_model.joblib' # Your updated path

# Define class names (as provided by you)
CLASS_NAMES = [
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

@st.cache_resource
def load_model_cached(model_path):
    """Loads the pre-trained model."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

model = load_model_cached(MODEL_PATH)

def process_and_visualize_image(image_bytes, resize_shape=(32, 32)):

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_cv is None:
            st.error("Could not decode image. Please upload a valid image file.")
            return None
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        processed_image_display = cv2.resize(img_rgb, resize_shape)
        processed_image_cv = cv2.resize(img_cv, resize_shape)

        gray_image = cv2.cvtColor(processed_image_cv, cv2.COLOR_BGR2GRAY)

        hog_features, hog_visualization_image = hog(
            gray_image,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True,
            feature_vector=True
        )
        if hog_visualization_image.max() > 0:
            hog_visualization_image = (hog_visualization_image / np.max(hog_visualization_image) * 255).astype(np.uint8)
        else:
            hog_visualization_image = hog_visualization_image.astype(np.uint8)

        sift = cv2.SIFT_create()
        keypoints, _ = sift.detectAndCompute(gray_image, None)
        sift_visualization_image = cv2.drawKeypoints(
            processed_image_cv,
            keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        sift_visualization_image_rgb = cv2.cvtColor(sift_visualization_image, cv2.COLOR_BGR2RGB)

        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3) # x direction
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3) # y direction 
        
        sobel_magnitude = cv2.magnitude(sobelx, sobely)
        sobel_edges_image = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return {
            "hog_features": hog_features,
            "original_processed_display": processed_image_display,
            "hog_visualization_image": hog_visualization_image,
            "sift_visualization_image": sift_visualization_image_rgb,
            "sobel_edges_image": sobel_edges_image
        }
    except Exception as e:
        st.error(f"Error during image processing: {e}")
        return None

# --- Streamlit UI ---
st.title("🚦 Traffic Sign Analyzer 🚦")

st.markdown("""
Upload an image of a traffic sign. The application will:
1.  Process the image (resize to 32x32).
2.  Extract HOG features and classify the sign using a pre-trained SVM model.
3.  Display visualizations: HOG, SIFT keypoints, and Sobel edges.
""")

if model is None:
    st.error(f"Model could not be loaded from {MODEL_PATH}. Please check the path and ensure the model file is valid. Check console for more details.")
else:
    uploaded_file = st.file_uploader(
        "Choose an image file (PPM, PNG, JPG)...",
        type=["ppm", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("🖼️ Processing Results")
        image_bytes = uploaded_file.getvalue()

        processing_results = process_and_visualize_image(image_bytes)

        if processing_results:
            try:
                prediction_index = model.predict(processing_results["hog_features"].reshape(1, -1))[0]
                if 0 <= prediction_index < len(CLASS_NAMES):
                    predicted_class_name = CLASS_NAMES[prediction_index]
                else:
                    predicted_class_name = "Unknown Class (index out of bounds)"
            except Exception as e:
                st.error(f"Error during classification: {e}")
                predicted_class_name = "Error in prediction"

            col1, col2 = st.columns(2)

            with col1:
                st.header("Input & Classification")
                st.image(
                    processing_results["original_processed_display"],
                    caption="Processed Input (32x32)",
                    use_container_width=True # Changed from use_column_width
                )
                st.success(f"**Predicted Class:** {predicted_class_name}")

            with col2:
                st.header("Visualizations")
                st.image(
                    processing_results["hog_visualization_image"],
                    caption="HOG Visualization",
                    use_container_width=True # Changed from use_column_width
                )
                st.image(
                    processing_results["sift_visualization_image"],
                    caption="SIFT Keypoints",
                    use_container_width=True # Changed from use_column_width
                )
                st.image(
                    processing_results["sobel_edges_image"],
                    caption="Sobel Edges",
                    use_container_width=True # Changed from use_column_width
                )
        else:
            st.warning("Image processing failed. Please try another image or check the console for errors if running locally.")



