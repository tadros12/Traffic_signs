import streamlit as st
import matplotlib.pyplot as plt
import cv2 # Import cv2 if not already imported, for BGR2RGB conversion
from backend import load_model, predict_multiple_signs, class_names # Use the new function

st.title("Traffic Sign Recognition Demo (GTSRB) - Multi-Sign Detection")

# Model loading (cache for performance)
@st.cache_resource
def get_model():
    # Ensure your model path is correct
    return load_model(model_path="/home/theodoros/projects/Traffic_signs/model_weights.pth", num_classes=43)

model = get_model()
device = "cpu"  # Or "cuda" if you have a GPU and PyTorch with CUDA

uploaded_file = st.file_uploader("Choose a traffic sign image (can contain multiple signs)...", type=['jpg', 'jpeg', 'png', 'ppm'])

if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    
    # Call the new processing function
    # The function predict_multiple_signs is now the main entry point from backend
    results = predict_multiple_signs(model, img_bytes, device=device)

    st.subheader("Detected Signs and Predictions on Original Image")

    # Display the annotated image
    # results['annotated_image_cv2'] is in BGR format from OpenCV
    annotated_img_rgb = cv2.cvtColor(results['annotated_image_cv2'], cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots()
    ax.imshow(annotated_img_rgb)
    ax.set_axis_off()
    st.pyplot(fig)

    # Optionally, list individual predictions if desired (they are already on the image)
    if results['predictions_list']:
        st.write("Details of Detections:")
        for i, pred_info in enumerate(results['predictions_list']):
            st.write(
                f"- Sign {i+1}: {pred_info['pred_label_name']} "
                f"(Confidence: {pred_info['confidence']*100:.2f}%)"
            )
    elif not results['predictions_list'] and uploaded_file: # Check if file was uploaded but no signs found
        st.info("No distinct traffic signs separated by white space were detected in the uploaded image.")


    # Display the K-Means segmentation of the original full image
    if results['segmented_original_img_cv2'] is not None:
        st.subheader("KMeans Segmentation of Original Image")
        fig2, ax2 = plt.subplots()
        # segmented_original_img_cv2 is a 2D array (labels), needs a colormap
        ax2.imshow(results['segmented_original_img_cv2'], cmap='nipy_spectral')
        ax2.set_axis_off()
        st.pyplot(fig2)
    else:
        st.info("Segmentation could not be performed (e.g., image was empty or not suitable).")

else:
    st.info("Upload an image (JPG, PNG, or PPM) to see predictions and segmentation.")