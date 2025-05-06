import streamlit as st
import matplotlib.pyplot as plt
from backend import load_model, predict_and_segment, class_names

st.title("Traffic Sign Recognition Demo (GTSRB)")

# Model loading (cache for performance)
@st.cache_resource
def get_model():
    return load_model("/home/theodoros/projects/Traffic_signs/model_weights.pth", num_classes=43)

model = get_model()
device = "cpu"  

uploaded_file = st.file_uploader("Choose a traffic sign image...", type=['jpg', 'jpeg', 'png', 'ppm'])
if uploaded_file is not None:
    img_bytes = uploaded_file.read()
    result = predict_and_segment(model, img_bytes, device=device)

    st.subheader(f"Prediction: {result['pred_label_name']} ({result['confidence']*100:.2f}%)")

    fig, ax = plt.subplots()
    img = result['image'][..., ::-1]  # Convert BGR to RGB for matplotlib

    ax.imshow(img)
    x1, y1, x2, y2 = result['bbox']
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', linewidth=2)
    ax.add_patch(rect)
    ax.set_axis_off()
    st.pyplot(fig)

    st.subheader("KMeans Segmentation")
    fig2, ax2 = plt.subplots()
    ax2.imshow(result['segmented_img'], cmap='nipy_spectral')
    ax2.set_axis_off()
    st.pyplot(fig2)
else:
    st.info("Upload an image (JPG, PNG, or PPM) to see predictions and segmentation.")