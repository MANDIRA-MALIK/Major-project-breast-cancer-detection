import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ========================
# Load Model
# ========================
model = tf.keras.models.load_model("model.keras", compile=False)

# ========================
# App Title
# ========================
st.title("Breast Cancer Detection + GradCAM 🔬")

# ========================
# Upload Image
# ========================
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ========================
# Image Preprocessing
# ========================
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ========================
# Grad-CAM Function
# ========================
def get_gradcam(img_array, model, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# ========================
# Display Results
# ========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success("Prediction: Malignant (Cancer)")
    else:
        st.success("Prediction: Benign (No Cancer)")

    # ========================
    # Grad-CAM
    # ========================
    try:
        # 👉 CHANGE layer name if needed
        last_conv_layer = "conv2d"

        heatmap = get_gradcam(img_array, model, last_conv_layer)

        # Resize heatmap
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)

        # Apply color map
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superimpose
        img = np.array(image.resize((224, 224)))
        superimposed_img = heatmap * 0.4 + img

        st.image(superimposed_img.astype(np.uint8), caption="Grad-CAM", use_column_width=True)

    except Exception as e:
        st.warning("Grad-CAM not working. Check layer name.")
