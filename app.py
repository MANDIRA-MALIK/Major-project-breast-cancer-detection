import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import model_from_json

# ---------------------------
# LOAD MODEL (JSON + WEIGHTS)
# ---------------------------
@st.cache_resource
def load_my_model():
    with open("model.json", "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights("model.weights.h5")
    return model

model = load_my_model()

# ---------------------------
# IMAGE PREPROCESS
# ---------------------------
def preprocess_image(image):
    img = image.resize((224, 224))  # adjust if needed
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------------------
# GRADCAM FUNCTION
# ---------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
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
    return heatmap.numpy()

# ---------------------------
# OVERLAY HEATMAP
# ---------------------------
def overlay_heatmap(heatmap, image):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + image
    return np.uint8(superimposed_img)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Breast Cancer Detection + GradCAM 🔬")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.error(f"Cancer Detected ❌ (Confidence: {prediction:.2f})")
    else:
        st.success(f"No Cancer Detected ✅ (Confidence: {1-prediction:.2f})")

    # ---------------------------
    # GRADCAM
    # ---------------------------
    try:
        last_conv_layer = "conv2d"  # ⚠️ change if needed

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)

        original = np.array(image.resize((224, 224)))
        cam_image = overlay_heatmap(heatmap, original)

        st.subheader("GradCAM Output")
        st.image(cam_image, use_column_width=True)

    except Exception as e:
        st.warning("GradCAM not working. Check layer name.")
        st.text(str(e))
