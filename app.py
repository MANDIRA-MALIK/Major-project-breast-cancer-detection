import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

st.title("Breast Cancer Detection + GradCAM 🔬")

# Load model
model = load_model("model.keras", compile=False)

# 🔥 GradCAM function
def get_gradcam(img_array, model, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap.numpy()

# Upload
file = st.file_uploader("Upload Image")

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    # Preprocess
    img_resized = img.resize((224,224))
    img_array = np.array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        st.error("⚠️ Malignant")
    else:
        st.success("✅ Benign")

    # 🔥 CHANGE THIS LAYER NAME
    heatmap = get_gradcam(img_array, model, layer_name="conv2d")

    # Resize heatmap
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    superimposed = heatmap * 0.4 + np.array(img)

    st.image(superimposed.astype("uint8"), caption="GradCAM 🔥")
