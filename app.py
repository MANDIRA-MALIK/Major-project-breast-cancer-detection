import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -------------------------
# LOAD MODEL
# -------------------------
model = tf.keras.models.load_model("model.h5", compile=False)

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------
# GRADCAM
# -------------------------
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

# -------------------------
# OVERLAY
# -------------------------
def overlay_heatmap(heatmap, image):
    image = np.array(image.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + image

    return np.uint8(superimposed)

# -------------------------
# MAIN FUNCTION
# -------------------------
def predict(image):
    img_array = preprocess(image)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        result = f"Cancer Detected ❌ ({pred:.2f})"
    else:
        result = f"No Cancer ✅ ({1-pred:.2f})"

    # GradCAM
    try:
        last_conv_layer = "conv2d"  # ⚠️ change if needed
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
        cam_image = overlay_heatmap(heatmap, image)
    except:
        cam_image = image

    return result, cam_image

# -------------------------
# GRADIO UI
# -------------------------
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Text(label="Prediction"),
        gr.Image(label="GradCAM")
    ],
    title="Breast Cancer Detection + GradCAM 🔬"
)

app.launch()
