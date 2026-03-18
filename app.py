import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# load model
model = tf.keras.models.load_model("model.h5", compile=False)

def predict(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return f"Cancer Detected ❌ ({pred:.2f})"
    else:
        return f"No Cancer ✅ ({1-pred:.2f})"

app = gr.Interface(fn=predict, inputs="image", outputs="text")
app.launch()
