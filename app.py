# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from focal_loss import FocalLoss

# Load model only once
@st.cache_resource
def load_model_once():
    return load_model("best_brain_tumor_model.h5", custom_objects={"FocalLoss": FocalLoss(gamma=2.0)})

model = load_model_once()
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']  # Update if needed
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classifier")
st.write("Upload an MRI scan and the model will predict the type of brain tumor.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: {predicted_class.upper()}")
    st.info(f"Confidence: {confidence:.2%}")
