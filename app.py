
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
# Import the FocalLoss class definition directly
# from focal_loss import FocalLoss # Removed this import

# Removed global FocalLoss definition


# Load model with custom objects
# @st.cache_resource # Removed caching decorator
def load_model_with_custom_objects():
    # Define the FocalLoss class directly inside the function
    class FocalLoss(tf.keras.losses.Loss):
        """
        Focal Loss for addressing class imbalance in medical datasets

        Formula:
            Focal Loss = -α(1 - p_t)^γ * log(p_t)

        Where:
        - p_t is the predicted probability for the true class.
        - α is the balancing factor to give more weight to underrepresented classes.
        - γ (gamma) reduces the loss contribution from easy examples and focuses on hard examples.
        """

        # Constructor: initialize alpha, gamma, and any additional kwargs
        def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
            super().__init__(**kwargs)
            self.alpha = alpha
            self.gamma = gamma

        # This function defines the actual focal loss calculation
        def call(self, y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            cross_entropy = -y_true * tf.math.log(y_pred)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_weight = self.alpha * tf.pow(1 - p_t, self.gamma)
            focal_loss = focal_weight * cross_entropy
            return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))

        # Save configuration to support model saving/loading
        def get_config(self):
            config = super().get_config()
            config.update({
                'alpha': self.alpha,
                'gamma': self.gamma
            })
            return config

    # Pass the alpha and gamma values that were used during training
    return load_model("best_brain_tumor_model.h5", custom_objects={"FocalLoss": FocalLoss(alpha=1.0, gamma=2.0)})


model = load_model_with_custom_objects()
# Update CLASS_NAMES to match the order from the training data generator
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
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
    # Use the same preprocessing function as during training
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: {predicted_class.upper()}")
    st.info(f"Confidence: {confidence:.2%}")