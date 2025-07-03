import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="FakeFinder", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")  # make sure this file is in your project root

model = load_model()

# UI
st.title("FakeFinder: Real vs Fake Facial Image Classifier")
st.write("Upload a facial image to check if it's real or fake")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((160, 160))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    label = "Real" if np.argmax(prediction) == 1 else "Fake"

    st.subheader("Prediction Result:")
    st.success(f"This image is classified as: **{label}**")
    st.write("Confidence:", f"Real: {prediction[1]:.2%} | Fake: {prediction[0]:.2%}")
