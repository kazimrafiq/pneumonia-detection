
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = [
    "Normal",
    "Lung Opacity",
    "No Lung Opacity / Not Normal"
]

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩻",
    layout="wide"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "efficientnetb0_final.keras",
        compile=False
    )

st.title("Pneumonia Detection")
st.write(
    "Upload a chest X-ray image for experimental "
    "three-class classification."
)

model = load_model()
st.success("Model loaded successfully.")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-ray",
        width=350
    )

    if st.button("Predict"):
        processed_image = image.convert("L").convert("RGB")
        processed_image = processed_image.resize(
            (160, 160),
            Image.Resampling.BOX
        )

        image_array = np.asarray(
            processed_image,
            dtype=np.float32
        )
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_class = CLASS_NAMES[
            int(np.argmax(predictions))
        ]

        st.subheader("Prediction")
        st.success(predicted_class)

        st.subheader("Class Probabilities")

        for name, probability in zip(CLASS_NAMES, predictions):
            st.write(f"{name}: {probability:.2%}")
            st.progress(float(probability))

st.warning(
    "Educational demonstration only. This model is not "
    "clinically validated and must not be used for medical "
    "diagnosis or treatment decisions."
)
