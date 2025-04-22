import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np
from huggingface_hub import hf_hub_download 

# Model repo mapping from Hugging Face
MODEL_REPOS = {
    "Tuberculosis": "monalika128/tb_model",
    "Brain Tumor": "monalika128/Brain_Tumor_Model",
    "Lung Cancer": "monalika128/LungCancer_Model",
    "Eye Disease": "monalika128/eye_disease_model",
    "COVID/Pneumonia": "monalika128/Pneumonia_Model",
    "Breast Cancer": "monalika128/breast_cancer_model"
}

# Class labels for each model
CLASS_LABELS = {
    "Tuberculosis": ["Normal", "Tuberculosis"],
    "Brain Tumor": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
    "Lung Cancer": ["Benign", "Malignant", "Normal"],
    "Eye Disease": ["Normal", "Cataract", "Diabetic Retinopathy", "Glaucoma"],
    "COVID/Pneumonia": ["COVID", "Normal", "Pneumonia"],
    "Breast Cancer": ["Benign", "Malignant", "Normal"]
}

@st.cache_resource
def load_model(repo_id):
    model_path = hf_hub_download(repo_id=repo_id, filename="model.h5")
    return tf.keras.models.load_model(model_path)

# Streamlit App UI
st.title("🧠 Smart Disease Predictor")
st.markdown("Upload a medical image and select a disease type to detect using AI.")

# Disease selection
disease = st.selectbox("🩺 Select Disease Type", list(MODEL_REPOS.keys()))

# File uploader
uploaded_image = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

# When image and disease are both selected
if uploaded_image and disease:
    try:
        # Try to open and display the image
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        image = image.resize((224, 224))
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        # Load model
        with st.spinner("🔄 Loading model..."):
            model = load_model(MODEL_REPOS[disease])

        # Predict
        prediction = model.predict(image_array)
        predicted_class = CLASS_LABELS[disease][np.argmax(prediction)]
        st.success(f"✅ Predicted Class: **{predicted_class}**")

    except UnidentifiedImageError:
        st.error("❌ Unable to read the image. Please upload a valid medical image.")
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
