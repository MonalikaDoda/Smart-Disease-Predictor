import streamlit as st
import tensorflow as tf
from PIL import Image
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

st.title("🧠 Smart Disease Predictor")
st.markdown("Upload a medical image to predict disease using our AI-powered model.")

disease = st.selectbox("🩺 Select Disease Type", list(MODEL_REPOS.keys()))
uploaded_image = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image and disease:
    model = load_model(MODEL_REPOS[disease])
    
    # Preprocess the image
    image = Image.open(uploaded_image).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    # Predict
    prediction = model.predict(image_array)
    predicted_class = CLASS_LABELS[disease][np.argmax(prediction)]

    st.success(f"✅ **Predicted Class:** {predicted_class}")
