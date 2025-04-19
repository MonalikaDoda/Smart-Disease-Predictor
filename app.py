import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page title
st.set_page_config(page_title="Smart Disease Predictor")

# Title
st.title("🧠🫁 Smart Disease Prediction from Medical Images")

# Load all models
@st.cache_resource
def load_models():
    models = {
        'brain': tf.keras.models.load_model('models/Brain_Tumor_Model.h5'),
        'pneumonia': tf.keras.models.load_model('models/Pneumonia_Model.h5'),
        'lung': tf.keras.models.load_model('models/LungCancer_Model.h5'),
        'eye': tf.keras.models.load_model('models/eye_disease_model.h5'),
        'tb': tf.keras.models.load_model('models/tb_model.h5'),
        'breast': tf.keras.models.load_model('models/breast_cancer_model.h5'),
    }
    return models

models = load_models()

# Disease labels
labels = {
    'brain': ['glioma', 'meningioma', 'no_tumor', 'pituitary'],
    'pneumonia': ['COVID', 'NORMAL', 'PNEUMONIA'],
    'lung': ['benign', 'malignant', 'normal'],
    'eye': ['normal', 'cataract', 'diabetic_retinopathy', 'glaucoma'],
    'tb': ['Normal', 'Tuberculosis'],
    'breast': ['benign', 'malignant', 'normal']
}

# Simple detection logic based on image shape and file name
def detect_disease_type(file_name, image):
    name = file_name.lower()
    if "brain" in name:
        return 'brain'
    elif "lung" in name:
        return 'lung'
    elif "pneumonia" in name or "covid" in name:
        return 'pneumonia'
    elif "eye" in name or "retina" in name:
        return 'eye'
    elif "tb" in name or "chest" in name:
        return 'tb'
    elif "breast" in name:
        return 'breast'
    else:
        # Fallback by image shape (if needed)
        if image.size[0] < 200:
            return 'eye'
        elif image.size[0] > 300 and image.size[1] > 300:
            return 'brain'
        else:
            return 'pneumonia'

# Upload image
uploaded_file = st.file_uploader("📤 Upload a medical image (e.g., chest x-ray, MRI, etc.)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Auto-detect disease
    disease = detect_disease_type(uploaded_file.name, image)
    st.info(f"Detected Disease Type: **{disease.upper()}**")

    # Preprocess image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = models[disease].predict(img_array)
    pred_class = labels[disease][np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    st.success(f"Prediction: **{pred_class.upper()}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
