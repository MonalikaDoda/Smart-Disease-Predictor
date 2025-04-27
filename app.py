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

# Target input size per model
TARGET_SIZES = {
    "Tuberculosis": (224, 224),
    "Brain Tumor": (150, 150),
    "Lung Cancer": (150, 150),  # Corrected to 150x150
    "Eye Disease": (224, 224),
    "COVID/Pneumonia": (224, 224),
    "Breast Cancer": (224, 224)
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

# File uploader with explicit accept parameter
uploaded_image = st.file_uploader(
    "📷 Upload an Image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="Please upload a single medical scan image"
)

# Prediction button - Only enabled when image is uploaded
predict_button = st.button("🔍 Predict", disabled=not uploaded_image)

if predict_button and uploaded_image:
    try:
        # Open and display image
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", width=None)

        # Resize and preprocess image
        target_size = TARGET_SIZES[disease]
        image = image.resize(target_size)
        image_array = np.array(image).astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # ✅ Unified for all diseases

        # Load model and predict with progress
        with st.spinner("🧠 Loading model and analyzing..."):
            model = load_model(MODEL_REPOS[disease])
            prediction = model.predict(image_array)

            # Get confidence scores
            confidence = np.max(prediction) * 100
            predicted_class = CLASS_LABELS[disease][np.argmax(prediction)]

            # Display results
            st.success(f"✅ **Prediction:** {predicted_class}")
            st.info(f"🔢 **Confidence:** {confidence:.2f}%")
            
            # Disclaimer for user
            st.warning("⚠️ **Disclaimer:** Please consult a medical professional for final diagnosis.")

            # Show full probability distribution (optional)
            with st.expander("📊 Detailed probabilities"):
                for class_name, prob in zip(CLASS_LABELS[disease], prediction[0]):
                    st.progress(float(prob), text=f"{class_name}: {prob*100:.2f}%")

    except UnidentifiedImageError:
        st.error("❌ Invalid image format. Please upload a valid medical JPG/PNG scan.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.stop()
