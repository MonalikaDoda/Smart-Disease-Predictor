import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# --- Constants ---
MODEL_CONFIG = {
    "Tuberculosis": {
        "repo": "monalika128/tb_model",
        "labels": ["Normal", "Tuberculosis"],
        "size": (224, 224)
    },
    # Add other diseases here...
}

# --- UI Setup ---
st.set_page_config(page_title="MedScan AI", page_icon="🏥", layout="centered")
st.title("🔍 MedScan AI")
st.markdown("Upload a medical scan for instant analysis")

# --- Core Function ---
def analyze_image(image, disease):
    """Process image and return prediction"""
    model = load_model(MODEL_CONFIG[disease]["repo"])
    img_array = np.array(image.resize(MODEL_CONFIG[disease]["size"])) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return MODEL_CONFIG[disease]["labels"][np.argmax(pred)], np.max(pred) * 100

@st.cache_resource
def load_model(repo_id):
    return tf.keras.models.load_model(hf_hub_download(repo_id=repo_id, filename="model.h5"))

# --- Streamlit UI ---
disease = st.selectbox("Select Disease", list(MODEL_CONFIG.keys()))
uploaded_file = st.file_uploader("Choose a scan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Scan", use_container_width=True)
    
    if st.button("Analyze Now"):
        with st.spinner("🔬 Analyzing..."):
            prediction, confidence = analyze_image(image, disease)
            
            # Display Results
            if prediction == "Normal":
                st.success(f"✅ Normal (Confidence: {confidence:.1f}%)")
            else:
                st.error(f"🚨 {prediction} Detected (Confidence: {confidence:.1f}%)")
            
            # Confidence Meter
            st.progress(int(confidence), text=f"Detection Confidence: {confidence:.1f}%")

# --- Minimal Patient Info (Optional) ---
with st.expander("🆔 Optional Patient Info"):
    st.text_input("Case Notes", help="For your reference only")

# --- How It Works ---
st.markdown("---")
st.info("💡 How it works: AI analyzes your scan without storing any data.")