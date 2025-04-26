import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import datetime
from io import BytesIO

# Model configuration
MODEL_REPOS = {
    "Tuberculosis": "monalika128/tb_model",
    "Brain Tumor": "monalika128/Brain_Tumor_Model",
    "Lung Cancer": "monalika128/LungCancer_Model",
    "Eye Disease": "monalika128/eye_disease_model",
    "COVID/Pneumonia": "monalika128/Pneumonia_Model",
    "Breast Cancer": "monalika128/breast_cancer_model"
}

CLASS_LABELS = {
    "Tuberculosis": ["Normal", "Tuberculosis"],
    "Brain Tumor": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
    "Lung Cancer": ["Benign", "Malignant", "Normal"],
    "Eye Disease": ["Normal", "Cataract", "Diabetic Retinopathy", "Glaucoma"],
    "COVID/Pneumonia": ["COVID", "Normal", "Pneumonia"],
    "Breast Cancer": ["Benign", "Malignant", "Normal"]
}

TARGET_SIZES = {
    "Tuberculosis": (224, 224),
    "Brain Tumor": (150, 150),
    "Lung Cancer": (144, 144),
    "Eye Disease": (224, 224),
    "COVID/Pneumonia": (224, 224),
    "Breast Cancer": (224, 224)
}

# UI Config
st.set_page_config(
    page_title="AI Disease Detector",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .report-title {
        font-size: 24px;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 30px;
    }
    .patient-info {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-card {
        border-left: 5px solid #3b82f6;
        padding: 15px;
        background-color: white;
        border-radius: 5px;
        margin: 10px 0;
    }
    .positive {
        border-left-color: #ef4444;
        background-color: #fef2f2;
    }
    .negative {
        border-left-color: #10b981;
        background-color: #ecfdf5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(repo_id):
    model_path = hf_hub_download(repo_id=repo_id, filename="model.h5")
    return tf.keras.models.load_model(model_path)

def generate_report(patient_data, image, prediction, confidence):
    """Generate PDF report (simplified for demo)"""
    from fpdf import FPDF
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Report header
    pdf.cell(200, 10, txt="Medical Imaging Report", ln=1, align='C')
    pdf.ln(10)
    
    # Patient info
    pdf.cell(200, 10, txt=f"Patient ID: {patient_data.get('id', 'N/A')}", ln=1)
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=1)
    pdf.ln(5)
    
    # Save image temporarily
    image_path = "temp_image.jpg"
    image.save(image_path)
    pdf.image(image_path, w=100)
    
    # Results
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Diagnosis Results", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Condition: {prediction}", ln=1)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=1)
    
    report_bytes = BytesIO(pdf.output(dest='S').encode('latin-1'))
    return report_bytes

# Main App
st.title("🏥 AI Disease Detector")
st.markdown("Upload a medical scan for automated analysis")

# Disease selection
disease = st.selectbox("Select Disease Type", list(MODEL_REPOS.keys()))

# Optional patient info
with st.expander("➕ Add Patient Information (Optional)"):
    patient_data = {
        "id": st.text_input("Patient ID"),
        "age": st.number_input("Age", min_value=1, max_value=120),
        "gender": st.selectbox("Gender", ["", "Male", "Female", "Other"])
    }

# Image upload
uploaded_image = st.file_uploader("Upload Medical Scan", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Scan", use_container_width=True)
    
    if st.button("Analyze Scan"):
        with st.spinner("Processing..."):
            # Preprocessing
            target_size = TARGET_SIZES[disease]
            img_array = np.array(image.resize(target_size)) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction
            model = load_model(MODEL_REPOS[disease])
            pred = model.predict(img_array)
            confidence = np.max(pred) * 100
            prediction = CLASS_LABELS[disease][np.argmax(pred)]
            
            # Display results
            result_class = "positive" if prediction != "Normal" else "negative"
            st.markdown(f"""
            <div class="result-card {result_class}">
                <h3>Analysis Result</h3>
                <p><strong>Condition:</strong> {prediction}</p>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate report
            report = generate_report(patient_data, image, prediction, confidence)
            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name=f"medical_report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

# How it works section
with st.expander("ℹ️ How this works"):
    st.markdown("""
    - AI analyzes medical images for specific conditions
    - Patient information is optional and never affects results
    - Reports include scan date/time for documentation
    """)