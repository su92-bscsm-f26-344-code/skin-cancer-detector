import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Page Config
st.set_page_config(page_title="Skin Cancer Detection AI", page_icon="🔬", layout="centered")

# Dark Neumorphism CSS Styling (Aap ki pasand ka luxury look)
st.markdown("""
    <style>
    .stApp { background-color: #21242b; color: #e2e8f0; font-family: 'Poppins', sans-serif; }
    .hero-title { text-align: center; font-size: 2.2rem; font-weight: 800; color: #dce4f2; margin-bottom: 5px; }
    .hero-subtitle { text-align: center; color: #8b95a5; font-size: 0.95rem; margin-bottom: 25px; }
    
    .stButton>button {
        width: 100%; background: linear-gradient(145deg, #3b82f6, #1d4ed8);
        color: white; border: none; border-radius: 50px; padding: 12px;
        font-weight: 700; box-shadow: 6px 6px 14px #15171c, -6px -6px 14px #2d313a;
    }
    .result-card {
        background: #1c1f26; border-radius: 20px; padding: 20px; text-align: center;
        margin-top: 20px; box-shadow: inset 4px 4px 10px #13151a, inset -4px -4px 10px #252932;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">🔬 Skin Lesion & Cancer Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Upload a skin image to analyze risk using a trained Convolutional Neural Network (CNN).</div>', unsafe_allow_html=True)

# Load CNN Model
@st.cache_resource
def load_csu_model():
    return load_model('skin_cancer_model.keras')

model = load_csu_model()

# Image Uploader
uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Image'):
        # Preprocess image for CNN
        image = image.resize((180, 180))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        prediction = model.predict(img_array)
        score = tf.nn.softmax(prediction[0]) if prediction.ndim > 1 else prediction[0]
        
        st.markdown(f"""
            <div class="result-card">
                <h3 style="color: #60a5fa;">Analysis Complete</h3>
                <p style="font-size: 1.1rem; font-weight: 600;">Confidence Score Calculated Successfully.</p>
            </div>
        """, unsafe_allow_html=True)