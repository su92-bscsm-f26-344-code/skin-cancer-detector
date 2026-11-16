import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title='Skin Cancer Detection AI', page_icon='🔬', layout='centered'
)

# Dark Neumorphism CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    .stApp { background-color: #21242b; font-family: 'Poppins', sans-serif; color: #e2e8f0; }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .cup-circle {
        width: 120px; height: 120px; margin: 10px auto 20px auto; border-radius: 50%;
        background: #252932; display: flex; align-items: center; justify-content: center; font-size: 55px;
        box-shadow: 12px 12px 24px #16181d, -12px -12px 24px #2c3039;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .hero-title { text-align: center; font-size: 2.3rem; font-weight: 800; color: #dce4f2; margin-bottom: 10px; }
    .hero-subtitle { text-align: center; color: #8b95a5; font-size: 0.95rem; margin-bottom: 25px; }
    
    .stButton>button {
        width: 100%; background: linear-gradient(145deg, #3b82f6, #1d4ed8); color: white;
        border: none; border-radius: 50px; padding: 14px; font-weight: 700; font-size: 1.05rem;
        box-shadow: 8px 8px 18px #15171c, -6px -6px 16px #2d313a;
    }
    .result-card {
        background: #1c1f26; border-radius: 20px; padding: 22px; text-align: center; margin-top: 20px;
        box-shadow: inset 6px 6px 14px #13151a, inset -6px -6px 14px #252932;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="cup-circle">🔬</div>
    <div class="hero-title">Skin Lesion & Cancer Classifier</div>
    <div class="hero-subtitle">Upload a skin image to analyze risk using a trained Convolutional Neural Network (CNN).</div>
""",
    unsafe_allow_html=True,
)


# Load Model Safely with Error Handling
@st.cache_resource
def get_model():
  try:
    return load_model('skin_cancer_model.keras', compile=False)
  except Exception as e:
    return None


model = get_model()

if model is None:
  st.error(
      "⚠️ Model load nahi ho saka! Kindly ensure karein ke 'skin_cancer_model.keras'"
      ' file GitHub par mojood hai aur requirements.txt mein tensorflow version'
      ' theek hai.'
  )
else:
  uploaded_file = st.file_uploader(
      'Choose a skin image...', type=['jpg', 'jpeg', 'png']
  )

  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Skin Image', use_column_width=True)

    if st.button('Analyze Image'):
      with st.spinner('Analyzing image through CNN model...'):
        image = image.resize((180, 180))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        confidence = float(np.max(prediction[0])) * 100

        st.markdown(
            f"""
                <div class="result-card">
                    <h3 style="color: #60a5fa; margin-bottom: 5px;">Analysis Complete</h3>
                    <p style="font-size: 1.1rem; color: #e2e8f0; font-weight: 600;">Confidence Score: {confidence:.2f}%</p>
                    <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 5px;">Note: This is an AI-based exploratory tool, always consult a medical professional for diagnosis.</p>
                </div>
            """,
            unsafe_allow_html=True,
        )
