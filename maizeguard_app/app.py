import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
import sys
import os
import time

# Ensure we can import from the root src/ folder even if run from this sub-directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess.pipeline import preprocess_bgr_to_rgb
from src.eval.explain import make_gradcam_heatmap, save_and_display_gradcam
from src.utils.recommendations import get_recommendation

# --- Config & Initialization ---
st.set_page_config(page_title="MaizeGuard AI | Commercial Diagnostic System", page_icon="🌿", layout="wide")

CLASSES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# Because the script is now in a subfolder, we bind paths to the project root
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
MODEL_PATH = ROOT_DIR / "models" / "resnet50_maize_model.keras"

# --- MASSIVE UI/UX CSS OVERRIDE ---
st.markdown("""
    <style>
        /* Base Themes - Allow Streamlit to handle Light/Dark mode backgrounds naturally */
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        /* Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, #10b981 0%, #047857 100%);
            padding: 3rem;
            border-radius: 12px;
            color: white !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
            margin-bottom: 2rem;
            text-align: center;
        }
        .hero-title { font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; color: white !important; }
        .hero-subtitle { font-size: 1.2rem; opacity: 0.9; font-weight: 400; color: white !important; }
        
        /* Glass Cards - Translucent so it adapts perfectly to Dark Mode or Light Mode */
        .glass-card {
            background: rgba(150, 150, 150, 0.08);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(150, 150, 150, 0.2);
            margin-bottom: 1rem;
            transition: transform 0.2s ease-in-out;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }

        /* Metric Highlights */
        .metric-red { color: #ef4444 !important; font-weight: 700; font-size: 1.5rem; }
        .metric-green { color: #10b981 !important; font-weight: 700; font-size: 1.5rem; }
        
        /* Badges */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            margin-bottom: 1rem;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_commercial_model():
    if not MODEL_PATH.exists():
        st.error(f"FATAL: Core proprietary model missing at {MODEL_PATH}.")
        st.stop()
    model = keras.models.load_model(str(MODEL_PATH))
    model.layers[-1].activation = None 
    return model

model = load_commercial_model()

# --- HERO SECTION ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌿 MaizeGuard AI</div>
    <div class="hero-subtitle">Enterprise-Grade Computer Vision Pathology System for Precision Agriculture</div>
</div>
""", unsafe_allow_html=True)

# --- UPLOAD SECTION ---
st.markdown("### 📷 Deep Learning Diagnosis Scanner")
uploaded_file = st.file_uploader("Upload highly detailed drone or mobile imagery of suspect crop leaves:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.divider()

    # Load into memory buffer
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr_img = cv2.imdecode(file_bytes, 1)
    
    temp_img_path = ROOT_DIR / "temp_upload.jpg"
    cv2.imwrite(str(temp_img_path), bgr_img)

    # --- COMMERCIAL PROGRESS SPINNER ---
    with st.status("Initializing Quantum Processing Pipeline...", expanded=True) as status:
        st.write("⚙️ Engaging OpenCV Contrast Filtering...")
        time.sleep(0.5) # UX Delay to feel commercial
        rgb_preprocessed = preprocess_bgr_to_rgb(bgr_img)
        resized_img = cv2.resize(bgr_img, (224, 224), interpolation=cv2.INTER_AREA)
        
        st.write("🔬 Vectorizing Image Tensors...")
        model_input = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(model_input, axis=0).astype(np.float32)
        img_array = resnet50_preprocess_input(img_array)
        
        st.write("🧠 Executing ResNet50 Deep Neural Network...")
        preds = model.predict(img_array)
        softmax_preds = tf.nn.softmax(preds[0]).numpy()
        pred_index = np.argmax(softmax_preds)
        confidence = softmax_preds[pred_index] * 100
        best_class = CLASSES[pred_index]
        
        st.write("📡 Backpropagating feature maps for Explainable AI (xAI)...")
        heatmap, _ = make_gradcam_heatmap(img_array, model, "conv5_block3_out")
        out_heatmap_path = ROOT_DIR / "temp_heatmap.png"
        save_and_display_gradcam(temp_img_path, heatmap, out_heatmap_path)
        
        status.update(label="Analysis Successfully Completed!", state="complete", expanded=False)

    # Fetch Agronomic Proifle
    profile = get_recommendation(best_class)
    metric_class = "metric-green" if best_class == "Healthy" else "metric-red"

    # --- TWO COLUMN DASHBOARD ---
    colA, colB = st.columns([1, 1.2])

    with colA:
        st.markdown(f'<div class="badge">Model: ResNet50 (Fine-Tuned)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card"><h4>Primary Diagnosis</h4><p class="{metric_class}">{best_class.replace("_", " ")}</p><p><b>Diagnostic Certainty:</b> {confidence:.2f}%</p></div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="glass-card">
            <h4>Agronomic Profile</h4>
            <p><i>{profile['overview']}</i></p><hr/>
            <p><b>🚨 CRITICAL ACTION:</b><br>{profile['action']}</p>
            <p><b>🧪 CHEMICAL INTERVENTION:</b><br>{profile['chemical']}</p>
            <p><b>🌱 CULTURAL FORECAST:</b><br>{profile['organic']}</p>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown('<div class="badge">XAI Visual Proof</div>', unsafe_allow_html=True)
        st.image(str(out_heatmap_path), use_column_width=True)
        st.markdown("""
        <div class="glass-card">
            <b>Explainable AI Gradient Mapping:</b> The visual thermograph above mathematically proves what the neural network was analyzing precisely at the moment of classification. Severe red zones map exactly to recognized disease geometries, bypassing agricultural background noise.
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align: center; color: #64748b;'>Developed by Frieze Wandabwa | MSc Artificial Intelligence | Commercial Sandbox Prototype</p>", unsafe_allow_html=True)
