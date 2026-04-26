<div align="center">
  <h1>🌿 MaizeGuard AI</h1>
  <h3>Enterprise-Grade Computer Vision Pathology System for Precision Agriculture</h3>
  <p><i>MSc Artificial Intelligence Thesis Project | Open University of Kenya</i></p>
</div>

---

## 📌 Project Overview
**Enhancing Maize Disease Detection Using Image Preprocessing and Deep Learning Models**. 

This repository contains the complete experimental framework and commercial deployment prototype for predicting maize crop diseases using Deep Convolutional Neural Networks (CNNs). It serves as a comprehensive comparative analysis between lightweight Custom CNN architectures and massive deep-layered Transfer Learning models (ResNet50), fundamentally questioning the role of classical computer vision preprocessing in modern Deep Learning.

The highlight of this repository is **MaizeGuard AI**, a live, interactive agricultural Software-as-a-Service (SaaS) prototype that allows farmers and agronomists to mathematically diagnose crop pathology in real-time.

---

## 🚀 The Prototype: MaizeGuard AI
We successfully migrated the experimental research into a live, professional web application built on **Streamlit**. 

**Core Features:**
*   **Live Deep Learning Diagnosis:** Upload any photo of a maize leaf, and the fine-tuned ResNet50 model will execute immediate probabilistic inference to identify the disease.
*   **Commercial Agronomics Database:** The engine does not just predict the disease; it outputs an exhaustive Agronomic Profile. It provides farmers with **immediate quarantine actions**, exact **chemical fungicide prescriptions** (e.g., *Pyraclostrobin*), and **organic/cultural prevention tactics**.
*   **Explainable AI (xAI):** Trust is critical in medical/agricultural AI. Using automated **Grad-CAM backpropagation**, the application mathematically proves *exactly* which pixels the neural network used to determine the illness by rendering a fluorescent heatmap natively in the browser.

---

## 🔬 Scientific Methodology & Thesis Results

The codebase executed three highly controlled predictive experiments to isolate the effects of classical OpenCV preprocessing (HSV segmenting, CLAHE contrast enhancement, and Morphological isolation) against varying depths of neural architectures.

### Experimental Outcomes
| Experiment | Architecture | Data Condition | Accuracy | Insight |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 01** | Custom CNN | **Raw** Photos | 93.69% | The baseline established that lightweight models struggle slightly with chaotic agricultural backgrounds. |
| **Exp 02** | Custom CNN | **Preprocessed** | 91.82% | *Accuracy Dropped.* The harsh OpenCV masking destroyed valuable spatial context, confusing the shallow CNN. |
| **Exp 04** | ResNet50 | **Preprocessed** | **96.25%** | *Massive Success.* The immense depth (50 layers) of ResNet was able to completely ignore the preprocessing noise and utilize the heightened contrast to isolate the distinct pathogen geometries perfectly. |

### The "Deep Learning Entropy" Hypothesis
This thesis successfully proved that while classical image masking (OpenCV) is highly destructive to lightweight CNNs traversing spatial geometries, massive Transfer Learning architectures (ResNet50) natively transcend that entropy. ResNet50 utilized the synthetically highlighted lesions to achieve cutting-edge accuracy scores across F1, Precision, and Recall metrics.

---

## 💻 Installation & Usage

To run the codebase and the live presentation module locally on your machine:

**1. Clone and Install Dependencies:**
```powershell
git clone https://github.com/FRIEZEWANDABWA/smart-agriculture.git
cd smart-agriculture
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**2. Launch the MaizeGuard AI Presentation App:**
*(Note: You must have the `resnet50_maize_model.keras` weights downloaded into the `models/` directory, as heavy AI constraints prevent large weights from living natively on GitHub).*
```powershell
python -m streamlit run maizeguard_app/app.py
```

**3. Test the Web UI:**
Simply open your web browser to `http://localhost:8501`, upload a test leaf `.jpg`, and observe the live diagnostic pipeline and xAI Heatmap generation!

---

<div align="center">
  <p><b>Developed and Architected by Frieze Wandabwa</b></p>
</div>
