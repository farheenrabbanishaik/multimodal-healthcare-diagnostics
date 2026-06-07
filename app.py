import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

st.set_page_config(page_title="Multi-Modal System for Healthcare Diagnostics", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# UI STYLING
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #00BFFF;
}
.sub-text {
    font-size: 18px;
    color: #AAAAAA;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center; color: #00BFFF;'>
 Multi-Modal System for Healthcare Diagnostics
</h1>
<p style='text-align: center; font-size:18px; color:gray;'>
Deep Learning-based system for automated detection of 
<b>Diabetic Retinopathy, Lung Diseases, and Heart Abnormalities</b> 
with Explainable AI (Grad-CAM & Integrated Gradients)
</p>
""", unsafe_allow_html=True)
st.markdown('<p class="sub-text">Clinical decision support system for multi-modal disease detection with interpretable deep learning models</p>', unsafe_allow_html=True)

st.divider()

st.sidebar.title("🩺 System Overview")

st.sidebar.markdown("""
###  Multi-Modal System

This platform analyzes multiple types of medical data:
- Retinal Images (Eye)
- Chest X-ray Images (Lung)
- ECG Signals (Heart)

---

### 🔍 Models Used
- ResNet18 (Image-based Diagnosis)
- 1D CNN (ECG Signal Analysis)

---

### ⚙️ System Features
- Transfer Learning
- Focal Loss (Handles class imbalance)
- Grad-CAM (Image Explainability)
- Integrated Gradients (Signal Explainability)

---

### ⚡ Clinical Decision Support
- Confidence-based predictions
- risk categorization
- Clinical recommendation (Routine Checkup / Refer / Retest)

---

### 📊 Output Provided
- Disease Prediction
- Confidence Score
- Visual Explanation
- Decision Support Recommendation
""")

# UTILITIES

def load_image(file):
    return Image.open(file).convert("RGB")

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])
    return transform(img).unsqueeze(0).to(device)

def softmax_probs(logits):
    return F.softmax(logits, dim=1).detach().cpu().numpy()[0]


def decision_support(module, predicted_class, confidence):
    
    # EYE (DR) 
    if module == "DR":
        if confidence < 0.60:
            return "⚠️ Low confidence → Retest recommended"

        elif predicted_class in ["Severe", "Proliferative"]:
            return "🚨 High Risk → Refer to Ophthalmologist"

        elif predicted_class in ["Moderate"]:
            return "⚠️ Monitor Closely + Follow-up"

        else:
            return "✅ Low Risk → Routine Checkup"

    # LUNG
    elif module == "LUNG":
        if confidence < 0.60:
            return "⚠️ Uncertain → Radiologist review needed"

        elif predicted_class in ["COVID", "Viral Pneumonia", "Lung Opacity"]:
            return "🚨 Abnormal → Immediate clinical attention"

        else:
            return "✅ Normal Lung"

    # HEART 
    elif module == "HEART":
        if confidence < 0.40:
            return "⚠️ Uncertain ECG → Re-test required"

        elif predicted_class != "Normal":
            return "🚨 Abnormal ECG → Cardiologist consultation"

        else:
            return "✅ Normal ECG"

    return "No decision"

# GRADCAM

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        pooled = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(len(pooled)):
            activations[i] *= pooled[i]

        heatmap = torch.sum(activations, dim=0)
        heatmap = F.relu(heatmap)

        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)

        return heatmap.cpu().numpy()

# FIXED SIZE FUNCTION
def overlay_heatmap(img, heatmap):
    img = np.array(img)
    h, w, _ = img.shape

    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay

# LOAD MODELS

@st.cache_resource
def load_dr_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load("best_dr_resnet18_focal.pth", map_location=device))
    model.to(device).eval()
    return model

@st.cache_resource
def load_lung_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load("lung_resnet18_model.pth", map_location=device))
    model.to(device).eval()
    return model

@st.cache_resource
def load_heart_model():
    class ECGModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1,32,5,padding=2)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32,64,5,padding=2)
            self.bn2 = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64,128,3,padding=1)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool = nn.MaxPool1d(2)
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(128*23,128)
            self.fc2 = nn.Linear(128,5)

        def forward(self,x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            return x

    model = ECGModel()
    model.load_state_dict(torch.load("heart_ecg_model.pth", map_location=device))
    model.to(device).eval()
    return model

dr_model = load_dr_model()
lung_model = load_lung_model()
heart_model = load_heart_model()

# UI TABS

tab1, tab2, tab3 = st.tabs([
    "👁️ Eye (DR)",
    "🫁 Lung",
    "❤️ Heart"
])

# DR TAB

with tab1:
    st.header("Diabetic Retinopathy Detection (Fundus Imaging)")
    st.caption("Upload a retinal fundus image to detect the severity of diabetic retinopathy using a deep learning model with Grad-CAM-based visual explanation.")

    file = st.file_uploader("Upload Fundus Image", type=["jpg","png","jpeg"])


    if file:
        img = load_image(file)
        tensor = preprocess_image(img)

        with torch.no_grad():
            logits = dr_model(tensor)

        probs = softmax_probs(logits)
        classes = ["No DR","Mild","Moderate","Severe","Proliferative"]
        idx = np.argmax(probs)

        cam = GradCAM(dr_model, dr_model.layer4[-1].conv2)
        heatmap = cam.generate(tensor)
        overlay = overlay_heatmap(img, heatmap)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Uploaded Image", width=350)

        with col2:
            st.markdown("### 🧾 Prediction")
            prediction = classes[idx]
            confidence = probs[idx]

            st.success(f"{prediction}")
            st.write(f"Confidence: {confidence:.2f}")

            decision = decision_support("DR", prediction, confidence)

            st.markdown("### Clinical Decision Support")
            st.info(decision)

        st.markdown("### 🔥 Grad-CAM Explanation")
        st.caption("Highlighted regions indicate where the model focused while making the prediction.")
        st.image(overlay, width=350)

# LUNG TAB

with tab2:
    st.header("Lung Disease Detection (Chest X-ray)")

    file = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"], key="lung")

    if file:
        img = load_image(file)
        tensor = preprocess_image(img)

        with torch.no_grad():
            logits = lung_model(tensor)

        probs = softmax_probs(logits)
        classes = ["COVID","Lung Opacity","Normal","Viral Pneumonia"]
        idx = np.argmax(probs)

        cam = GradCAM(lung_model, lung_model.layer4[-1].conv2)
        heatmap = cam.generate(tensor)
        overlay = overlay_heatmap(img, heatmap)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Uploaded X-ray", width=350)

        with col2:
            st.markdown("### 🧾 Prediction")
            prediction = classes[idx]
            confidence = probs[idx]

            st.success(f"{prediction}")
            st.write(f"Confidence: {confidence:.2f}")

            decision = decision_support("LUNG", prediction, confidence)

            st.markdown("### Clinical Decision Support")
            st.info(decision)

        st.markdown("### 🔥 Grad-CAM Explanation")
        st.image(overlay, caption="Grad-CAM", width=350)

# HEART TAB

with tab3:
    st.header("Heart Abnormality Detection (ECG)")

    file = st.file_uploader("Upload ECG CSV/TXT (187 values)", type=["csv","txt"])

    if file:
        try:
            data = np.loadtxt(file, delimiter=",").flatten()

            if len(data) != 187:
                st.error("ECG must contain 187 values")
                st.stop()

            data = (data - np.mean(data)) / (np.std(data) + 1e-8)

            signal = torch.tensor(data, dtype=torch.float32)\
                        .unsqueeze(0).unsqueeze(0).to(device)

            logits = heart_model(signal)
            probs = softmax_probs(logits)

            classes = ["Normal","Supraventricular","Ventricular","Fusion","Unknown"]
            idx = np.argmax(probs)

            col1, col2 = st.columns(2)

            fig1, ax1 = plt.subplots(figsize=(10,4))
            ax1.plot(data)
            ax1.set_title("ECG Signal")
            ax1.grid(alpha=0.3)

            with col1:
                st.markdown("### 📈 ECG Signal")
                st.pyplot(fig1)

            with col2:
                st.markdown("### 🧾 Prediction")
                if idx == 0:
                    st.success("✅ Normal ECG")
                else:
                    st.error("⚠️ Abnormal ECG")

                prediction = classes[idx]
                confidence = probs[idx]

                st.info(f"Detected Class: {prediction}")   
                st.write(f"Confidence: {confidence:.2f}")
                decision = decision_support("HEART", prediction, confidence)

                st.markdown("### Clinical Decision Support")
                st.info(decision)

            ig = IntegratedGradients(heart_model)
            baseline = torch.zeros_like(signal).to(device)

            attr, delta = ig.attribute(
                signal,
                baselines=baseline,
                target=int(idx),
                n_steps=50,
                return_convergence_delta=True
            )

            attr = attr.squeeze().cpu().numpy()
            ecg = signal.squeeze().cpu().numpy()
            attr = attr / (np.max(np.abs(attr)) + 1e-8)

            fig2, ax2 = plt.subplots(figsize=(10,4))
            ax2.plot(ecg, label="ECG Signal")
            ax2.plot(attr*2, label="Integrated Gradients")
            ax2.legend()
            ax2.set_title("ECG Explanation")
            ax2.grid(alpha=0.3)

            st.markdown("### 🔍 Explainability")
            st.pyplot(fig2)

            st.write("Convergence Delta:", float(delta))

        except Exception as e:
            st.error(f"Error: {e}")

# FOOTER

st.divider()
st.markdown("© 2026 Multi-Modal Healthcare Diagnostic System")