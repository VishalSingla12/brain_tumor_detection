import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import config
import utils
from model import BrainTumorCNN
from gradcam import generate_gradcam

st.set_page_config(
    page_title="Brain Tumor Detector",
    
    layout="wide"
)

st.title(" Brain Tumor Detector — CNN + Grad-CAM")
st.markdown("""Presented by: Vishal Singla and Rishit Goel""")

with st.expander("device"):
    device = utils.get_device()
    st.info(f"Running on: {device}")

@st.cache_resource
def load_model():
    if not os.path.exists(config.MODEL_SAVE_PATH):
        return None
    
    model = BrainTumorCNN(num_classes=config.NUM_CLASSES)
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

model = load_model()

if model is None:
    st.error("Model not found! Please train the model first using `python train.py`.")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader("Upload an MRI Scan", type=["jpg", "jpeg", "png"])
    
    image = None
    if uploaded_file is not None:
        if isinstance(uploaded_file, str): 
            image = Image.open(uploaded_file).convert('RGB')
        else: 
            image = Image.open(uploaded_file).convert('RGB')
        
        st.image(image, caption="Original Image", use_container_width=True)

with col2:
    st.subheader("Prediction & Explanation")
    
    if image is not None:
        img_resized = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        img_tensor = np.array(img_resized) / 255.0
        img_tensor = (img_tensor - mean) / std
        img_tensor = torch.FloatTensor(img_tensor).permute(2, 0, 1).unsqueeze(0).to(device)
        
        if st.button("Analyze MRI", type="primary"):
            with st.spinner("Analyzing..."):
                heatmap, overlay, logits = generate_gradcam(model, img_tensor)
                
                probs = F.softmax(logits, dim=1).cpu().detach().numpy()[0]
                pred_idx = np.argmax(probs)
                pred_label = config.CLASS_NAMES[pred_idx]
                pred_prob = probs[pred_idx]
                
                if pred_prob >= 0.8:
                    st.success(f"Prediction: **{pred_label}** ({pred_prob:.2%})")
                else:
                    st.success(f"Prediction: **{pred_label}** ({pred_prob:.2%})")
                
                prob_df = pd.DataFrame({
                    'Class': config.CLASS_NAMES,
                    'Probability': probs
                })
                st.bar_chart(prob_df.set_index('Class'))
                st.image(overlay, caption=f"Grad-CAM Heatmap (Focus Area)", use_container_width=True)
                
                import io
                buf = io.BytesIO()
                overlay.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                label="Download Grad-CAM Overlay",
                data=byte_im,
                file_name="gradcam_overlay.png",
                mime="image/png"
                    )
st.markdown("---")
st.markdown("""
The **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualization shows the regions of the image that were most important for the model's prediction.
- **Red/Yellow areas**: High importance .
- **Blue areas**: Low importance.
""")