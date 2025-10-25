import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np


BRAIN_CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
BLOOD_CLASSES = ['Benign', 'Pre-B', 'Pro-B', 'early Pre-B'] 


@st.cache_resource
def load_brain_model():
    return YOLO('./runs/detect/train/weights/best.pt')

@st.cache_resource
def load_blood_model():
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=False, aux_logits=False)
    
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.6),
        torch.nn.Linear(512, 4) 
    )
   
    model.load_state_dict(torch.load('./googlenet_trained_weights.pth', map_location='cpu'))
    model.eval()
    return model

blood_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


st.set_page_config(page_title="Détection de Cancer par IA", page_icon="🏥", layout="wide")

st.title("🏥 Diagnostic Multimodal par IA")
st.markdown("### Détection de Cancers Cérébraux & Sanguins")


col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🧠 Tumeurs Cérébrales")
    st.info("Upload d'image IRM pour détecter: Glioma, Meningioma, Pituitary")
    
with col2:
    st.markdown("#### 🩸 Cancers Sanguins")
    st.info("Upload d'image de cellules sanguines pour détecter les leucémies")

st.markdown("---")


tab1, tab2 = st.tabs(["🧠 Tumeurs Cérébrales (IRM)", "🩸 Cancers Sanguins (Cellules)"])


with tab1:
    st.header("Détection de Tumeurs Cérébrales")
    
    brain_file = st.file_uploader("Choisir une image IRM", type=['jpg', 'jpeg', 'png'], key="brain")
    
    if brain_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(brain_file)
            st.image(image, caption="Image IRM uploadée", use_container_width=True)
        
        if st.button("🔍 Analyser l'IRM", key="analyze_brain"):
            with st.spinner("Analyse en cours..."):
                brain_model = load_brain_model()
                results = brain_model.predict(image, conf=0.25)
                
                with col2:
            
                    result_img = results[0].plot()
                    st.image(result_img, caption="Résultat de détection", use_container_width=True)
                
                st.markdown("### 📊 Résultats")
                
                if len(results[0].boxes) > 0:
                    for i, box in enumerate(results[0].boxes):
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = BRAIN_CLASSES[cls]
                        

                        if class_name == "No Tumor":
                            st.success(f"✅ **{class_name}** - Confiance: {conf*100:.1f}%")
                        else:
                            st.error(f"⚠️ **{class_name}** détecté - Confiance: {conf*100:.1f}%")
                        
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        st.caption(f"Position: [{int(x1)}, {int(y1)}] → [{int(x2)}, {int(y2)}]")
                else:
                    st.warning("⚠️ Aucune tumeur détectée")

with tab2:
    st.header("Détection de Cancers Sanguins")
    
    blood_file = st.file_uploader("Choisir une image de cellules sanguines", type=['jpg', 'jpeg', 'png'], key="blood")
    
    if blood_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(blood_file).convert('RGB')
            st.image(image, caption="Image de cellules uploadée", use_container_width=True)
        
        if st.button("🔍 Analyser les cellules", key="analyze_blood"):
            with st.spinner("Analyse en cours..."):
                blood_model = load_blood_model()
                
                
                img_tensor = blood_transform(image).unsqueeze(0)
                
            
                with torch.no_grad():
                    outputs = blood_model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                
                pred_class = BLOOD_CLASSES[predicted.item()]
                conf_value = confidence.item()
                
                with col2:
                    st.markdown("### 📊 Résultats")
                    
                
                    if pred_class == "Normal":
                        st.success(f"✅ **{pred_class}**")
                        st.metric("Confiance", f"{conf_value*100:.1f}%")
                    else:
                        st.error(f"⚠️ **{pred_class}** détecté")
                        st.metric("Confiance", f"{conf_value*100:.1f}%")
                    
                    
                    st.progress(conf_value)
                    
                    
                    st.markdown("#### Distribution des probabilités")
                    prob_data = {BLOOD_CLASSES[i]: float(probabilities[0][i]) * 100 
                                for i in range(len(BLOOD_CLASSES))}
                    
                    for class_name, prob in sorted(prob_data.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"**{class_name}**: {prob:.2f}%")
                        st.progress(prob/100)

with st.sidebar:
    st.header("ℹ️ Informations")
    
    st.markdown("### 🧠 Modèle Tumeurs Cérébrales")
    st.write("- **Architecture**: YOLOv8n")
    st.write("- **Classes**: 4")
    st.write("- **Dataset**: MRI Brain Tumors")
    st.write("- **mAP50**: 96.1%")
    
    st.markdown("---")
    
    st.markdown("### 🩸 Modèle Cancers Sanguins")
    st.write("- **Architecture**: GoogLeNet")
    st.write("- **Classes**: 4")
    st.write("- **Dataset**: Blood Cell Images")
