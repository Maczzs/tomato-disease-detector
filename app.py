# %%
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Load your trained model
model = YOLO('best.pt')

st.title("🍅 Tomato Doctor: AI Disease Diagnostic")
st.write("Upload a photo of a tomato leaf to detect diseases and get treatment advice.")

# 2. File Uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("Checking for diseases...")
    
    # 3. Run AI Inference
    results = model(image)
    
    # 4. Show Results
    for r in results:
        # This saves the image with the boxes drawn on it
        res_plotted = r.plot()
        st.image(res_plotted, caption='AI Detection Results', use_container_width=True)
        
        if len(r.boxes) == 0:
            st.success("No diseases detected! Your plant looks healthy.")
        else:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                conf = float(box.conf[0])
                st.warning(f"Detected: **{label}** ({conf:.2%} confidence)")
                
                # Grade A: Add 'Actionable Diagnostics' here
                if "Blight" in label:
                    st.info("💡 **Advice:** Remove infected leaves and apply a copper-based fungicide.")


