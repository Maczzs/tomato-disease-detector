import streamlit as st
import cv2
import numpy as np
import av
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO  # NEW: We use ultralytics directly instead of ONNX

st.set_page_config(page_title="Tomato Health Pro", layout="centered")
st.title("Tomato Health Scanner")

# Load the PyTorch model directly
@st.cache_resource
def load_model():
    # Make sure 'best.pt' is in the same folder as this app.py file
    return YOLO("best.pt")

model = load_model()

def run_detection(img):
    # YOLOv8 handles all the resizing and math internally, so we don't 
    # need the manual ONNX pre-processing code anymore!
    
    # Run the image through the AI
    # conf=0.45 ignores guesses under 45% confidence
    # imgsz=416 balances speed and accuracy
    results = model(img, conf=0.45, imgsz=416)[0] 
    
    # YOLO provides a built-in function to draw the boxes directly
    # on our image, which saves us a ton of math!
    annotated_img = results.plot()
    
    # Extract the names of the detected diseases
    detected_labels = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        if class_name not in detected_labels:
            detected_labels.append(class_name)
            
    return annotated_img, detected_labels

# --- UI TABS ---
tab1, tab2 = st.tabs(["📷 Live Camera", "📁 Full-Res Upload"])

with tab1:
    class VideoProcessor:
        def __init__(self):
            self.frame_lock = threading.Lock()
            self.last_raw_frame = None
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            with self.frame_lock: self.last_raw_frame = img
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    ctx = webrtc_streamer(
        key="diagnostic-scanner",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"facingMode": "environment", "width": 640}, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    if ctx.video_processor:
        if st.button("📸 CAPTURE AND ANALYZE"):
            with ctx.video_processor.frame_lock:
                if ctx.video_processor.last_raw_frame is not None:
                    snap = ctx.video_processor.last_raw_frame.copy()
                    result_img, labels = run_detection(snap)
                    st.session_state["result_img"] = result_img
                    st.session_state["result_labels"] = labels

with tab2:
    uploaded_file = st.file_uploader("Upload original leaf photo (High Quality)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        full_res_img = cv2.imdecode(file_bytes, 1)
        
        st.write(f"Detected Resolution: {full_res_img.shape[1]}x{full_res_img.shape[0]}")
        
        if st.button("🔍 ANALYZE ORIGINAL PHOTO"):
            with st.spinner("Processing image..."):
                result_img, labels = run_detection(full_res_img)
                st.session_state["result_img"] = result_img
                st.session_state["result_labels"] = labels

# --- SHARED RESULTS DISPLAY ---
if "result_img" in st.session_state:
    st.divider()
    # Display the image with the AI's drawn boxes
    st.image(st.session_state["result_img"], channels="BGR", use_container_width=True)
    
    if st.session_state["result_labels"]:
        st.subheader("Detected Condition(s):")
        for label in st.session_state["result_labels"]:
            st.success(f"**{label}**")
    else:
        st.warning("No disease detected. Please ensure the leaf is clear and centered.")

    if st.button("Clear and Start Over"):
        for key in ["result_img", "result_labels"]:
            if key in st.session_state: del st.session_state[key]
        st.rerun()
