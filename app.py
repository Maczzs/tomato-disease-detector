import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import av
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. Page Config
st.set_page_config(page_title="Tomato Disease Analyzer", layout="centered")
st.title("Tomato Disease Scanner")
st.write("PLEASE WORK OMG. Point and click 'Capture' to analyze.")

# 2. Class Names
CLASSES = [
    "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", 
    "Septoria Leaf Spot", "Spider Mites", "Target Spot", 
    "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"
]

# 3. Load ONNX Session (Lightweight)
@st.cache_resource
def load_session():
    return ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])

session = load_session()

def run_detection(img):
    """This function only runs when you press the Capture button"""
    h_orig, w_orig = img.shape[:2]
    img_resized = cv2.resize(img, (320, 320))
    img_input = img_resized.transpose(2, 0, 1)
    img_input = img_input[np.newaxis, :, :, :].astype(np.float32) / 255.0

    outputs = session.run(None, {session.get_inputs()[0].name: img_input})
    output = outputs[0][0].T 
    
    found_something = False
    for row in output:
        scores = row[4:]
        class_id = np.argmax(scores)
        score = scores[class_id]
        
        if score > 0.40:
            found_something = True
            x, y, w, h = row[0], row[1], row[2], row[3]
            x1 = int((x - w/2) * w_orig / 320)
            y1 = int((y - h/2) * h_orig / 320)
            x2 = int((x + w/2) * w_orig / 320)
            y2 = int((y + h/2) * h_orig / 320)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{CLASSES[class_id]}: {score:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return img, found_something

# 4. Video Processor
class VideoProcessor:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.last_raw_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # We just store the frame and send it back immediately.
        # This keeps the FPS high and smooth.
        with self.frame_lock:
            self.last_raw_frame = img

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Updated RTC Configuration for high compatibility
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
            # Optional: Add a free TURN server here if you sign up for Metered.ca
            # {
            #     "urls": "turn:global.relay.metered.ca:80",
            #     "username": "your_username",
            #     "credential": "your_password"
            # }
        ]
    }
)

# 5. Camera Setup
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

ctx = webrtc_streamer(
    key="smooth-camera",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"facingMode": "environment", "width": 640, "height": 480},
        "audio": False
    },
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

# 6. Capture and Analyze Logic
if ctx.video_processor:
    if st.button("CAPTURE & ANALYZE"):
        with ctx.video_processor.frame_lock:
            if ctx.video_processor.last_raw_frame is not None:
                # Copy the frame so we don't mess with the live feed
                snap = ctx.video_processor.last_raw_frame.copy()
                
                # RUN AI ONLY ONCE
                with st.spinner("Analyzing leaf..."):
                    result_img, found = run_detection(snap)
                    st.session_state["last_snap"] = result_img
                    st.session_state["found"] = found
            else:
                st.error("Camera not ready. Please wait.")

# 7. Show the Result
if "last_snap" in st.session_state:
    st.divider()
    st.subheader("Final Diagnosis")
    st.image(st.session_state["last_snap"], channels="BGR", use_container_width=True)
    
    if not st.session_state["found"]:
        st.info("No diseases detected. The leaf appears healthy or the image is unclear.")
    
    if st.button("Clear Photo"):
        del st.session_state["last_snap"]
        st.rerun()
        
    if "diag_snap" in st.session_state:
    st.info("**Note for User:** Early Blight usually has dark 'target-like' spots. "
            "If you see leaves curling upwards without spots, it is likely Yellow Leaf Curl.")
