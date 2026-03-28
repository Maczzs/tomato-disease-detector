import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import av
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. Page Configuration
st.set_page_config(page_title="Tomato Diagnostic System", layout="centered")
st.title("Tomato Disease Analyzer")
st.write("Point the camera at a tomato leaf. The system will identify the specific condition.")

# 2. Define Class Names (Ensure these match your training order)
CLASSES = [
    "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", 
    "Septoria Leaf Spot", "Spider Mites", "Target Spot", 
    "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"
]

# 3. Load Optimized Inference Session
@st.cache_resource
def load_session():
    return ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])

session = load_session()

def process_image(img, session, conf_threshold=0.45):
    h_orig, w_orig = img.shape[:2]
    
    # Pre-process for ONNX (320x320)
    img_resized = cv2.resize(img, (320, 320))
    img_input = img_resized.transpose(2, 0, 1)
    img_input = img_input[np.newaxis, :, :, :].astype(np.float32) / 255.0

    # Run Detection
    outputs = session.run(None, {session.get_inputs()[0].name: img_input})
    output = outputs[0][0] # Shape: [84, 2100] for YOLOv8-320
    
    # YOLOv8 output processing
    # Transpose to get [2100, 84] where 84 = 4 box coords + 80 class scores
    output = output.T 
    
    for row in output:
        # Get the scores for your 10 classes (indices 4 to 13)
        scores = row[4:]
        class_id = np.argmax(scores)
        score = scores[class_id]
        
        if score > conf_threshold:
            # Scale coordinates back to original image size
            x, y, w, h = row[0], row[1], row[2], row[3]
            
            x1 = int((x - w/2) * w_orig / 320)
            y1 = int((y - h/2) * h_orig / 320)
            x2 = int((x + w/2) * w_orig / 320)
            y2 = int((y + h/2) * h_orig / 320)
            
            # Draw Box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Label with specific Class Name
            label = f"{CLASSES[class_id]}: {score:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img

# 4. Video Processing Logic
class VideoProcessor:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.active_frame = None
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Analyze every 5th frame to stay within Render's CPU limits
        if self.frame_count % 5 == 0:
            img = process_image(img, session)
            with self.frame_lock:
                self.active_frame = img

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 5. Interface and Camera Connection
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

ctx = webrtc_streamer(
    key="diagnostic-scanner",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {"facingMode": "environment"}, 
        "audio": False
    },
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

# 6. Snapshot Display
if ctx.video_processor:
    if st.button("Capture Diagnostic Result"):
        with ctx.video_processor.frame_lock:
            if ctx.video_processor.active_frame is not None:
                st.session_state["diag_snapshot"] = ctx.video_processor.active_frame

if "diag_snapshot" in st.session_state:
    st.divider()
    st.subheader("Analysis Summary")
    st.image(st.session_state["diag_snapshot"], channels="BGR", use_container_width=True)
    if st.button("Clear and Restart Scanner"):
        del st.session_state["diag_snapshot"]
        st.rerun()
