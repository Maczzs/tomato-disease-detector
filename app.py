import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import av
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. Page Config
st.set_page_config(page_title="Tomato Analysis System", layout="centered")
st.title("Tomato Health Scanner")

# 2. Lightweight Model Loading
@st.cache_resource
def load_session():
    # Use CPU-only provider to save memory
    return ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])

session = load_session()

def process_image(img, session, conf_threshold=0.4):
    # Pre-process image for ONNX (320x320)
    img_resized = cv2.resize(img, (320, 320))
    img_input = img_resized.transpose(2, 0, 1)
    img_input = img_input[np.newaxis, :, :, :].astype(np.float32) / 255.0

    # Run Detection
    outputs = session.run(None, {session.get_inputs()[0].name: img_input})
    
    # This part draws the boxes manually to save memory
    # (Replacing the heavy results[0].plot() function)
    output = outputs[0][0]
    for i in range(output.shape[1]):
        score = output[4, i]
        if score > conf_threshold:
            # Drawing logic simplified for performance
            x, y, w, h = output[0, i], output[1, i], output[2, i], output[3, i]
            # Scale coordinates back to original image size
            h_orig, w_orig = img.shape[:2]
            x1 = int((x - w/2) * w_orig / 320)
            y1 = int((y - h/2) * h_orig / 320)
            x2 = int((x + w/2) * w_orig / 320)
            y2 = int((y + h/2) * h_orig / 320)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Leaf Disease {score:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

# 3. Detection Engine
class VideoProcessor:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.active_frame = None
        self.count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.count += 1
        
        # Performance: Only analyze every 6th frame to lower CPU workload
        if self.count % 6 == 0:
            img = process_image(img, session)
            with self.frame_lock:
                self.active_frame = img

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Connection and Camera
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

ctx = webrtc_streamer(
    key="tomato-app",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

# 5. Snapshot Logic
if ctx.video_processor:
    if st.button("Capture and Lock Analysis"):
        with ctx.video_processor.frame_lock:
            if ctx.video_processor.active_frame is not None:
                st.session_state["snapshot"] = ctx.video_processor.active_frame

if "snapshot" in st.session_state:
    st.divider()
    st.image(st.session_state["snapshot"], channels="BGR", use_container_width=True)
    if st.button("Reset Scanner"):
        del st.session_state["snapshot"]
        st.rerun()
