import streamlit as st
from ultralytics import YOLO
import av
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. Page Layout
st.set_page_config(page_title="Tomato Disease Detection System", layout="centered")
st.title("Tomato Health Analyzer")
st.write("Use this tool to scan tomato leaves for diseases in real-time.")

# 2. Load Optimized Model
@st.cache_resource
def load_model():
    # Loading the ONNX file you just exported
    return YOLO('best.onnx', task='detect')

model = load_model()

# 3. Detection Engine
class VideoProcessor:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.active_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # High-speed inference
        results = model.predict(img, conf=0.40, imgsz=320, verbose=False)
        annotated_img = results[0].plot()

        with self.frame_lock:
            self.active_frame = annotated_img

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# 4. Connection Logic (Fixes the 'Constant Loading' issue)
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ]
    }
)

# 5. Camera Interface
ctx = webrtc_streamer(
    key="tomato-scanner",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "facingMode": "environment", # Focuses on the back camera
            "width": {"ideal": 640},
            "height": {"ideal": 480}
        },
        "audio": False
    },
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

# 6. Snapshot Feature
if ctx.video_processor:
    if st.button("Capture and Lock Image"):
        with ctx.video_processor.frame_lock:
            if ctx.video_processor.active_frame is not None:
                st.session_state["result_snapshot"] = ctx.video_processor.active_frame
            else:
                st.error("Wait for camera to initialize before capturing.")

# 7. Results Section
if "result_snapshot" in st.session_state:
    st.divider()
    st.subheader("Analysis Result")
    st.image(st.session_state["result_snapshot"], channels="BGR", use_container_width=True)
    
    if st.button("Reset Scanner"):
        del st.session_state["result_snapshot"]
        st.rerun()
