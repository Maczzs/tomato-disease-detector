import streamlit as st
from ultralytics import YOLO
import av
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. Page Configuration
st.set_page_config(page_title="Tomato Disease Analyzer", layout="centered")
st.title("Tomato Health Scanner")

# 2. Load Model (Optimized ONNX)
@st.cache_resource
def load_model():
    # This loads the optimized brain you exported in VS Code
    return YOLO('best.onnx', task='detect')

model = load_model()

# 3. Detection Engine
class VideoProcessor:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.last_annotated_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # The core detection code:
        # imgsz=320 ensures the CPU can keep up with the video stream
        results = model.predict(img, conf=0.45, imgsz=320, verbose=False)
        annotated_img = results[0].plot()

        # Update the 'last_annotated_frame' for the snapshot feature
        with self.frame_lock:
            self.last_annotated_frame = annotated_img

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# 4. Connection Configuration (STUN server for mobile networks)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 5. The Live Interface
ctx = webrtc_streamer(
    key="detection-system",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "facingMode": "environment", # Directs phone to use the back camera
            "width": {"ideal": 640},
            "height": {"ideal": 480}
        },
        "audio": False
    },
    async_processing=True,
)

# 6. Snapshot and Lock Logic
if ctx.video_processor:
    if st.button("Capture and Lock Detection"):
        with ctx.video_processor.frame_lock:
            if ctx.video_processor.last_annotated_frame is not None:
                # Save the frame to the server's session memory
                st.session_state["captured_image"] = ctx.video_processor.last_annotated_frame
            else:
                st.warning("Camera is initializing. Please try again in a moment.")

# 7. Display Results
if "captured_image" in st.session_state:
    st.markdown("---")
    st.subheader("Locked Analysis Result")
    st.image(st.session_state["captured_image"], channels="BGR", use_container_width=True)
    
    if st.button("Clear Captured Result"):
        del st.session_state["captured_image"]
        st.rerun()
