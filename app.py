import streamlit as st
from ultralytics import YOLO
import av
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. Page Setup
st.set_page_config(page_title="Tomato Analysis System", layout="centered")
st.title("Tomato Health Scanner")
st.write("Click Start to open the camera. Point at a leaf and click Capture to lock the image.")

# 2. Load the optimized brain (ONNX)
@st.cache_resource
def load_model():
    # task='detect' is required for ONNX models
    return YOLO('best.onnx', task='detect')

model = load_model()

# 3. Frame Processing Class
class VideoProcessor:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.active_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Performance: imgsz=320 is critical for smooth video on mobile
        results = model.predict(img, conf=0.40, imgsz=320, verbose=False)
        annotated_img = results[0].plot()

        with self.frame_lock:
            self.active_frame = annotated_img

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# 4. Connection Configuration
# We use a Google STUN server to fix the 'constant loading' issue on mobile networks
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 5. Camera Interface
# This section mimics a native camera app behavior
ctx = webrtc_streamer(
    key="camera-app",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "facingMode": "environment", # Tries to force the back camera
            "width": {"ideal": 640},
            "height": {"ideal": 480}
        },
        "audio": False
    },
    async_processing=True,
)

# 6. Capture Control
if ctx.video_processor:
    if st.button("Capture Photo"):
        with ctx.video_processor.frame_lock:
            if ctx.video_processor.active_frame is not None:
                st.session_state["locked_result"] = ctx.video_processor.active_frame
            else:
                st.error("Camera feed not ready. Please wait.")

# 7. Result Display
if "locked_result" in st.session_state:
    st.markdown("---")
    st.subheader("Captured Analysis")
    st.image(st.session_state["locked_result"], channels="BGR", use_container_width=True)
    
    if st.button("Discard and Scan Again"):
        del st.session_state["locked_result"]
        st.rerun()
