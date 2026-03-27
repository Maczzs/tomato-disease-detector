import streamlit as st
from ultralytics import YOLO
import av
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. Page Setup
st.set_page_config(page_title="Tomato AI Pro", layout="centered")
st.title("Tomato Disease Detector/Scanner")
st.write("Point at a leaf. If using a phone, use the 'SELECT DEVICE' button below to switch to the back camera.")

# 2. Fast Model Loading
@st.cache_resource
def load_model():
    # Use the .onnx version we created for max speed
    return YOLO('best.onnx', task='detect')

model = load_model()

# 3. High-Performance Video Processor
class VideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.last_annotated_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Optimization: Only run AI every 5 frames to keep the video smooth
        if self.frame_count % 5 == 0:
            # imgsz=320 makes the 'math' 4x faster for the Render CPU
            results = model.predict(img, conf=0.45, imgsz=320, verbose=False)
            self.last_annotated_frame = results[0].plot()

        # If we have a processed frame, show it. Otherwise, show raw video.
        if self.last_annotated_frame is not None:
            return av.VideoFrame.from_ndarray(self.last_annotated_frame, format="bgr24")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Global WebRTC Config (Essential for Mobile/WiFi)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 5. The Real-Time Camera Interface
ctx = webrtc_streamer(
    key="tomato-ai-final",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    # This setting allows the user to choose their camera (Front/Back)
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)

# 6. User Instructions
if ctx.state.playing:
    st.success("✅ AI Active. Hold the leaf steady for 1 second to detect.")
else:
    st.info("💡 Click 'START' and allow camera access to begin.")

st.markdown("""
---
**Presentation Tip:** If the video is blank on Android, tap the **Lock Icon** in the Chrome address bar and ensure **Camera Permission** is allowed for this specific site.
""")
