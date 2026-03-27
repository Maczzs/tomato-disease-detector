import streamlit as st
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# 1. Page Configuration
st.set_page_config(page_title="Tomato Doctor Live", layout="wide")

st.title("🍅 Tomato Doctor: Real-Time AI Diagnostic")
st.write("Point your camera at a tomato leaf to detect diseases instantly.")

# 2. Load the trained model (best.pt must be in the same folder)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 3. Define the Real-Time Processing Logic
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLOv8 inference
        # conf=0.5 reduces 'flickering' by only showing high-certainty boxes
        results = model.predict(img, conf=0.5)

        # Draw the boxes and labels on the frame
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# 4. WebRTC Configuration (Helps connection on mobile data/WiFi)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 5. The Camera Interface
ctx = webrtc_streamer(
    key="tomato-ai",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# 6. Actionable Advice Section (Displayed below the video)
st.markdown("---")
st.subheader("📋 Treatment Recommendations")
st.info("The AI will highlight diseases in the video feed above. If a disease is detected, follow the standard IPM protocols for your region.")

# Example Advice Logic (This appears as a guide)
with st.expander("See Treatment Guide for Common Diseases"):
    st.write("""
    * **Bacterial Spot:** Remove infected debris; use copper-based bactericides.
    * **Early/Late Blight:** Improve air circulation; apply fungicides early in the morning.
    * **Yellow Leaf Curl:** Control whitefly populations using yellow sticky traps.
    """)
