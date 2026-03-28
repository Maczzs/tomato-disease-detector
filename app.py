import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import av
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Tomato Health Pro", layout="centered")
st.title("Tomato Health Scanner")

CLASSES = [
    "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", 
    "Septoria Leaf Spot", "Spider Mites", "Target Spot", 
    "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"
]

@st.cache_resource
def load_session():
    # Performance optimization for Render
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession("best.onnx", sess_options=opts, providers=['CPUExecutionProvider'])

session = load_session()

def run_detection(img):
    h_orig, w_orig = img.shape[:2]
    img_resized = cv2.resize(img, (320, 320))
    img_input = img_resized.transpose(2, 0, 1)
    img_input = img_input[np.newaxis, :, :, :].astype(np.float32) / 255.0

    outputs = session.run(None, {session.get_inputs()[0].name: img_input})
    output = outputs[0][0].T 
    
    detected_labels = [] # We will store names here to show under the photo
    
    for row in output:
        scores = row[4:]
        class_id = np.argmax(scores)
        score = scores[class_id]
        
        if score > 0.45: # Raised slightly for better accuracy
            x, y, w, h = row[0], row[1], row[2], row[3]
            x1, y1 = int((x - w/2) * w_orig / 320), int((y - h/2) * h_orig / 320)
            x2, y2 = int((x + w/2) * w_orig / 320), int((y + h/2) * h_orig / 320)
            
            # 1. DRAW BOX
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # 2. DRAW TEXT (Smart positioning to prevent clipping)
            label_text = f"{CLASSES[class_id]} ({int(score*100)}%)"
            text_y = y1 - 10 if y1 > 20 else y1 + 20 # Put text inside if too high
            
            cv2.putText(img, label_text, (x1, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if CLASSES[class_id] not in detected_labels:
                detected_labels.append(CLASSES[class_id])
    
    return img, detected_labels

class VideoProcessor:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.last_raw_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.frame_lock:
            self.last_raw_frame = img
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Using the robust STUN list we discussed
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

ctx = webrtc_streamer(
    key="diagnostic-scanner",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

if ctx.video_processor:
    if st.button("📸 CAPTURE AND ANALYZE"):
        with ctx.video_processor.frame_lock:
            if ctx.video_processor.last_raw_frame is not None:
                snap = ctx.video_processor.last_raw_frame.copy()
                with st.spinner("Processing..."):
                    result_img, labels = run_detection(snap)
                    st.session_state["result_img"] = result_img
                    st.session_state["result_labels"] = labels
            else:
                st.error("Camera not ready.")

# --- RESULTS DISPLAY ---
if "result_img" in st.session_state:
    st.divider()
    st.image(st.session_state["result_img"], channels="BGR", use_container_width=True)
    
    # NEW: Display class names clearly under the image
    if st.session_state["result_labels"]:
        st.subheader("Detected Condition(s):")
        for label in st.session_state["result_labels"]:
            # Highlight with a background color
            st.success(f"**{label}**")
            
            # Expert advice logic
            if label == "Early Blight":
                st.info("Tip: Look for dark 'target' spots. Remove infected lower leaves.")
            elif label == "Yellow Leaf Curl Virus":
                st.info("Tip: Identified by upward leaf rolling. Control whiteflies to prevent spread.")
    else:
        st.warning("No disease detected. Please try a closer or clearer photo.")

    if st.button("Clear Results"):
        for key in ["result_img", "result_labels"]:
            if key in st.session_state: del st.session_state[key]
        st.rerun()
