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

# 1. LOAD THE LIGHTWEIGHT MODEL
# This strictly limits the RAM usage so Render doesn't crash
@st.cache_resource
def load_session():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # Prevents CPU traffic jams
    return ort.InferenceSession("best.onnx", sess_options=opts, providers=['CPUExecutionProvider'])

session = load_session()

def run_detection(img):
    h_orig, w_orig = img.shape[:2]
    
    # Dynamic scaling so boxes look good on both webcams and 4K phone photos
    base_thickness = max(2, int(w_orig / 250))
    base_font_scale = max(0.6, w_orig / 900)
    
    # 2. PREPARE IMAGE FOR ONNX
    # We use 416 to match the balanced export we did earlier
    img_resized = cv2.resize(img, (416, 416))
    img_input = img_resized.transpose(2, 0, 1)
    img_input = img_input[np.newaxis, :, :, :].astype(np.float32) / 255.0

    # 3. RUN THE AI
    outputs = session.run(None, {session.get_inputs()[0].name: img_input})
    output = outputs[0][0].T 
    
    boxes = []
    scores_list = []
    class_ids = []

    # Filter out weak guesses
    for row in output:
        scores = row[4:]
        class_id = np.argmax(scores)
        score = scores[class_id]
        
        if score > 0.45:
            x, y, w, h = row[0], row[1], row[2], row[3]
            
            # Map coordinates back to the original photo size
            x1 = int((x - w/2) * w_orig / 416)
            y1 = int((y - h/2) * h_orig / 416)
            x2 = int((x + w/2) * w_orig / 416)
            y2 = int((y + h/2) * h_orig / 416)
            
            # Save for the NMS filter
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores_list.append(score)
            class_ids.append(class_id)

    # 4. CLEAN UP BOXES (Non-Maximum Suppression)
    # This prevents the AI from drawing 10 boxes on top of the same spot
    indices = cv2.dnn.NMSBoxes(boxes, scores_list, 0.45, 0.45)
    detected_labels = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), base_thickness)
            
            label_text = f"{CLASSES[class_ids[i]]} ({int(scores_list[i]*100)}%)"
            cv2.putText(img, label_text, (x, max(20, y - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, (255, 255, 255), base_thickness)
            
            if CLASSES[class_ids[i]] not in detected_labels:
                detected_labels.append(CLASSES[class_ids[i]])

    return img, detected_labels

# --- UI TABS ---
tab1, tab2 = st.tabs(["📷 Live Camera", "📁 Fast Upload"])

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
    uploaded_file = st.file_uploader("Upload leaf photo", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        full_res_img = cv2.imdecode(file_bytes, 1)
        
        # PRE-SHRINKER: Stops massive phone photos from lagging the server
        max_size = 800
        h, w = full_res_img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            full_res_img = cv2.resize(full_res_img, (int(w * scale), int(h * scale)))
        
        st.write(f"Optimized Resolution: {full_res_img.shape[1]}x{full_res_img.shape[0]}")
        
        if st.button("🔍 ANALYZE PHOTO"):
            with st.spinner("Processing image..."):
                result_img, labels = run_detection(full_res_img)
                st.session_state["result_img"] = result_img
                st.session_state["result_labels"] = labels

# --- SHARED RESULTS DISPLAY ---
if "result_img" in st.session_state:
    st.divider()
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
