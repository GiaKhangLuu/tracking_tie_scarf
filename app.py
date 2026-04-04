import av
import cv2
import numpy as np
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO

st.set_page_config(page_title="YOLO26 Pose POC", layout="wide")
st.title("YOLO26 Pose (Upper-body) — Streamlit Community Cloud POC")

st.sidebar.header("Performance / Model")
weights = st.sidebar.text_input("Weights", "yolo26n-pose.pt")
conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("IoU", 0.0, 1.0, 0.45, 0.01)
max_people = st.sidebar.number_input("Max people", 1, 10, 3, 1)

flip = st.sidebar.checkbox("Mirror (selfie)", True)
draw_skeleton = st.sidebar.checkbox("Draw skeleton", True)

# Community Cloud CPU: keep it light
target_fps = st.sidebar.slider("Target inference FPS (server)", 1, 30, 8, 1)
resize_width = st.sidebar.selectbox("Resize width", [256, 320, 416, 512, 640], index=2)

UPPER_BODY_IDXS = list(range(0, 11))
UPPER_BODY_EDGES = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (0, 5), (0, 6),
    (1, 0), (2, 0),
    (3, 1), (4, 2),
]

@st.cache_resource
def load_model(path: str):
    # One model per Streamlit process
    return YOLO(path)

class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.last_t = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        now = time.time()
        min_dt = 1.0 / max(1, int(target_fps))
        if (now - self.last_t) < min_dt:
            # If we skip inference, still display the frame (no overlay)
            img = frame.to_ndarray(format="bgr24")
            if flip:
                img = cv2.flip(img, 1)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        self.last_t = now

        img = frame.to_ndarray(format="bgr24")
        if flip:
            img = cv2.flip(img, 1)

        if self.model is None:
            self.model = load_model(weights)

        # Downscale for speed
        h, w = img.shape[:2]
        if w != int(resize_width):
            new_w = int(resize_width)
            new_h = int(h * (new_w / w))
            small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            small = img

        results = self.model.predict(
            source=small,
            conf=conf,
            iou=iou,
            max_det=int(max_people),
            verbose=False,
        )

        out = small.copy()

        if results and results[0].keypoints is not None and results[0].keypoints.xy is not None:
            r = results[0]
            kpts_xy = r.keypoints.xy.cpu().numpy()

            kpts_conf = None
            if hasattr(r.keypoints, "conf") and r.keypoints.conf is not None:
                kpts_conf = r.keypoints.conf.cpu().numpy()

            people = min(kpts_xy.shape[0], int(max_people))
            for p in range(people):
                pts = kpts_xy[p]
                cfs = kpts_conf[p] if kpts_conf is not None else None

                for k in UPPER_BODY_IDXS:
                    x, y = pts[k]
                    if np.isnan(x) or np.isnan(y):
                        continue
                    if cfs is not None and cfs[k] < conf:
                        continue
                    cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)

                if draw_skeleton:
                    for a, b in UPPER_BODY_EDGES:
                        ax, ay = pts[a]
                        bx, by = pts[b]
                        if np.isnan(ax) or np.isnan(ay) or np.isnan(bx) or np.isnan(by):
                            continue
                        if cfs is not None and (cfs[a] < conf or cfs[b] < conf):
                            continue
                        cv2.line(out, (int(ax), int(ay)), (int(bx), int(by)), (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(out, format="bgr24")

st.markdown(
    """
    **Notes for Streamlit Community Cloud:**
    - This runs inference **on the server CPU**, so keep the model small and resolution low.
    - If WebRTC won’t connect from your network, you’ll likely need a **TURN server**.
    """)

webrtc_streamer(
    key="pose",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PoseProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
