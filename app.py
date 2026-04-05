import av
import cv2
import numpy as np
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

import mediapipe as mp

st.set_page_config(page_title="MediaPipe Hands POC", layout="wide")
st.title("MediaPipe Hands (Hand keypoints) — Streamlit Community Cloud POC")

st.sidebar.header("Performance / Model")
max_num_hands = st.sidebar.number_input("Max hands", 1, 4, 2, 1)
model_complexity = st.sidebar.selectbox("Model complexity", [0, 1], index=0)
min_det_conf = st.sidebar.slider("Min detection confidence", 0.0, 1.0, 0.5, 0.01)
min_trk_conf = st.sidebar.slider("Min tracking confidence", 0.0, 1.0, 0.5, 0.01)

flip = st.sidebar.checkbox("Mirror (selfie)", True)
draw_connections = st.sidebar.checkbox("Draw connections", True)

# Community Cloud CPU: keep it light
target_fps = st.sidebar.slider("Target processing FPS (server)", 1, 30, 12, 1)
resize_width = st.sidebar.selectbox("Resize width", [256, 320, 416, 512, 640], index=2)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


class HandsProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_t = 0.0

        # Create the MediaPipe Hands object once per processor instance
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=int(max_num_hands),
            model_complexity=int(model_complexity),
            min_detection_confidence=float(min_det_conf),
            min_tracking_confidence=float(min_trk_conf),
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        now = time.time()
        min_dt = 1.0 / max(1, int(target_fps))
        if (now - self.last_t) < min_dt:
            img = frame.to_ndarray(format="bgr24")
            if flip:
                img = cv2.flip(img, 1)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        self.last_t = now

        img = frame.to_ndarray(format="bgr24")
        if flip:
            img = cv2.flip(img, 1)

        # Downscale for speed
        h, w = img.shape[:2]
        if w != int(resize_width):
            new_w = int(resize_width)
            new_h = int(h * (new_w / w))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        rgb.flags.writeable = True

        out = img.copy()

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if draw_connections:
                    mp_draw.draw_landmarks(
                        out,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                else:
                    mp_draw.draw_landmarks(
                        out,
                        hand_landmarks,
                        None,
                        mp_styles.get_default_hand_landmarks_style(),
                        None,
                    )

        return av.VideoFrame.from_ndarray(out, format="bgr24")


st.markdown(
    """
**Notes for Streamlit Community Cloud:**
- This runs **on the server CPU**, so keep resolution low.
- If WebRTC won’t connect from your network, you may need a **TURN server**.
"""
)

webrtc_streamer(
    key="hands",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=HandsProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

