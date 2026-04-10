import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from twilio.rest import Client
import numpy as np
import time
import streamlit as st
import traceback
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from pose_template_features import angle_3pts, make_feature_vector_pose, pose_ids_for_indices, min_template_sae
from drawing import draw_status_pil
import random

import mediapipe as mp

account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)

token = client.tokens.create()

st.set_page_config(page_title="Pose Compare POC", layout="wide")
st.title("Theo dõi tư thế  đeo khăn quàng đỏ")

pose_min_det_conf = 0.5
pose_min_trk_conf = 0.5

flip = True
draw_connections = True

resize_width = 416

TEMPLATE_SEQ_PATH = "./template_seq.npy"
MIN_VIS = 0.6
WRONG_ANGLE_THRESHOLD = 50
K_CONSECUTIVE_FRAMES = 50
W_ELBOW_ANGLE = 0.7
W_WRIST_ANGLE = 0.7
NUM_FRAME_TO_WARMUP = 10

# -----------------------------
# Load template seq
# -----------------------------
try:
    template_seq = np.load(TEMPLATE_SEQ_PATH).astype(np.float32)
    if template_seq.ndim != 2:
        raise ValueError("template_seq.npy must be a 2D array (T, D).")
except Exception as e:
    st.error(f"Failed to load template sequence from {TEMPLATE_SEQ_PATH}: {e}")
    st.stop()

# -----------------------------
# Pose helpers / feature extraction
# -----------------------------
mp_pose = mp.solutions.pose
LM_POSE = mp_pose.PoseLandmark

# minimal connections for drawing
CONNECTIONS = [
    (LM_POSE.LEFT_SHOULDER, LM_POSE.LEFT_ELBOW),
    (LM_POSE.LEFT_ELBOW, LM_POSE.LEFT_WRIST),
    (LM_POSE.LEFT_WRIST, LM_POSE.LEFT_INDEX),
    (LM_POSE.RIGHT_SHOULDER, LM_POSE.RIGHT_ELBOW),
    (LM_POSE.RIGHT_ELBOW, LM_POSE.RIGHT_WRIST),
    (LM_POSE.RIGHT_WRIST, LM_POSE.RIGHT_INDEX),
    (LM_POSE.LEFT_SHOULDER, LM_POSE.RIGHT_SHOULDER),
]
TARGET = [
    (LM_POSE.LEFT_SHOULDER, 11),
    (LM_POSE.RIGHT_SHOULDER, 12),
    (LM_POSE.LEFT_ELBOW, 13),
    (LM_POSE.RIGHT_ELBOW, 14),
    (LM_POSE.LEFT_WRIST, 15),
    (LM_POSE.RIGHT_WRIST, 16),
    (LM_POSE.LEFT_INDEX, 19),
    (LM_POSE.RIGHT_INDEX, 20),
]

# -----------------------------
# Video Processor
# -----------------------------
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=float(pose_min_det_conf),
            min_tracking_confidence=float(pose_min_trk_conf),
        )

        self.ids = pose_ids_for_indices()

        # run state machine
        self.state_list = ['IDLE', 'WRONG', 'CORRECT', 'WARMUP']
        self.run_state_id = 0
        self.warmup_count = 0
        self.text_state = None
        self.status_text = ""
        self.CORRECT_COUNT = 0
        self.WRONG_COUNT = 0

        self.rng = np.random.default_rng()

        self.icon_smile = self._load_rgba("./asset/smile.png")
        self.icon_sad   = self._load_rgba("./asset/sad.png")
        
        self.particles = []  # list[dict]
        self.frame_idx = 0

    def _overlay_rgba(self, dst_bgr, icon_rgba, x, y, alpha_mul=1.0):
        """Alpha blend icon_rgba onto dst_bgr at top-left (x,y)."""
        H, W = dst_bgr.shape[:2]
        ih, iw = icon_rgba.shape[:2]

        # clip region
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + iw), min(H, y + ih)
        if x0 >= x1 or y0 >= y1:
            return

        icon_roi = icon_rgba[(y0 - y):(y1 - y), (x0 - x):(x1 - x)]
        icon_rgb = icon_roi[:, :, :3].astype(np.float32)
        alpha = (icon_roi[:, :, 3:4].astype(np.float32)) / 255.0

        dst_roi = dst_bgr[y0:y1, x0:x1].astype(np.float32)
        out_roi = alpha * icon_rgb + (1.0 - alpha) * dst_roi
        dst_bgr[y0:y1, x0:x1] = out_roi.astype(np.uint8)

    def _spawn_particles(self, out_bgr, kind, n=2):
        """
        kind: "smile" or "sad"
        Spawns particles from random positions with random velocities.
        """
        H, W = out_bgr.shape[:2]
        base = self.icon_smile if kind == "smile" else self.icon_sad

        lifetime_s = 1.0
        now = time.time()

        for _ in range(n):
            # scale for resize_width=416
            scale = float(np.clip(W / 900.0, 0.30, 0.75))
            size = int(base.shape[1] * scale)
            size = max(24, min(size, 72))
            icon = cv2.resize(base, (size, size), interpolation=cv2.INTER_AREA)

            # bottom 1/3 region
            y_min = int(H * (2.0 / 3.0))
            y_max = max(y_min + 1, H - size)

            x = int(self.rng.integers(0, max(1, W - size)))
            y = int(self.rng.integers(y_min, y_max))

            vx = float(self.rng.uniform(-1.2, 1.2))
            vy = float(self.rng.uniform(-2.0, -0.8))

            self.particles.append({
                "icon": icon,
                "x": float(x),
                "y": float(y),
                "vx": vx,
                "vy": vy,
                "born": now,
                "life": lifetime_s,
                "angle": float(self.rng.uniform(0, 360)),
                "spin": float(self.rng.uniform(-4.0, 4.0)),
            })
    
    def _render_particles(self, out_bgr):
        """Move + draw particles; drop expired ones."""
        H, W = out_bgr.shape[:2]
        now = time.time()
        y_floor = int(H * (2.0 / 3.0))

        alive = []
        for p in self.particles:
            age = now - p["born"]
            if age >= p["life"]:
                continue

            # Fade: 1.0 -> 0.0 over 1 second
            alpha_mul = 1.0 - (age / p["life"])

            # Move (optional)
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["angle"] += p["spin"]

            # Keep them in bottom 1/3: clamp at y_floor
            if p["y"] < y_floor:
                p["y"] = float(y_floor)
                p["vy"] = 0.0

            icon = p["icon"]
            ih, iw = icon.shape[:2]

            # Rotate (optional)
            M = cv2.getRotationMatrix2D((iw / 2, ih / 2), p["angle"], 1.0)
            rotated = cv2.warpAffine(
                icon, M, (iw, ih),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

            self._overlay_rgba(out_bgr, rotated, int(p["x"]), int(p["y"]), alpha_mul=alpha_mul)
            alive.append(p)

        self.particles = alive

    def _load_rgba(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # RGBA
        if img is None or img.shape[2] != 4:
            raise ValueError(f"Icon must be a PNG with alpha (RGBA): {path}")
        img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
        return img

    def _draw_status(self, out, text, ok=None):
        # ok: True (green), False (red), None (yellow)
        if ok is True:
            color = (0, 200, 0)
        elif ok is False:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        # background box for readability
        x, y = 20, 40
        cv2.rectangle(out, (x - 10, y - 28), (x + 900, y + 12), (0, 0, 0), -1)
        cv2.putText(out, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    def _draw_pose(self, out, lms, min_vis=0.6):
        H, W = out.shape[:2]

        # draw connections
        if draw_connections:
            for a, b in CONNECTIONS:
                la = lms[a.value]; lb = lms[b.value]
                if la.visibility < min_vis or lb.visibility < min_vis:
                    continue
                ax, ay = int(la.x * W), int(la.y * H)
                bx, by = int(lb.x * W), int(lb.y * H)
                cv2.line(out, (ax, ay), (bx, by), (255, 0, 0), 2)

        # draw landmark ids
        for lm_enum, lm_id in TARGET:
            lm = lms[lm_enum.value]
            if lm.visibility < min_vis:
                continue
            x, y = int(lm.x * W), int(lm.y * H)
            cv2.circle(out, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(out, str(lm_id), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    @property
    def run_state(self):
        return self.state_list[self.run_state_id]

    def count_detected_lm(self, lms):
        count = 0
        for lm_enum, _ in TARGET:
            if lms[lm_enum.value].visibility >= MIN_VIS:
                count += 1
        return count

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            if flip:
                img = cv2.flip(img, 1)

            # Downscale for speed
            H0, W0 = img.shape[:2]
            if W0 != int(resize_width):
                new_w = int(resize_width)
                new_h = int(H0 * (new_w / W0))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_result = self.pose.process(rgb)
            rgb.flags.writeable = True

            out = img.copy()

            if pose_result.pose_landmarks:
                lms = pose_result.pose_landmarks.landmark
                detected_cnt = self.count_detected_lm(lms)

                if self.run_state == 'IDLE':
                    if detected_cnt >= len(TARGET):
                        self.warmup_count += 1
                    else:
                        self.warmup_count = 0

                    if self.warmup_count >= NUM_FRAME_TO_WARMUP:
                        self.run_state_id = 3  # WARMUP
                        self.warmup_count = 0

                self._draw_pose(out, lms, min_vis=MIN_VIS)

                # compute angles + rel positions for text
                feat = make_feature_vector_pose(lms, self.ids, min_vis=MIN_VIS)
                
                if feat is not None and detected_cnt >= len(TARGET):
                    min_err, _ = min_template_sae(
                        feat, 
                        template_seq, 
                        w_elbow_angle=W_ELBOW_ANGLE, 
                        w_wrist_angle=W_WRIST_ANGLE)
                else:
                    min_err = float('inf')

                if self.run_state != "IDLE":
                    if min_err <= WRONG_ANGLE_THRESHOLD:
                        self.CORRECT_COUNT += 1
                        self.WRONG_COUNT -= 1
                    else:
                        self.CORRECT_COUNT -= 1
                        self.WRONG_COUNT += 1
                    self.CORRECT_COUNT = min(max(self.CORRECT_COUNT, 0), K_CONSECUTIVE_FRAMES)
                    self.WRONG_COUNT = min(max(self.WRONG_COUNT, 0), K_CONSECUTIVE_FRAMES)

                if self.run_state == 'WARMUP':
                    if self.WRONG_COUNT >= K_CONSECUTIVE_FRAMES:
                        self.run_state_id = 1  # WRONG
                        self.CORRECT_COUNT = 0
                    if self.CORRECT_COUNT >= K_CONSECUTIVE_FRAMES:
                        self.run_state_id = 2  # CORRECT
                        self.WRONG_COUNT = 0

                if self.run_state == 'CORRECT' and self.WRONG_COUNT >= K_CONSECUTIVE_FRAMES:
                    self.run_state_id = 1  # WRONG
                    self.CORRECT_COUNT = 0

                if self.run_state == 'WRONG' and self.CORRECT_COUNT >= K_CONSECUTIVE_FRAMES:
                    self.run_state_id = 2  # CORRECT
                    WRONG_COUNT = 0

                if self.run_state == 'IDLE':
                    self.status_text = "Hệ thống chưa phát hiện đủ các khớp tay,\nvui lòng đứng xa ra"
                    self.text_state = None
                elif self.run_state == 'WARMUP':
                    self.status_text = "Đang theo dõi các khớp tay"
                    self.text_state = None
                elif self.run_state == 'CORRECT':
                    self.status_text = "Bạn đang thực hiện đúng"
                    self.text_state = True
                elif self.run_state == 'WRONG':
                    self.status_text = "Bạn đang thực hiện sai, vui lòng điều chỉnh"
                    self.text_state = False

                out = draw_status_pil(out, self.status_text, ok=self.text_state)

                if self.run_state == "CORRECT":
                    # spawn rate: every frame_idx%2 -> fewer particles
                    if self.frame_idx % 2 == 0:
                        self._spawn_particles(out, kind="smile", n=1)
                elif self.run_state == "WRONG":
                    if self.frame_idx % 2 == 0:
                        self._spawn_particles(out, kind="sad", n=1)
                else:
                    # optional: clear particles when not correct/wrong
                    self.particles.clear()
                
                # Always render existing particles
                self._render_particles(out)

                MAX_PARTICLES = 3
                if len(self.particles) > MAX_PARTICLES:
                    self.particles = self.particles[-MAX_PARTICLES:]

                self.frame_idx += 1

            return av.VideoFrame.from_ndarray(out, format="bgr24")
        except Exception as e:
            st.error(f"Error in video processing: {e}")
            traceback.print_exc()
            return frame

webrtc_ctx = webrtc_streamer(
    key="pose",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PoseProcessor,
    rtc_configuration={"iceServers": token.ice_servers},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
