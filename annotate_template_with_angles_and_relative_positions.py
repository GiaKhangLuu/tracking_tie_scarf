import cv2
import mediapipe as mp
from tqdm import tqdm
import numpy as np

from pose_template_features import angle_3pts, make_feature_vector_pose, pose_ids_for_indices

mp_pose = mp.solutions.pose
LM = mp_pose.PoseLandmark

# --------- CONFIG ----------
TEMPLATE_VIDEO = "./asset/correct_tie.mp4"
OUT_VIDEO = "./asset/output_annotated.avi"
OUT_NPY = "./template_seq.npy"

TARGET_LMS = [
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,   # 11, 12
    LM.LEFT_ELBOW, LM.RIGHT_ELBOW,         # 13, 14
    LM.LEFT_WRIST, LM.RIGHT_WRIST,         # 15, 16
    LM.LEFT_INDEX, LM.RIGHT_INDEX,
]

CONNECTIONS = [
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW),
    (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    (LM.LEFT_WRIST, LM.LEFT_INDEX),

    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    (LM.RIGHT_WRIST, LM.RIGHT_INDEX),

    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
]

DOT_COLOR = (0, 255, 0)
LINE_COLOR = (255, 0, 0)
DOT_RADIUS = 5
LINE_THICKNESS = 2
MIN_VIS = 0.6

def pt_xy(lms, idx, w, h):
    lm = lms[idx.value]
    return int(lm.x * w), int(lm.y * h), lm.visibility

def main():
    K = 2  # append template feature only every K frames

    cap = cv2.VideoCapture(TEMPLATE_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {TEMPLATE_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = None

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps if fps > 0 else 30.0, (w, h))
    if not out.isOpened():
        raise RuntimeError("Could not open VideoWriter (MJPG).")

    ids = pose_ids_for_indices()
    seq = []
    frame_i = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, tqdm(total=total, desc=f"Annotate template (K={K})", unit="frame") as pbar:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            left_elbow = None
            right_elbow = None
            left_wrist = None
            right_wrist = None

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark

                # Only append the template vector every K frames
                if frame_i % K == 0:
                    feat = make_feature_vector_pose(lms, ids, min_vis=MIN_VIS)
                    if feat is not None:
                        seq.append(feat)

                # Visibility helper
                def vis_ok(i): return lms[i].visibility >= MIN_VIS

                # Indices
                LS, RS, LE, RE, LW, RW, LI, RI = 11, 12, 13, 14, 15, 16, 19, 20

                # Angles
                if vis_ok(LS) and vis_ok(LE) and vis_ok(LW):
                    a = (lms[LS].x, lms[LS].y)
                    b = (lms[LE].x, lms[LE].y)
                    c = (lms[LW].x, lms[LW].y)
                    left_elbow = angle_3pts(a, b, c)

                if vis_ok(RS) and vis_ok(RE) and vis_ok(RW):
                    a = (lms[RS].x, lms[RS].y)
                    b = (lms[RE].x, lms[RE].y)
                    c = (lms[RW].x, lms[RW].y)
                    right_elbow = angle_3pts(a, b, c)

                if vis_ok(LE) and vis_ok(LW) and vis_ok(LI):
                    a = (lms[LE].x, lms[LE].y)
                    b = (lms[LW].x, lms[LW].y)
                    c = (lms[LI].x, lms[LI].y)
                    left_wrist = angle_3pts(a, b, c)

                if vis_ok(RE) and vis_ok(RW) and vis_ok(RI):
                    a = (lms[RE].x, lms[RE].y)
                    b = (lms[RW].x, lms[RW].y)
                    c = (lms[RI].x, lms[RI].y)
                    right_wrist = angle_3pts(a, b, c)

                # Draw connections
                for a_lm, b_lm in CONNECTIONS:
                    ax, ay, av = pt_xy(lms, a_lm, w, h)
                    bx, by, bv = pt_xy(lms, b_lm, w, h)
                    if av < MIN_VIS or bv < MIN_VIS:
                        continue
                    cv2.line(frame, (ax, ay), (bx, by), LINE_COLOR, LINE_THICKNESS)

                # Draw key points
                for lm_enum in [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
                                LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
                                LM.LEFT_WRIST, LM.RIGHT_WRIST,
                                LM.LEFT_INDEX, LM.RIGHT_INDEX]:
                    x, y, v = pt_xy(lms, lm_enum, w, h)
                    if v < MIN_VIS:
                        continue
                    cv2.circle(frame, (x, y), DOT_RADIUS, DOT_COLOR, -1)

            # Overlay text
            text1 = f"L_elbow: {left_elbow:.1f}" if left_elbow is not None else "L_elbow: --"
            text2 = f"R_elbow: {right_elbow:.1f}" if right_elbow is not None else "R_elbow: --"
            text3 = f"L_wrist: {left_wrist:.1f}" if left_wrist is not None else "L_wrist: --"
            text4 = f"R_wrist: {right_wrist:.1f}" if right_wrist is not None else "R_wrist: --"

            # show whether this frame was sampled into the template array
            text5 = f"sampled: {'YES' if (frame_i % K == 0) else 'no'}  frame={frame_i}"

            cv2.putText(frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, text2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, text3, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, text4, (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, text5, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            out.write(frame)
            pbar.update(1)
            frame_i += 1

    cap.release()
    out.release()

    if len(seq) == 0:
        raise RuntimeError("No usable frames for template features (try lowering MIN_VIS or K).")

    seq = np.stack(seq, axis=0).astype(np.float32)
    np.save(OUT_NPY, seq)

    print(f"Saved annotated template video: {OUT_VIDEO}")
    print(f"Saved template feature sequence (every K={K} frames): {OUT_NPY} shape={seq.shape}")

if __name__ == "__main__":
    main()
