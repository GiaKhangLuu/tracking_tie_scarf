import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import shutil

from pose_template_features import make_feature_vector_pose, pose_ids_for_indices

mp_pose = mp.solutions.pose

LM = mp_pose.PoseLandmark

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

def angle_3pts(a, b, c):
    import numpy as _np
    a = _np.array(a); b = _np.array(b); c = _np.array(c)
    ba = a - b
    bc = c - b
    denom = (_np.linalg.norm(ba) * _np.linalg.norm(bc)) + 1e-9
    cosang = _np.clip(_np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(_np.degrees(_np.arccos(cosang)))

def pt_xy(lms, lm_enum, w, h):
    lm = lms[lm_enum.value]
    return int(lm.x * w), int(lm.y * h), lm.visibility

def min_template_sae(user_f, template_seq, w_elbow_angle=1, w_wrist_angle=1):
    da = np.abs(template_seq[:, :2] - user_f[:2]).sum(axis=1) * w_elbow_angle
    db = np.abs(template_seq[:, 2:4] - user_f[2:4]).sum(axis=1) * w_wrist_angle
    err = (da + db) * 0.5
    idx = int(np.argmin(err))
    return float(err[idx]), idx

def find_runs(mask):
    """mask: list[bool]. Returns list of (start, end) inclusive runs where mask is True."""
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs

def save_segment_annotated(
    input_video,
    out_path,
    start_frame,
    end_frame,
    errors,
    fps,
    w,
    h,
    threshold,          # kept in signature though not printed below (you can add if you want)
    min_vis=0.6,
    slow_factor=2.0):
    cap = cv2.VideoCapture(input_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if fps is None or fps <= 0:
        fps = 30.0
    out_fps = max(1.0, fps / float(slow_factor))

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for: {out_path}")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        frame_i = start_frame
        while frame_i <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            left_elbow = None
            right_elbow = None
            left_wrist = None
            right_wrist = None
            lw_rel = None
            rw_rel = None

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark

                def vis_ok(i): return lms[i].visibility >= min_vis

                # indices
                LS, RS, LE, RE, LW, RW = 11, 12, 13, 14, 15, 16
                LP, RP, LI, RI, LT, RT = 17, 18, 19, 20, 21, 22  # if you want later

                # --- compute angles ---
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

                # "wrist angle" = angle at wrist using (elbow, wrist, index)
                # (this matches your feature split where wrist-angle is its own term)
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

                # --- relative wrist positions ---
                if vis_ok(LS) and vis_ok(RS) and vis_ok(LW) and vis_ok(RW):
                    LS_xy = np.array([lms[LS].x, lms[LS].y], dtype=np.float32)
                    RS_xy = np.array([lms[RS].x, lms[RS].y], dtype=np.float32)
                    LW_xy = np.array([lms[LW].x, lms[LW].y], dtype=np.float32)
                    RW_xy = np.array([lms[RW].x, lms[RW].y], dtype=np.float32)

                    shoulder_w = np.linalg.norm(LS_xy - RS_xy) + 1e-9
                    lw_rel = (LW_xy - LS_xy) / shoulder_w
                    rw_rel = (RW_xy - RS_xy) / shoulder_w

                # --- draw pose connections ---
                H, W = frame.shape[:2]
                for a_lm, b_lm in CONNECTIONS:
                    ax, ay, av = pt_xy(lms, a_lm, W, H)
                    bx, by, bv = pt_xy(lms, b_lm, W, H)
                    if av < min_vis or bv < min_vis:
                        continue
                    cv2.line(frame, (ax, ay), (bx, by), LINE_COLOR, LINE_THICKNESS)

                # --- draw key points + ids ---
                for lm_enum in [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
                                LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
                                LM.LEFT_WRIST, LM.RIGHT_WRIST,
                                LM.LEFT_INDEX, LM.RIGHT_INDEX]:
                    x, y, v = pt_xy(lms, lm_enum, W, H)
                    if v < min_vis:
                        continue
                    cv2.circle(frame, (x, y), DOT_RADIUS, DOT_COLOR, -1)
                    cv2.putText(frame, str(lm_enum.value), (x + 6, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, DOT_COLOR, 1)

            # --- overlay your text exactly ---
            text1 = f"L_elbow: {left_elbow:.1f}" if left_elbow is not None else "L_elbow: --"
            text2 = f"R_elbow: {right_elbow:.1f}" if right_elbow is not None else "R_elbow: --"
            text3 = f"L_wrist: {left_wrist:.1f}" if left_wrist is not None else "L_wrist: --"
            text4 = f"R_wrist: {right_wrist:.1f}" if right_wrist is not None else "R_wrist: --"

            if lw_rel is not None:
                text5 = f"LW_rel: ({lw_rel[0]:+.2f}, {lw_rel[1]:+.2f})"
            else:
                text5 = "LW_rel: (--, --)"

            if rw_rel is not None:
                text6 = f"RW_rel: ({rw_rel[0]:+.2f}, {rw_rel[1]:+.2f})"
            else:
                text6 = "RW_rel: (--, --)"
            
            text7 = f"Error: {errors[frame_i-start_frame]:.2f}"

            cv2.putText(frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, text2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, text3, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, text4, (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, text5, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, text6, (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, text7, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out.write(frame)
            frame_i += 1

    out.release()
    cap.release()

def save_segment(input_video, out_path, start_frame, end_frame, fps, w, h, slow_factor=2.0):
    cap = cv2.VideoCapture(input_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if fps is None or fps <= 0:
        fps = 30.0

    out_fps = max(1.0, fps / float(slow_factor))  # 2.0 => half fps => 2x slower

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for: {out_path}")

    frame_i = start_frame
    while frame_i <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
        frame_i += 1

    out.release()
    cap.release()

def extract_wrong_segments(
    input_video,
    template_seq_path="template_seq.npy",
    out_dir="wrong_segments",
    threshold=2.5,
    k_consecutive=20,        # minimum run length to keep
    min_vis=0.6,
    w_elbow_angle=1,
    w_wrist_angle=1,
    w_pos=1.0,
):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    template_seq = np.load(template_seq_path).astype(np.float32)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = None

    ids = pose_ids_for_indices()

    # Pass 1: compute wrong mask per frame
    wrong_mask = []
    errors = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, tqdm(total=total, desc="Compute errors", unit="frame") as pbar:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                feat = make_feature_vector_pose(res.pose_landmarks.landmark, ids, min_vis=min_vis)
                if feat is not None:
                    min_err, _best = min_template_sae(feat, template_seq, w_elbow_angle=w_elbow_angle, w_wrist_angle=w_wrist_angle)
                    errors.append(min_err)
                    wrong_mask.append(min_err > threshold)
                else:
                    # if visibility too low, you can choose:
                    # - treat as wrong, or
                    # - treat as not-wrong, or
                    # - treat as "unknown"
                    errors.append(float("inf"))
                    wrong_mask.append(True)
            else:
                errors.append(float("inf"))
                wrong_mask.append(True)

            pbar.update(1)

    cap.release()

    # Find consecutive wrong runs
    runs = find_runs(wrong_mask)

    # Filter by k_consecutive
    kept = [(s, e) for (s, e) in runs if (e - s + 1) >= k_consecutive]

    print(f"Found wrong runs: {len(runs)}; kept (len >= {k_consecutive}): {len(kept)}")

    # Pass 2: save each kept run as its own clip
    #for idx, (s, e) in enumerate(kept, start=1):
    #    out_path = os.path.join(out_dir, f"wrong_segment_{idx:02d}_frames_{s:06d}-{e:06d}.avi")
    #    save_segment(input_video, out_path, s, e, fps, w, h)
    #    print(f"Saved: {out_path}")
    
    for idx, (s, e) in enumerate(kept, start=1):
        out_path = os.path.join(out_dir, f"wrong_segment_{idx:02d}_frames_{s:06d}-{e:06d}.avi")
        save_segment_annotated(
            input_video=input_video,
            out_path=out_path,
            start_frame=s,
            end_frame=e,
            errors=errors[s:e+1],
            fps=fps,
            w=w,
            h=h,
            threshold=threshold,
            min_vis=min_vis,
            slow_factor=2.0,
        )
        print(f"Saved (annotated): {out_path}")

if __name__ == "__main__":
    extract_wrong_segments(
        input_video="./asset/tiktok_dance.mp4",
        template_seq_path="./template_seq.npy",
        out_dir="./wrong_segments",
        threshold=50,
        k_consecutive=50,
        min_vis=0.6,
        w_elbow_angle=0.7,
        w_wrist_angle=0.7,
        w_pos=1.0,
    )
