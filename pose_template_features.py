import numpy as np

def angle_3pts(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def _pt(lms, idx):
    lm = lms[idx]
    return np.array([lm.x, lm.y], dtype=np.float32), lm.visibility

def make_feature_vector_pose(lms, ids, min_vis=0.6):
    """
    lms: mediapipe pose landmarks list
    ids: dict of name -> landmark_index (int)
    returns np.array (D,) or None
    """
    pts = {}
    for k, idx in ids.items():
        p, vis = _pt(lms, idx)
        if vis < min_vis:
            return None
        pts[k] = p

    try:
        # angles
        left_elbow = angle_3pts(pts["LS"], pts["LE"], pts["LW"])
        left_wrist = angle_3pts(pts["LE"], pts["LW"], pts["LI"])
        right_elbow = angle_3pts(pts["RS"], pts["RE"], pts["RW"])
        right_wrist = angle_3pts(pts["RE"], pts["RW"], pts["RI"])

        return np.array([
            left_elbow, right_elbow,
            left_wrist, right_wrist,
        ], dtype=np.float32)
    except Exception as e:
        print(f"Error computing feature vector: {e}")
        return None

def pose_ids_for_indices():
    # Pose indices you requested:
    # 11,12 shoulders; 13,14 elbows; 15,16 wrists; 17-22 hand points
    return {
        "LS": 11, "RS": 12,
        "LE": 13, "RE": 14,
        "LW": 15, "RW": 16,
        "LP": 17, "RP": 18,
        "LI": 19, "RI": 20,
        "LT": 21, "RT": 22,
    }

def min_template_sae(user_f, template_seq, w_elbow_angle=1, w_wrist_angle=1):
    da = np.abs(template_seq[:, :2] - user_f[:2]).sum(axis=1) * w_elbow_angle
    db = np.abs(template_seq[:, 2:4] - user_f[2:4]).sum(axis=1) * w_wrist_angle
    err = (da + db) * 0.5
    idx = int(np.argmin(err))
    return float(err[idx]), idx
