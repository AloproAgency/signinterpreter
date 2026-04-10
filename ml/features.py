"""
V7 - Feature extraction module.
Extracts 165 normalized features per frame (upper-body only).

Layout:
  - Upper-body pose (positions in body frame): 13 landmarks x 3 = 39
    Normalized by shoulder midpoint and shoulder distance
  - Left hand SHAPE (relative to wrist): 21 landmarks x 3 = 63
    Normalized by wrist position and own size → captures finger configuration
  - Right hand SHAPE (relative to wrist): 21 landmarks x 3 = 63
    Same normalization
  Total: 165 features

The split design:
  - Pose tells WHERE the hand is in body space
  - Hand shape tells WHAT the hand is doing (fist, open, pointing, etc.)
  - Hand shape is NOT crushed by shoulder normalization → fingers are visible
  - Hand shape features are scaled (HAND_WEIGHT) to balance with pose features
"""

import numpy as np

# MediaPipe Pose landmark indices (upper body only)
# Reference: https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
UPPER_BODY_LANDMARKS = [
    0,   # nose
    2,   # left eye
    5,   # right eye
    7,   # left ear
    8,   # right ear
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip (used for chest reference, optional)
    24,  # right hip (used for chest reference, optional)
]

# Actually, since we want sit/stand invariance, drop hips
UPPER_BODY_LANDMARKS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]
# 0:nose, 2:l_eye, 5:r_eye, 7:l_ear, 8:r_ear,
# 11:l_shoulder, 12:r_shoulder, 13:l_elbow, 14:r_elbow,
# 15:l_wrist, 16:r_wrist, 17:l_pinky, 18:r_pinky

N_POSE_LANDMARKS = len(UPPER_BODY_LANDMARKS)  # 13

# Per hand: 21 landmarks × 3 coords + 3 (palm orientation vector) = 66
HAND_FEATURES = 21 * 3 + 3  # 66 per hand

# Per-frame features: pose(39) + lh(66) + rh(66) = 171
FRAME_FEATURE_DIM = N_POSE_LANDMARKS * 3 + HAND_FEATURES + HAND_FEATURES  # 171

# Wrist velocity: NORMALIZED direction of movement (unit vector)
# for left wrist (3) + right wrist (3) = 6
VELOCITY_DIM = 6
VELOCITY_WEIGHT = 3.0  # multiplied by the normalized direction vector (magnitude 1)

# Total feature dim after velocity enrichment
FEATURE_DIM = FRAME_FEATURE_DIM + VELOCITY_DIM  # 171 + 6 = 177

# Hand shape features importance multiplier.
HAND_WEIGHT = 3.0

# Palm orientation importance multiplier.
PALM_WEIGHT = 5.0


def get_landmarks(results):
    """Extract pose, left hand, right hand landmarks as numpy arrays."""
    pose = None
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.pose_landmarks.landmark])

    lh = None
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark])

    rh = None
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark])

    return pose, lh, rh


def normalize(landmarks_xyz, midpoint, scale):
    """Translate to origin, scale by reference distance."""
    return (landmarks_xyz - midpoint) / scale


def normalize_hand(hand_xyz):
    """
    Normalize a hand to make its SHAPE invariant to position and size.
    - Center on wrist (landmark 0)
    - Scale by max distance from wrist to any landmark
    Returns: (21, 3) normalized hand
    """
    wrist = hand_xyz[0]
    centered = hand_xyz - wrist
    distances = np.linalg.norm(centered, axis=1)
    scale = distances.max()
    if scale < 1e-6:
        scale = 1.0
    return centered / scale


def palm_orientation(hand_xyz):
    """
    Compute palm normal vector using 3 points:
    - 0: wrist
    - 5: base of index finger
    - 17: base of pinky finger

    The cross product of (wrist→index) × (wrist→pinky) gives
    a vector perpendicular to the palm. Its direction tells us
    which way the palm is facing (towards camera, away, left, right, up, down).

    Returns: (3,) normalized direction vector
    """
    wrist = hand_xyz[0]
    index_base = hand_xyz[5]
    pinky_base = hand_xyz[17]

    v1 = index_base - wrist
    v2 = pinky_base - wrist
    normal = np.cross(v1, v2)

    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return np.zeros(3)
    return normal / norm


def extract_hand_features(hand_xyz):
    """
    Extract hand shape (63) + palm orientation (3) = 66 features.
    Hand shape is weighted by HAND_WEIGHT, palm by PALM_WEIGHT.
    """
    if hand_xyz is None:
        return np.zeros(HAND_FEATURES)

    hand_norm = normalize_hand(hand_xyz)
    shape_features = hand_norm.flatten() * HAND_WEIGHT  # 63

    palm_dir = palm_orientation(hand_xyz) * PALM_WEIGHT  # 3

    return np.concatenate([shape_features, palm_dir])  # 66


def extract_features_from_results(results):
    """
    Returns 171 normalized features per frame (upper body only).

    Layout:
      [0:39]    Upper-body pose (body-normalized)
      [39:105]  Left hand: shape (63) + palm orientation (3) = 66
      [105:171] Right hand: shape (63) + palm orientation (3) = 66
    """
    pose, lh, rh = get_landmarks(results)

    pose_features = np.zeros(N_POSE_LANDMARKS * 3)

    if pose is not None:
        sl = pose[11]
        sr = pose[12]
        midpoint = (sl + sr) / 2
        scale = np.linalg.norm(sr - sl)
        if scale < 1e-6:
            scale = 1.0
        upper_pose = pose[UPPER_BODY_LANDMARKS]
        pose_features = normalize(upper_pose, midpoint, scale).flatten()

    lh_features = extract_hand_features(lh)
    rh_features = extract_hand_features(rh)

    return np.concatenate([pose_features, lh_features, rh_features])


def add_wrist_velocity(sequence):
    """
    Enrich a sequence (N, 171) with wrist velocity (N, 6) → (N, 177).

    Wrist positions are in the pose features:
    - Left wrist (landmark 15, index 9 in UPPER_BODY_LANDMARKS): features[27:30]
    - Right wrist (landmark 16, index 10 in UPPER_BODY_LANDMARKS): features[30:33]

    Velocity = position[t] - position[t-1], weighted by VELOCITY_WEIGHT.
    First frame velocity = 0.

    This captures the DIRECTION of movement:
    - Arm coming towards body = negative velocity
    - Arm going away from body = positive velocity
    """
    seq = np.asarray(sequence, dtype='float32')
    if seq.ndim != 2:
        return seq

    # If already enriched, return as-is
    if seq.shape[1] == FEATURE_DIM:
        return seq

    n_frames = seq.shape[0]

    # Wrist indices in the feature vector (after pose normalization)
    # UPPER_BODY_LANDMARKS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]
    # Index of landmark 15 (l_wrist) = 9, so features[9*3 : 9*3+3] = [27:30]
    # Index of landmark 16 (r_wrist) = 10, so features[10*3 : 10*3+3] = [30:33]
    lw_idx = 9 * 3   # 27
    rw_idx = 10 * 3  # 30

    velocity = np.zeros((n_frames, VELOCITY_DIM), dtype='float32')

    for i in range(1, n_frames):
        # Left wrist: compute direction vector, normalize to unit length
        lw_diff = seq[i, lw_idx:lw_idx+3] - seq[i-1, lw_idx:lw_idx+3]
        lw_norm = np.linalg.norm(lw_diff)
        if lw_norm > 1e-6:
            velocity[i, 0:3] = (lw_diff / lw_norm) * VELOCITY_WEIGHT
        # else: stays zero (no movement)

        # Right wrist: same
        rw_diff = seq[i, rw_idx:rw_idx+3] - seq[i-1, rw_idx:rw_idx+3]
        rw_norm = np.linalg.norm(rw_diff)
        if rw_norm > 1e-6:
            velocity[i, 3:6] = (rw_diff / rw_norm) * VELOCITY_WEIGHT

    return np.concatenate([seq, velocity], axis=1)
