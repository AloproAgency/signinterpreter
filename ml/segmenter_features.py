"""
Signal functions used by EnrichedSignSegmenter.

Each function consumes either the current 171-dim frame vector OR a short
rolling history of such vectors, and returns a scalar in [0, 1] (or a plain
bool) measuring some aspect of "is this frame part of a sign?".

The 7 signals are:
  1. motion                 — hand-dims motion energy (normalised)
  2. in_signing_zone        — at least one wrist in front of torso, above waist
  3. handshape_formed       — fingers are in a deliberate configuration
  4. has_bell_velocity      — energy history shows accel→decel bell curve
  5. head_coherent          — head not wandering chaotically
  6. trajectory_sign_like   — wrist trajectory has mid straightness (not random, not ruler-straight)
  7. bimanual_coherent      — when both hands active, motion correlates (None if unimanual)

`compute_sign_score` fuses them into a single [0, 1] likelihood.

All feature arrays follow the layout in ml/features.py:
  pose[0:39] — 13 upper-body landmarks × (x, y, z)
  LH shape[39:102] (scaled by HAND_WEIGHT=3)
  LH palm normal[102:105] (scaled by PALM_WEIGHT=5)
  RH shape[105:168]  RH palm[168:171]
Wrist indices in pose: l_wrist=9 → feat 27:30, r_wrist=10 → feat 30:33.
"""
import numpy as np

from ml.features import HAND_WEIGHT

# Normalisation reference for motion_energy → [0, 1].
MOTION_REF = 0.04

# Signing zone (body-normalised coordinates: shoulder-midpoint origin,
# shoulder-distance scale, y growing downward).
ZONE_Y_TOP = -2.2    # forehead
ZONE_Y_BOTTOM = 0.8  # waist
ZONE_X_HALF = 1.5    # lateral half-extent (shoulder-widths)

# Handshape thresholds
HAND_FORMED_SPREAD_MIN = 0.04   # fingertips moderately spread
HAND_FORMED_CURL_MIN = 0.6      # any finger bent ≥ 34°

# Trajectory straightness ranges
TRAJ_SIGN_MIN = 0.25    # below = chaotic (random motion)
TRAJ_SIGN_MAX = 0.95    # above = ruler-straight (transition)

# Weights for the composite score. Motion + zone are gating essentials,
# handshape / velocity_profile are strong corroborators, the rest tune.
SIGNAL_WEIGHTS = {
    'motion':           2.0,
    'in_zone':          2.0,
    'handshape':        1.5,
    'velocity_profile': 1.0,
    'head_ok':          0.5,
    'trajectory':       1.0,
    'bimanual':         0.5,
}


# ─────────────────────────── 1. motion ───────────────────────────

def normalised_motion(motion_energy):
    """Map raw motion energy (mean |Δ| on hand dims) to [0, 1]."""
    return float(min(1.0, motion_energy / MOTION_REF))


# ──────────────────────── 2. in_signing_zone ─────────────────────

def in_signing_zone(features):
    """True if at least one wrist sits inside the front-of-torso zone."""
    if features is None or len(features) < 33:
        return False
    lw_x, lw_y = float(features[27]), float(features[28])
    rw_x, rw_y = float(features[30]), float(features[31])
    in_y = lambda y: ZONE_Y_TOP <= y <= ZONE_Y_BOTTOM
    in_x = lambda x: abs(x) < ZONE_X_HALF
    return (in_y(lw_y) and in_x(lw_x)) or (in_y(rw_y) and in_x(rw_x))


# ───────────────────── 3. handshape_is_formed ────────────────────

def _hand_landmarks(features, side='right'):
    """Return (21, 3) landmarks for one hand, undoing HAND_WEIGHT scaling."""
    if features is None or len(features) < 168:
        return None
    block = features[105:168] if side == 'right' else features[39:102]
    arr = np.asarray(block, dtype='float32') / HAND_WEIGHT
    return arr.reshape(21, 3)


def _finger_curl(landmarks, mcp, pip, tip):
    """Angle at the PIP joint between the MCP-PIP and PIP-TIP segments."""
    v1 = landmarks[pip] - landmarks[mcp]
    v2 = landmarks[tip] - landmarks[pip]
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.arccos(cos_a))


def handshape_formed(features):
    """Return 1.0 if the dominant hand is in a deliberate configuration,
    else fade towards 0.0. Takes the hand that is actually present."""
    rh = _hand_landmarks(features, 'right')
    lh = _hand_landmarks(features, 'left')
    scores = []
    for h in (rh, lh):
        if h is None or np.all(h == 0):
            continue
        fingertips = h[[4, 8, 12, 16, 20]]   # thumb/index/middle/ring/pinky tip
        spread = float(np.std(fingertips))
        # Curl of each finger (mcp-pip-tip triplets)
        curls = [
            _finger_curl(h, 2, 3, 4),    # thumb IP
            _finger_curl(h, 5, 6, 8),    # index
            _finger_curl(h, 9, 10, 12),  # middle
            _finger_curl(h, 13, 14, 16), # ring
            _finger_curl(h, 17, 18, 20), # pinky
        ]
        max_curl = max(curls) if curls else 0.0
        # Formed = either visibly spread OR at least one finger clearly bent.
        score = 0.0
        if spread > HAND_FORMED_SPREAD_MIN:
            score = max(score, min(1.0, (spread - HAND_FORMED_SPREAD_MIN) * 20))
        if max_curl > HAND_FORMED_CURL_MIN:
            score = max(score, min(1.0, (max_curl - HAND_FORMED_CURL_MIN) * 1.5))
        scores.append(score)
    if not scores:
        return 0.0
    return float(max(scores))


# ────────────────────── 4. velocity_profile ──────────────────────

def has_bell_velocity(energy_history, window=6):
    """True if the motion_energy history shows a bell (accel→decel) — quadratic
    fit with negative curvature over the last `window` frames."""
    if len(energy_history) < window:
        return False
    y = np.asarray(energy_history[-window:], dtype='float32')
    if y.max() - y.min() < 0.01:
        return False
    t = np.arange(window, dtype='float32')
    a, _, _ = np.polyfit(t, y, 2)
    return bool(a < -0.001)


# ───────────────────────── 5. head_coherent ──────────────────────

def head_coherent(pose_history, window=6):
    """True if the nose has either stayed still or oscillated in a single
    axis (rhythmic nod / shake = linguistically-meaningful non-manual)."""
    if len(pose_history) < window:
        return True   # not enough data — don't penalise
    noses = np.asarray([p[0:3] for p in pose_history[-window:]], dtype='float32')
    total = float(np.sum(np.linalg.norm(np.diff(noses, axis=0), axis=1)))
    if total < 0.06:
        return True   # almost still
    if total > 0.25:
        return False  # head clearly wandering
    # Moderate motion: accept if movement concentrated on a single axis.
    span = noses.max(axis=0) - noses.min(axis=0)
    dominant = span.max()
    secondary = np.sort(span)[-2]
    return dominant > 2.5 * secondary


# ────────────────── 6. trajectory_sign_like ──────────────────────

def trajectory_sign_like(wrist_history, window=8):
    """Score in [0, 1] based on trajectory straightness of the dominant wrist."""
    if len(wrist_history) < window:
        return 0.5   # undetermined
    w = np.asarray(wrist_history[-window:], dtype='float32')
    diffs = np.linalg.norm(np.diff(w, axis=0), axis=1)
    path = float(diffs.sum())
    if path < 1e-5:
        return 0.0   # totally still → probably a hold, but motion=0 will gate it
    disp = float(np.linalg.norm(w[-1] - w[0]))
    straightness = disp / (path + 1e-6)
    if straightness < TRAJ_SIGN_MIN:
        return float(straightness / TRAJ_SIGN_MIN)
    if straightness > TRAJ_SIGN_MAX:
        return float(max(0.0, 1.0 - (straightness - TRAJ_SIGN_MAX) / (1.0 - TRAJ_SIGN_MAX)))
    return 1.0


# ────────────────── 7. bimanual_coherent ─────────────────────────

def bimanual_coherent(lh_history, rh_history, window=6):
    """Return None if unimanual, else correlation of L/R motion magnitudes
    as a score in [0, 1]. Coherent bimanual signs have positively
    correlated motion profiles."""
    if len(lh_history) < window or len(rh_history) < window:
        return None
    lh = np.asarray(lh_history[-window:], dtype='float32')
    rh = np.asarray(rh_history[-window:], dtype='float32')
    if np.all(lh == 0) or np.all(rh == 0):
        return None   # one hand absent → unimanual
    lh_mag = np.linalg.norm(np.diff(lh, axis=0), axis=1)
    rh_mag = np.linalg.norm(np.diff(rh, axis=0), axis=1)
    if lh_mag.std() < 1e-4 or rh_mag.std() < 1e-4:
        return None
    corr = float(np.corrcoef(lh_mag, rh_mag)[0, 1])
    return max(0.0, corr)   # only positive correlation counts


# ────────────────────── compute composite ────────────────────────

def compute_sign_score(
    motion_energy,
    features,
    energy_history,
    pose_history,
    wrist_history,
    lh_history,
    rh_history,
):
    """Weighted fusion of the 7 signals → [0, 1]."""
    signals = {
        'motion':           normalised_motion(motion_energy),
        'in_zone':          1.0 if in_signing_zone(features) else 0.0,
        'handshape':        handshape_formed(features),
        'velocity_profile': 1.0 if has_bell_velocity(energy_history) else 0.0,
        'head_ok':          1.0 if head_coherent(pose_history) else 0.0,
        'trajectory':       trajectory_sign_like(wrist_history),
        'bimanual':         bimanual_coherent(lh_history, rh_history),
    }
    # Drop signals that are undetermined (None).
    effective = {k: v for k, v in signals.items() if v is not None}
    weights = {k: SIGNAL_WEIGHTS[k] for k in effective}
    total_w = sum(weights.values())
    if total_w <= 0:
        return 0.0, effective
    score = sum(effective[k] * weights[k] for k in effective) / total_w
    return float(score), effective
