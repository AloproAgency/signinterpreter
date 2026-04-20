"""
Phonological sign descriptor v2 (Agent B experiment).

Extends the 34-dim base descriptor with ~22 signer-invariant features covering:
 - per-finger curl angles (5)
 - thumb geometry (2)
 - hand compactness (1)
 - movement enrichment: angular velocity, curvature, zero-crossings,
   autocorrelation, spectral centroid, velocity-profile timing (7)
 - location enrichment: dwell fraction, start/end xy, trajectory extent (6)
 - bimanual enrichment: min-dist, time offset, parallelism (3)
 - elegance: jerk RMS, handshape transition count (2)

Total new: 26. PHONO_DIM_V2 = 34 + 26 + 12 = 72.
v3 adds 12 signed DIRECTIONAL movement features — net displacement,
sub-interval displacements (onset / offset), and Lévy areas — because
every other movement feature was a scalar magnitude invariant to the
direction of motion.  Two signs with the same handshape that differ
only in direction (up vs down, clockwise vs counter-clockwise) now
become linearly separable.

Drop-in replacement for ml.phono_features but NEVER writes to prod paths.
"""
import os
import sys

import numpy as np

# Allow running standalone (without PYTHONPATH) AND from scripts under V8/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_V8_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _V8_ROOT not in sys.path:
    sys.path.insert(0, _V8_ROOT)

from ml.features import (  # noqa: E402
    N_POSE_LANDMARKS, HAND_FEATURES, HAND_WEIGHT, PALM_WEIGHT,
    to_right_dominant, hand_activity, both_hands_missing, interpolate_holes,
)

# ---------------------------------------------------------------------------
# Layout references for the 171-dim frame vector (same as ml.phono_features)
# ---------------------------------------------------------------------------
POSE_END = N_POSE_LANDMARKS * 3
LH_SHAPE = slice(POSE_END, POSE_END + 63)
LH_PALM = slice(POSE_END + 63, POSE_END + 66)
RH_SHAPE = slice(POSE_END + HAND_FEATURES, POSE_END + HAND_FEATURES + 63)
RH_PALM = slice(POSE_END + HAND_FEATURES + 63, POSE_END + HAND_FEATURES + 66)

POSE_NOSE = 0
POSE_L_SH, POSE_R_SH = 5, 6
POSE_L_WRIST, POSE_R_WRIST = 9, 10


# ---------------------------------------------------------------------------
# Small helpers (copied from phono_features)
# ---------------------------------------------------------------------------
def _pose_xyz(frame, idx):
    return frame[idx * 3: idx * 3 + 3]


def _wrist_track(seq, side='right'):
    idx = POSE_R_WRIST if side == 'right' else POSE_L_WRIST
    return seq[:, idx * 3: idx * 3 + 3]


def _hand_landmarks(seq, side='right'):
    sl = RH_SHAPE if side == 'right' else LH_SHAPE
    block = seq[:, sl] / HAND_WEIGHT
    return block.reshape(block.shape[0], 21, 3)


def _palm_vec(seq, side='right'):
    sl = RH_PALM if side == 'right' else LH_PALM
    return seq[:, sl] / PALM_WEIGHT


def _nonzero_frames(seq, side='right'):
    sl = RH_SHAPE if side == 'right' else LH_SHAPE
    return np.any(seq[:, sl] != 0, axis=1)


def _safe_path_length(track):
    diffs = np.diff(track, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _fft_peak(signal, n_fft=32):
    signal = signal - signal.mean()
    if len(signal) < 4 or np.all(signal == 0):
        return 0.0, 0.0
    f = np.fft.rfft(signal, n=n_fft)
    mag = np.abs(f)
    if len(mag) <= 1:
        return 0.0, 0.0
    peak = int(np.argmax(mag[1:]) + 1)
    return float(peak) / n_fft, float(mag[peak] / len(signal))


def _count_velocity_peaks(speed, rel_height=0.3):
    if len(speed) < 3:
        return 0
    top = speed.max()
    if top <= 1e-6:
        return 0
    thresh = rel_height * top
    peaks = 0
    for i in range(1, len(speed) - 1):
        if speed[i] > speed[i - 1] and speed[i] >= speed[i + 1] and speed[i] >= thresh:
            peaks += 1
    return peaks


# ---------------------------------------------------------------------------
# New helpers for v2 features
# ---------------------------------------------------------------------------
def _angle_between(v1, v2, eps=1e-8):
    """Angle in radians between two 3D vectors (or batches)."""
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    denom = np.maximum(n1 * n2, eps)
    cos = np.clip(np.sum(v1 * v2, axis=-1) / denom, -1.0, 1.0)
    return np.arccos(cos)


# MediaPipe hand landmark indices per finger:
#   thumb   1,2,3,4
#   index   5,6,7,8
#   middle  9,10,11,12
#   ring   13,14,15,16
#   pinky  17,18,19,20
# We'll measure the PIP joint curl, i.e. angle between (PIP->MCP) and (PIP->DIP).
_FINGER_JOINTS = [
    (1, 2, 3),     # thumb: CMC-MCP-IP  (no true PIP)
    (5, 6, 7),     # index: MCP-PIP-DIP
    (9, 10, 11),   # middle
    (13, 14, 15),  # ring
    (17, 18, 19),  # pinky
]


def _finger_curl_angles(hand_xyz):
    """
    For each of the 5 fingers, compute the angle at the middle joint.
    Returns (5,) in radians. A straight finger ≈ pi, a fully curled ≈ 0.5π.
    hand_xyz: (21, 3)
    """
    out = np.zeros(5, dtype='float32')
    for i, (a, b, c) in enumerate(_FINGER_JOINTS):
        v1 = hand_xyz[a] - hand_xyz[b]
        v2 = hand_xyz[c] - hand_xyz[b]
        out[i] = float(_angle_between(v1, v2))
    return out


def _convex_hull_area_2d(pts):
    """Convex-hull area of 2D points (Shoelace on monotone-chain hull)."""
    pts = np.asarray(pts, dtype='float64')
    if len(pts) < 3:
        return 0.0
    # Sort lexicographically
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _zero_crossings(x):
    """Number of times the sign of x flips (x is 1D). Ignores exact zeros."""
    if len(x) < 2:
        return 0
    s = np.sign(x)
    s = s[s != 0]
    if len(s) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(s)) > 0))


def _autocorr_peak(x):
    """
    Peak (non-zero-lag) of normalised autocorrelation in [0, 1].
    High values → periodic/oscillatory signal.
    """
    x = np.asarray(x, dtype='float64')
    if len(x) < 6:
        return 0.0
    x = x - x.mean()
    denom = (x * x).sum()
    if denom < 1e-12:
        return 0.0
    # Only consider lags 2..n/2 — lag 1 tends to be dominated by smoothness
    n = len(x)
    lags = range(2, max(3, n // 2))
    best = 0.0
    for k in lags:
        num = float((x[:-k] * x[k:]).sum())
        val = num / denom
        if val > best:
            best = val
    return float(best)


def _spectral_centroid(x, n_fft=32):
    """Normalised spectral centroid of a 1D signal (0..0.5). Invariant to DC."""
    x = np.asarray(x, dtype='float64')
    if len(x) < 4 or np.all(x == 0):
        return 0.0
    x = x - x.mean()
    f = np.fft.rfft(x, n=n_fft)
    mag = np.abs(f)
    if mag.sum() < 1e-12 or len(mag) <= 1:
        return 0.0
    freqs = np.arange(len(mag)) / n_fft
    return float((freqs * mag).sum() / (mag.sum() + 1e-12))


def _velocity_peak_timing(speed):
    """Position (0..1) of the global maximum of the speed profile."""
    if len(speed) == 0:
        return 0.0
    if speed.max() <= 1e-9:
        return 0.0
    return float(np.argmax(speed) / max(1, len(speed) - 1))


def _path_curvature_mean(track):
    """
    Mean curvature κ = |dθ/ds| of a 3D trajectory, projected to xy plane.
    Uses discrete turning angle per step normalised by arc length.
    """
    track = np.asarray(track, dtype='float64')
    if len(track) < 3:
        return 0.0
    d = np.diff(track[:, :2], axis=0)
    n = np.linalg.norm(d, axis=1)
    keep = n > 1e-6
    if keep.sum() < 2:
        return 0.0
    d = d[keep]
    n = n[keep]
    # angle of each step
    theta = np.arctan2(d[:, 1], d[:, 0])
    dtheta = np.diff(np.unwrap(theta))
    ds = 0.5 * (n[:-1] + n[1:])
    kappa = np.abs(dtheta) / np.maximum(ds, 1e-6)
    return float(kappa.mean())


def _angular_velocity_wrist(track):
    """
    Mean magnitude of change in step-direction (in radians/step). Rotation of
    the wrist's velocity vector, in the xy plane.
    """
    track = np.asarray(track, dtype='float64')
    if len(track) < 3:
        return 0.0
    d = np.diff(track[:, :2], axis=0)
    n = np.linalg.norm(d, axis=1)
    keep = n > 1e-6
    d = d[keep]
    if len(d) < 2:
        return 0.0
    theta = np.arctan2(d[:, 1], d[:, 0])
    return float(np.mean(np.abs(np.diff(np.unwrap(theta)))))


def _dwell_fraction(track, speed=None, rel_thresh=0.2):
    """Fraction of frames in which step-speed is below rel_thresh * max."""
    if speed is None:
        if len(track) < 2:
            return 0.0
        speed = np.linalg.norm(np.diff(track, axis=0), axis=1)
    if len(speed) == 0:
        return 0.0
    top = speed.max()
    if top <= 1e-9:
        return 1.0
    thr = rel_thresh * top
    return float(np.mean(speed < thr))


def _jerk_rms(track):
    """RMS of 3rd derivative of position (proxy for smoothness)."""
    t = np.asarray(track, dtype='float64')
    if len(t) < 4:
        return 0.0
    # Third discrete difference ~ finite-difference jerk (unit: per step^3).
    j = np.diff(t, n=3, axis=0)
    return float(np.sqrt(np.mean(np.linalg.norm(j, axis=1) ** 2)))


def _handshape_transition_count(openness, rel_thresh=0.15):
    """
    Count direction-change events in the openness signal that exceed a
    relative threshold. Reflects how many distinct handshape states a
    sign traverses.
    """
    if len(openness) < 3:
        return 0.0
    diff = np.diff(openness)
    if np.max(np.abs(diff)) < 1e-9:
        return 0.0
    scale = np.max(np.abs(diff))
    sig = np.where(np.abs(diff) >= rel_thresh * scale, np.sign(diff), 0)
    # Merge repeated signs and count transitions
    last = 0
    count = 0
    for s in sig:
        if s == 0:
            continue
        if s != last and last != 0:
            count += 1
        last = s
    return float(count)


def _hands_time_offset(lh_active, rh_active):
    """
    Time offset (in fraction of sequence) between the first active frames
    of left and right hands. >0 means RH started after LH.
    """
    n = max(1, len(lh_active))
    lh_idx = np.where(lh_active)[0]
    rh_idx = np.where(rh_active)[0]
    if len(lh_idx) == 0 or len(rh_idx) == 0:
        return 0.0
    return float((rh_idx[0] - lh_idx[0]) / n)


def _hands_parallelism(wl, wr, both):
    """
    Cosine similarity between per-frame velocity vectors of both hands
    (averaged). +1 = parallel, -1 = opposite (mirror), 0 = independent.
    """
    if both.sum() < 3:
        return 0.0
    wl_b = wl[both]
    wr_b = wr[both]
    dl = np.diff(wl_b, axis=0)
    dr = np.diff(wr_b, axis=0)
    nl = np.linalg.norm(dl, axis=1)
    nr = np.linalg.norm(dr, axis=1)
    mask = (nl > 1e-6) & (nr > 1e-6)
    if mask.sum() < 2:
        return 0.0
    cos = np.sum(dl[mask] * dr[mask], axis=1) / (nl[mask] * nr[mask])
    return float(np.mean(cos))


# ---------------------------------------------------------------------------
# Signed DIRECTIONAL features (v3 addition — 12 dims)
# ---------------------------------------------------------------------------
# Every other movement feature in v1/v2 is a scalar magnitude (path
# length, speed peaks, PCA eigenvalue angle, ...) and therefore
# invariant to the direction of motion.  Two signs with the same
# handshape that differ only in direction ("up vs down", "clockwise
# vs counter-clockwise") get mapped to almost identical descriptors.
# These 12 dims are explicitly SIGNED and encode:
#
#   * net displacement vector        (dx, dy, dz)             → 3
#   * onset-third displacement       (start → 1/3 of active)  → 3
#   * offset-third displacement      (2/3 → end of active)    → 3
#   * Lévy areas on xy, xz, yz       (chirality / rotation)   → 3
#
# All live on the right-dominant wrist (seq is already passed
# through `to_right_dominant`).  Inactive hands → zeros.


def _active_window(track, active):
    """Return the (T_act, 3) sub-track during the active frames.  If
    fewer than 2 active frames, returns `None`."""
    if active.sum() < 2:
        return None
    return track[active]


def _net_displacement(track):
    """Signed end-start vector (smoothed over 3 frames at each side)."""
    n = len(track)
    if n < 2:
        return np.zeros(3, dtype='float32')
    k = min(3, n)
    start = track[:k].mean(axis=0)
    end   = track[-k:].mean(axis=0)
    return (end - start).astype('float32')


def _interval_displacement(track, a_frac, b_frac):
    """Signed displacement vector between the mean of the [a_frac,
    a_frac+0.1] window and the mean of the [b_frac, b_frac+0.1]
    window (window size = 10 % of active length, min 2 frames)."""
    n = len(track)
    if n < 4:
        return np.zeros(3, dtype='float32')
    win = max(2, int(0.10 * n))
    a0 = max(0, int(a_frac * n))
    a1 = min(n, a0 + win)
    b0 = min(n - win, int(b_frac * n))
    b1 = min(n, b0 + win)
    start = track[a0:a1].mean(axis=0)
    end   = track[b0:b1].mean(axis=0)
    return (end - start).astype('float32')


def _levy_area(a, b):
    """Discrete Lévy area of a 2-D path (a, b) — ½ Σ (a_i·Δb_i −
    b_i·Δa_i).  Positive ⇒ counter-clockwise rotation of (a, b).
    Captures the chirality / sense of motion that all magnitude
    features throw away."""
    if len(a) < 2:
        return 0.0
    da = np.diff(a)
    db = np.diff(b)
    return 0.5 * float(np.sum(a[:-1] * db - b[:-1] * da))


def _levy_all(track):
    """Three signed Lévy areas — one per coordinate plane — that
    together encode 2-D rotation sense in frontal, axial and
    sagittal projections."""
    if len(track) < 3:
        return np.zeros(3, dtype='float32')
    xy = _levy_area(track[:, 0], track[:, 1])
    xz = _levy_area(track[:, 0], track[:, 2])
    yz = _levy_area(track[:, 1], track[:, 2])
    return np.array([xy, xz, yz], dtype='float32')


# ---------------------------------------------------------------------------
# Per-finger DIRECTION vectors (wrist → fingertip), v3 addition — 15 dims
# ---------------------------------------------------------------------------
# The curl angles only say HOW MUCH each finger is bent, not the
# ORIENTATION in which the finger points.  Two handshapes with
# identical curl angles but different pointing directions (e.g. index
# pointing UP vs FORWARD vs DOWN) are phonologically distinct in most
# sign languages.  These 15 dims are the normalised 3-D direction
# from the wrist (landmark 0) to each fingertip (landmarks 4, 8, 12,
# 16, 20) on the CENTRAL active frame.
#
# Fingertips × xyz = 5 × 3 = 15 dims.  Signed (unit vector components).

FINGERTIP_IDX = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky


def _finger_directions(hand_xyz):
    """Return (5, 3) unit vectors from wrist (idx 0) to each fingertip.
    Degenerate fingers (zero-length vector) emit zeros."""
    wrist = hand_xyz[0]
    out = np.zeros((5, 3), dtype='float32')
    for i, tip in enumerate(FINGERTIP_IDX):
        v = hand_xyz[tip] - wrist
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            out[i] = (v / n).astype('float32')
    return out


# ---------------------------------------------------------------------------
# Hand / palm rotation signature, v3d — 9 dims
# ---------------------------------------------------------------------------
# Problem this block solves:
#   A sign where the hand stays open and the wrist is stationary looks
#   identical to one where the hand stays open and the arm rotates
#   around the wrist — every wrist-translation feature (path length,
#   dir_net_*, wrist Lévy) is ≈ 0 in both cases, and palm_std_*
#   captures only the MAGNITUDE of rotation, not its sense.
#
# The 9 dims added below are:
#   palm_delta_x/y/z            end-frame palm-normal minus start-frame
#                               palm-normal.  Signed — positive vs
#                               negative rotation give opposite sign.
#   palm_levy_xy/xz/yz          Lévy area of the palm-normal trajectory
#                               on each coordinate plane.  Signed —
#                               clockwise vs counter-clockwise rotation
#                               give opposite sign.
#   mtip_levy_xy/xz/yz          Lévy area of the middle-fingertip
#                               trajectory (landmark 12 relative to
#                               wrist).  Captures hand rotation even
#                               when the palm normal barely moves
#                               (e.g. axial rotation of the forearm).
#
# All 9 values → 0 for a static hand and ≠ 0 for any rotation, with
# explicit signs that encode rotation sense.

MIDTIP_IDX = 12


def _signed_delta(a, b):
    return (b - a).astype('float32')


# ---------------------------------------------------------------------------
# Anchor-distance features (v3e) — 18 dims
# ---------------------------------------------------------------------------
# Track each wrist RELATIVE TO EACH SHOULDER over time.  V8 previously
# only had global shoulder-midpoint normalisation + averaged distances
# to fixed head/chest points — it could not distinguish a hand that
# stays on its own side from one that crosses the body to the opposite
# shoulder.
#
# Layout (18 dims):
#
#   Ipsilateral (hand-on-same-side) trajectories — (dx, dy) at 3 timepoints
#     RH → right-shoulder        start (dx, dy) + mid (dx, dy) + end (dx, dy) = 6
#     LH → left-shoulder         start (dx, dy) + mid (dx, dy) + end (dx, dy) = 6
#
#   Contralateral (crossing-body) — 3-D Euclidean distance at 2 timepoints
#     RH → left-shoulder         start_d3d + end_d3d                         = 2
#     LH → right-shoulder        start_d3d + end_d3d                         = 2
#
#   Radial approach/departure speed — signed peak rate of change of
#   ipsilateral distance (positive = hand moving AWAY from its shoulder)
#     RH ↔ right-shoulder                                                    = 1
#     LH ↔ left-shoulder                                                     = 1
#
# Per-hand: if the hand is absent for the whole sequence, all its dims
# emit zeros so the classifier receives a stable zero-block rather than
# random noise.


def _shoulder_xyz(seq, side):
    """Right shoulder = pose idx 6, left shoulder = pose idx 5 in the
    V8 slim layout (see features.py)."""
    idx = 6 if side == 'right' else 5
    return seq[:, idx * 3: idx * 3 + 3]


def _sample_3(values, active_idx):
    """Pick start, mid, end samples from `values` using indices in
    `active_idx` (avg of 3 frames each side for smoothness)."""
    if len(active_idx) == 0:
        return values[0] * 0.0, values[0] * 0.0, values[0] * 0.0
    k = min(3, len(active_idx))
    i_s = active_idx[:k]
    i_m = active_idx[len(active_idx) // 2: len(active_idx) // 2 + max(1, k // 2 + 1)]
    i_e = active_idx[-k:]
    return (values[i_s].mean(0),
            values[i_m].mean(0),
            values[i_e].mean(0))


def _signed_peak_radial_rate(dist_series):
    """Signed peak |Δd| in the active distance series.  Sign = sign of
    the net (end − start) drift, so positive = hand moved AWAY from
    its shoulder on average, negative = toward it."""
    if len(dist_series) < 2:
        return 0.0
    diffs = np.diff(dist_series)
    peak = float(np.max(np.abs(diffs)))
    net = float(dist_series[-1] - dist_series[0])
    sign = 1.0 if net >= 0 else -1.0
    return sign * peak


def _anchor_features(seq, wrist_track, hand_active, ipsi_side):
    """Produce 9 dims for one hand:
        ipsi_dx/dy at start, mid, end     (6)
        contra_d3d at start, end          (2)
        ipsi radial peak rate (signed)    (1)
    `ipsi_side` = 'right' or 'left' (the shoulder on the same side as
    this hand).
    """
    if hand_active.sum() < 2:
        return np.zeros(9, dtype='float32')
    act_idx = np.where(hand_active)[0]
    ipsi_sh   = _shoulder_xyz(seq, ipsi_side)
    contra_sh = _shoulder_xyz(seq, 'left' if ipsi_side == 'right' else 'right')

    # Ipsi: vector wrist - shoulder at start/mid/end, xy only
    v_ipsi = (wrist_track - ipsi_sh)[:, :2]
    s_ipsi, m_ipsi, e_ipsi = _sample_3(v_ipsi, act_idx)

    # Contra: 3-D distance
    d_contra = np.linalg.norm(wrist_track - contra_sh, axis=1)
    s_contra = float(d_contra[act_idx[:min(3, len(act_idx))]].mean())
    e_contra = float(d_contra[act_idx[-min(3, len(act_idx)):]].mean())

    # Signed radial peak rate on ipsi distance
    d_ipsi = np.linalg.norm(wrist_track - ipsi_sh, axis=1)[hand_active]
    radial_peak = _signed_peak_radial_rate(d_ipsi)

    return np.asarray([
        s_ipsi[0], s_ipsi[1],
        m_ipsi[0], m_ipsi[1],
        e_ipsi[0], e_ipsi[1],
        s_contra, e_contra,
        radial_peak,
    ], dtype='float32')


def _anchor_block(seq, rh_active, lh_active, wrist_r, wrist_l):
    """Full 18-dim anchor-distance block: right-hand (ipsi=R-shoulder)
    concat left-hand (ipsi=L-shoulder)."""
    r = _anchor_features(seq, wrist_r, rh_active, 'right')
    l = _anchor_features(seq, wrist_l, lh_active, 'left')
    return np.concatenate([r, l]).astype('float32')


def _palm_rotation_features(seq, rh_active, rh_palm, rh_landmarks):
    """Compute the 9 palm/hand rotation features for the right hand.
    Returns a flat 9-vector.  Zeros when the hand is absent or too
    few active frames are available."""
    out = np.zeros(9, dtype='float32')
    if rh_active.sum() < 3:
        return out
    act = np.where(rh_active)[0]
    k = min(3, len(act))

    # Palm-normal trajectory over active frames, re-normalised (palm
    # vectors are already unit, but averaging will dilute — keep raw).
    palm = rh_palm[act]

    # Palm delta: end mean minus start mean (signed, 3 dims)
    palm_s = palm[:k].mean(axis=0)
    palm_e = palm[-k:].mean(axis=0)
    out[0:3] = _signed_delta(palm_s, palm_e)

    # Palm Lévy areas on xy/xz/yz planes (signed, 3 dims)
    out[3] = _levy_area(palm[:, 0], palm[:, 1])
    out[4] = _levy_area(palm[:, 0], palm[:, 2])
    out[5] = _levy_area(palm[:, 1], palm[:, 2])

    # Middle-fingertip trajectory RELATIVE TO WRIST (so it's zero when
    # the whole hand translates rigidly but non-zero when the hand
    # rotates around the wrist — which is the exact case we want).
    mtip = rh_landmarks[act, MIDTIP_IDX] - rh_landmarks[act, 0]
    out[6] = _levy_area(mtip[:, 0], mtip[:, 1])
    out[7] = _levy_area(mtip[:, 0], mtip[:, 2])
    out[8] = _levy_area(mtip[:, 1], mtip[:, 2])
    return out


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
PHONO_V1_DIM = 34
PHONO_NEW_DIM = 26
PHONO_DIR_DIM      = 12     # v3a: signed wrist displacements + Lévy
PHONO_FDIR_DIM     = 15     # v3b: per-finger direction at CENTRAL frame
PHONO_FDIR_SE_DIM  = 30     # v3c: per-finger direction at START + END
PHONO_ROT_DIM      = 9      # v3d: palm + middle-tip rotation signature
PHONO_ANCHOR_DIM   = 18     # v3e: per-hand anchor distances (to each shoulder)
PHONO_DIM_V2 = (PHONO_V1_DIM + PHONO_NEW_DIM
                + PHONO_DIR_DIM + PHONO_FDIR_DIM
                + PHONO_FDIR_SE_DIM + PHONO_ROT_DIM
                + PHONO_ANCHOR_DIM)  # 144


def phonological_descriptor_v2(seq):
    """
    Compute a PHONO_DIM_V2-dim phonological descriptor from (T, 171+).

    The first 34 dims match `ml.phono_features.phonological_descriptor` so
    feature importance rankings stay comparable.
    """
    seq = np.asarray(seq, dtype='float32')
    if seq.ndim != 2 or seq.shape[1] < 171:
        return np.zeros(PHONO_DIM_V2, dtype='float32')
    seq = interpolate_holes(seq[:, :171])
    seq = to_right_dominant(seq)

    rh_active = _nonzero_frames(seq, 'right')
    lh_active = _nonzero_frames(seq, 'left')

    rh_landmarks = _hand_landmarks(seq, 'right')
    rh_palm = _palm_vec(seq, 'right')

    wrist_r = _wrist_track(seq, 'right')
    wrist_l = _wrist_track(seq, 'left')

    # ======================================================================
    #                          V1 (34 features) — unchanged
    # ======================================================================
    # Handshape (6)
    if rh_active.sum() == 0:
        handshape_feats = np.zeros(6, dtype='float32')
        openness_per_frame = np.zeros(seq.shape[0], dtype='float32')
    else:
        fingertips = rh_landmarks[:, [4, 8, 12, 16, 20], :]
        wrist_hand = rh_landmarks[:, 0:1, :]
        openness_per_frame = np.linalg.norm(fingertips - wrist_hand, axis=2).mean(axis=1)
        openness_active = openness_per_frame[rh_active]
        avg_open = float(openness_active.mean())
        range_open = float(openness_active.max() - openness_active.min())
        spread = np.linalg.norm(np.diff(fingertips, axis=1), axis=2).mean(axis=1)
        avg_spread = float(spread[rh_active].mean())
        thumb_idx_dist = np.linalg.norm(rh_landmarks[:, 4, :] - rh_landmarks[:, 8, :], axis=1)
        thumb_idx_mean = float(thumb_idx_dist[rh_active].mean())
        start_open = float(openness_per_frame[rh_active][:3].mean()) if rh_active.sum() >= 3 else avg_open
        end_open = float(openness_per_frame[rh_active][-3:].mean()) if rh_active.sum() >= 3 else avg_open
        handshape_feats = np.array([
            avg_open, range_open, avg_spread, thumb_idx_mean,
            start_open, end_open,
        ], dtype='float32')

    # Location (5)
    nose_y = -1.5
    chest_y = 0.3
    if rh_active.sum() == 0:
        location_feats = np.zeros(5, dtype='float32')
    else:
        w = wrist_r[rh_active]
        d_face = np.linalg.norm(w - np.array([0, nose_y, 0]), axis=1).min()
        d_chest = np.linalg.norm(w - np.array([0, chest_y, 0]), axis=1).min()
        d_forehead = np.linalg.norm(w - np.array([0, -2.0, 0]), axis=1).min()
        avg_y = float(w[:, 1].mean())
        avg_abs_x = float(np.abs(w[:, 0]).mean())
        location_feats = np.array([d_face, d_chest, d_forehead, avg_y, avg_abs_x],
                                  dtype='float32')

    # Movement (12)
    if rh_active.sum() < 2:
        movement_feats = np.zeros(12, dtype='float32')
    else:
        w = wrist_r[rh_active]
        path_len = _safe_path_length(w)
        displacement = float(np.linalg.norm(w[-1] - w[0]))
        straightness = displacement / (path_len + 1e-6)
        bbox_x = float(w[:, 0].max() - w[:, 0].min())
        bbox_y = float(w[:, 1].max() - w[:, 1].min())
        xy = w[:, :2] - w[:, :2].mean(axis=0)
        if len(xy) >= 2:
            cov = xy.T @ xy / len(xy)
            eigvals, eigvecs = np.linalg.eigh(cov)
            main_angle = float(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))
            eccent = float(eigvals[-1] / (eigvals.sum() + 1e-6))
        else:
            main_angle, eccent = 0.0, 0.0
        v = np.linalg.norm(np.diff(w, axis=0), axis=1)
        v_peak = float(v.max()) if len(v) else 0.0
        v_mean = float(v.mean()) if len(v) else 0.0
        n_peaks = float(_count_velocity_peaks(v))
        fft_x = _fft_peak(wrist_r[:, 0])
        fft_y = _fft_peak(wrist_r[:, 1])
        movement_feats = np.array([
            path_len, straightness, bbox_x, bbox_y,
            main_angle, eccent,
            v_peak, v_mean, n_peaks,
            fft_x[1], fft_y[0], fft_y[1],
        ], dtype='float32')

    # Orientation (6)
    if rh_active.sum() == 0:
        orient_feats = np.zeros(6, dtype='float32')
    else:
        p = rh_palm[rh_active]
        orient_feats = np.concatenate([
            p.mean(axis=0), p.std(axis=0),
        ]).astype('float32')

    # Non-manual (2)
    nose = seq[:, POSE_NOSE * 3: POSE_NOSE * 3 + 3]
    if len(nose) >= 4:
        head_osc_vert = float(np.abs(np.diff(nose[:, 1])).sum())
        head_osc_horiz = float(np.abs(np.diff(nose[:, 0])).sum())
    else:
        head_osc_vert, head_osc_horiz = 0.0, 0.0
    nonmanual_feats = np.array([head_osc_vert, head_osc_horiz], dtype='float32')

    # Bimanuality (3)
    lh_n = int(lh_active.sum())
    rh_n = int(rh_active.sum())
    is_bimanual = 1.0 if (lh_n >= 5 and rh_n >= 5) else 0.0
    both = lh_active & rh_active
    if both.sum() >= 2:
        dist = np.linalg.norm(wrist_l[both] - wrist_r[both], axis=1)
        hands_dist_mean = float(dist.mean())
        wl = wrist_l[both]
        wr = wrist_r[both]
        ml = np.linalg.norm(np.diff(wl, axis=0), axis=1)
        mr = np.linalg.norm(np.diff(wr, axis=0), axis=1)
        if len(ml) >= 2 and ml.std() > 1e-6 and mr.std() > 1e-6:
            symm = float(np.corrcoef(ml, mr)[0, 1])
        else:
            symm = 0.0
    else:
        hands_dist_mean = 0.0
        symm = 0.0
    bimanual_feats = np.array([is_bimanual, hands_dist_mean, symm], dtype='float32')

    # ======================================================================
    #                        NEW V2 FEATURES (26)
    # ======================================================================

    # --- Handshape advanced (8 dims) --------------------------------------
    # 5 per-finger curl angles + thumb-palm dist + thumb-index angle +
    # hand compactness (convex-hull / bbox)
    if rh_active.sum() == 0:
        curl_feats = np.zeros(5, dtype='float32')
        thumb_palm_dist = 0.0
        thumb_idx_angle = 0.0
        hand_compactness = 0.0
    else:
        # Use the central active frame for handshape geometry
        act_idx = np.where(rh_active)[0]
        mid = act_idx[len(act_idx) // 2]
        hand = rh_landmarks[mid]  # (21, 3)
        curl_feats = _finger_curl_angles(hand).astype('float32')
        # Palm centroid = mean of MCPs (landmarks 5, 9, 13, 17) + wrist
        palm_ctr = hand[[0, 5, 9, 13, 17]].mean(axis=0)
        thumb_palm_dist = float(np.linalg.norm(hand[4] - palm_ctr))
        v1 = hand[4] - hand[0]    # wrist -> thumb tip
        v2 = hand[8] - hand[0]    # wrist -> index tip
        thumb_idx_angle = float(_angle_between(v1, v2))
        # Hand compactness in xy
        pts = hand[:, :2]
        hull_a = _convex_hull_area_2d(pts)
        bbox_a = float((pts[:, 0].max() - pts[:, 0].min()) *
                       (pts[:, 1].max() - pts[:, 1].min()))
        hand_compactness = hull_a / (bbox_a + 1e-6) if bbox_a > 1e-6 else 0.0

    handshape_v2 = np.concatenate([
        curl_feats,
        np.array([thumb_palm_dist, thumb_idx_angle, hand_compactness],
                 dtype='float32'),
    ])

    # --- Movement enriched (7 dims) ---------------------------------------
    if rh_active.sum() < 3:
        movement_v2 = np.zeros(7, dtype='float32')
    else:
        w = wrist_r[rh_active]
        v_full = np.linalg.norm(np.diff(w, axis=0), axis=1)
        ang_vel = _angular_velocity_wrist(w)
        curvature = _path_curvature_mean(w)
        # Zero-crossings of vertical and horizontal velocity
        dy = np.diff(w[:, 1])
        dx = np.diff(w[:, 0])
        zc_y = float(_zero_crossings(dy))
        zc_x = float(_zero_crossings(dx))
        # Autocorrelation peak on speed (detects periodicity / repetitions)
        ac_peak = _autocorr_peak(v_full)
        # Spectral centroid of wrist y trajectory
        spec_c = _spectral_centroid(wrist_r[:, 1])
        # Velocity peak timing (where is the fastest moment)
        v_timing = _velocity_peak_timing(v_full)
        movement_v2 = np.array([
            ang_vel, curvature, zc_x, zc_y, ac_peak, spec_c, v_timing,
        ], dtype='float32')

    # --- Location enriched (6 dims) ---------------------------------------
    # start_x, start_y, end_x, end_y, dwell_fraction, trajectory_extent
    if rh_active.sum() < 2:
        location_v2 = np.zeros(6, dtype='float32')
    else:
        w = wrist_r[rh_active]
        n_take = min(3, len(w))
        start_x = float(w[:n_take, 0].mean())
        start_y = float(w[:n_take, 1].mean())
        end_x = float(w[-n_take:, 0].mean())
        end_y = float(w[-n_take:, 1].mean())
        # Dwell fraction on full wrist track (includes zeros = no hand = fast)
        v_full = np.linalg.norm(np.diff(w, axis=0), axis=1)
        dwell = _dwell_fraction(w, speed=v_full)
        # Trajectory extent: max distance between any two points (O(n^2) OK, n<=30)
        diffs = w[:, None, :] - w[None, :, :]
        extent = float(np.linalg.norm(diffs, axis=-1).max())
        location_v2 = np.array([start_x, start_y, end_x, end_y, dwell, extent],
                               dtype='float32')

    # --- Bimanual enriched (3 dims) ---------------------------------------
    if both.sum() >= 3:
        min_hands_dist = float(
            np.linalg.norm(wrist_l[both] - wrist_r[both], axis=1).min())
        parallelism = _hands_parallelism(wrist_l, wrist_r, both)
    else:
        min_hands_dist = 0.0
        parallelism = 0.0
    time_offset = _hands_time_offset(lh_active, rh_active) if (lh_n > 0 and rh_n > 0) else 0.0
    bimanual_v2 = np.array([min_hands_dist, time_offset, parallelism],
                           dtype='float32')

    # --- Elegance (2 dims) ------------------------------------------------
    if rh_active.sum() < 4:
        jerk = 0.0
    else:
        jerk = _jerk_rms(wrist_r[rh_active])
    if rh_active.sum() >= 3:
        transition_count = _handshape_transition_count(openness_per_frame[rh_active])
    else:
        transition_count = 0.0
    elegance_v2 = np.array([jerk, transition_count], dtype='float32')

    # --- Directional (signed) movement features — v3 (12 dims) -----------
    # These are the features that distinguish signs with the same
    # handshape but different motion direction.  Every element below is
    # SIGNED (not an absolute value / magnitude).
    track_act = _active_window(wrist_r, rh_active)
    if track_act is None:
        net_disp    = np.zeros(3, dtype='float32')
        onset_disp  = np.zeros(3, dtype='float32')
        offset_disp = np.zeros(3, dtype='float32')
        levy        = np.zeros(3, dtype='float32')
    else:
        net_disp    = _net_displacement(track_act)
        onset_disp  = _interval_displacement(track_act, 0.00, 0.33)
        offset_disp = _interval_displacement(track_act, 0.66, 1.00)
        levy        = _levy_all(track_act)
    directional_v3 = np.concatenate([net_disp, onset_disp,
                                     offset_disp, levy]).astype('float32')

    # --- Per-finger direction (wrist → fingertip) ----------------------
    # Three keyframes × 5 fingers × 3 coords = 45 dims total, split as:
    #   finger_dir       — CENTRAL active frame       (15 dims)
    #   finger_dir_start — avg of first 3 active fr.  (15 dims)
    #   finger_dir_end   — avg of last  3 active fr.  (15 dims)
    # Captures both the static handshape orientation AND any rotation
    # of the fingers during the sign (e.g. index starts pointing UP,
    # ends pointing FORWARD).  Each vector is unit-normalised.
    def _avg_finger_dir(frame_indices):
        mats = np.stack([_finger_directions(rh_landmarks[i])
                         for i in frame_indices], axis=0)
        avg = mats.mean(axis=0)
        # re-normalise to unit vectors after averaging
        n = np.linalg.norm(avg, axis=1, keepdims=True)
        n = np.where(n > 1e-6, n, 1.0)
        return (avg / n).reshape(-1).astype('float32')

    if rh_active.sum() == 0:
        finger_dir       = np.zeros(15, dtype='float32')
        finger_dir_start = np.zeros(15, dtype='float32')
        finger_dir_end   = np.zeros(15, dtype='float32')
    else:
        act_idx = np.where(rh_active)[0]
        mid = act_idx[len(act_idx) // 2]
        finger_dir = _finger_directions(rh_landmarks[mid]).reshape(-1)
        k = min(3, len(act_idx))
        finger_dir_start = _avg_finger_dir(act_idx[:k])
        finger_dir_end   = _avg_finger_dir(act_idx[-k:])

    # --- Hand/palm ROTATION signature (9 dims) --------------------------
    # Signed palm-normal delta + palm Lévy + middle-tip Lévy.  Zero for
    # a static hand, non-zero (with rotation-direction sign) for any
    # rotation around the wrist.
    rotation_v3d = _palm_rotation_features(seq, rh_active, rh_palm,
                                           rh_landmarks)

    # --- Anchor distances per hand per shoulder (18 dims, v3e) ---------
    anchor_v3e = _anchor_block(seq, rh_active, lh_active, wrist_r, wrist_l)

    descriptor = np.concatenate([
        handshape_feats,   # 6
        location_feats,    # 5
        movement_feats,    # 12
        orient_feats,      # 6
        nonmanual_feats,   # 2
        bimanual_feats,    # 3    -> total v1 = 34
        handshape_v2,      # 8
        movement_v2,       # 7
        location_v2,       # 6
        bimanual_v2,       # 3
        elegance_v2,       # 2    -> total new = 26
        directional_v3,    # 12   -> directional (v3a) = 12
        finger_dir,        # 15   -> finger direction central (v3b) = 15
        finger_dir_start,  # 15   -> finger direction start (v3c) = 15
        finger_dir_end,    # 15   -> finger direction end   (v3c) = 15
        rotation_v3d,      # 9    -> palm + midtip rotation (v3d) = 9
        anchor_v3e,        # 18   -> per-hand anchor distances (v3e) = 18
    ])
    assert descriptor.shape == (PHONO_DIM_V2,), descriptor.shape
    return descriptor.astype('float32')


PHONO_FEATURE_NAMES_V2 = [
    # --- v1 (34) ---
    'hs_openness_avg', 'hs_openness_range', 'hs_finger_spread', 'hs_thumb_idx',
    'hs_openness_start', 'hs_openness_end',
    'loc_d_face', 'loc_d_chest', 'loc_d_forehead', 'loc_avg_y', 'loc_avg_abs_x',
    'mv_path_len', 'mv_straightness', 'mv_bbox_x', 'mv_bbox_y',
    'mv_pca_angle', 'mv_pca_eccent',
    'mv_v_peak', 'mv_v_mean', 'mv_n_peaks',
    'mv_fft_x_mag', 'mv_fft_y_freq', 'mv_fft_y_mag',
    'palm_x', 'palm_y', 'palm_z', 'palm_std_x', 'palm_std_y', 'palm_std_z',
    'head_osc_vert', 'head_osc_horiz',
    'is_bimanual', 'hands_dist', 'hands_symm',
    # --- v2 additions (26) ---
    'hs2_thumb_curl', 'hs2_index_curl', 'hs2_middle_curl',
    'hs2_ring_curl', 'hs2_pinky_curl',
    'hs2_thumb_palm_dist', 'hs2_thumb_idx_angle', 'hs2_compactness',
    'mv2_angular_vel', 'mv2_curvature', 'mv2_zc_x', 'mv2_zc_y',
    'mv2_autocorr_peak', 'mv2_spec_centroid', 'mv2_v_peak_timing',
    'loc2_start_x', 'loc2_start_y', 'loc2_end_x', 'loc2_end_y',
    'loc2_dwell_frac', 'loc2_traj_extent',
    'bm2_min_hands_dist', 'bm2_time_offset', 'bm2_parallelism',
    'elg_jerk_rms', 'elg_hs_transitions',
    # --- v3a directional (12) — SIGNED direction-of-motion features ---
    'dir_net_dx',   'dir_net_dy',   'dir_net_dz',
    'dir_onset_dx', 'dir_onset_dy', 'dir_onset_dz',
    'dir_off_dx',   'dir_off_dy',   'dir_off_dz',
    'dir_levy_xy', 'dir_levy_xz', 'dir_levy_yz',
    # --- v3b per-finger direction CENTRAL (15) — wrist→tip unit vectors
    'fdir_thumb_dx',  'fdir_thumb_dy',  'fdir_thumb_dz',
    'fdir_index_dx',  'fdir_index_dy',  'fdir_index_dz',
    'fdir_middle_dx', 'fdir_middle_dy', 'fdir_middle_dz',
    'fdir_ring_dx',   'fdir_ring_dy',   'fdir_ring_dz',
    'fdir_pinky_dx',  'fdir_pinky_dy',  'fdir_pinky_dz',
    # --- v3c per-finger direction START (15) ---------------------------
    'fdirs_thumb_dx',  'fdirs_thumb_dy',  'fdirs_thumb_dz',
    'fdirs_index_dx',  'fdirs_index_dy',  'fdirs_index_dz',
    'fdirs_middle_dx', 'fdirs_middle_dy', 'fdirs_middle_dz',
    'fdirs_ring_dx',   'fdirs_ring_dy',   'fdirs_ring_dz',
    'fdirs_pinky_dx',  'fdirs_pinky_dy',  'fdirs_pinky_dz',
    # --- v3c per-finger direction END (15) -----------------------------
    'fdire_thumb_dx',  'fdire_thumb_dy',  'fdire_thumb_dz',
    'fdire_index_dx',  'fdire_index_dy',  'fdire_index_dz',
    'fdire_middle_dx', 'fdire_middle_dy', 'fdire_middle_dz',
    'fdire_ring_dx',   'fdire_ring_dy',   'fdire_ring_dz',
    'fdire_pinky_dx',  'fdire_pinky_dy',  'fdire_pinky_dz',
    # --- v3d rotation signature (9) — palm delta + Lévy + midtip Lévy -
    'rot_palm_dx',  'rot_palm_dy',  'rot_palm_dz',
    'rot_palm_levy_xy', 'rot_palm_levy_xz', 'rot_palm_levy_yz',
    'rot_mtip_levy_xy', 'rot_mtip_levy_xz', 'rot_mtip_levy_yz',
    # --- v3e anchor distances per hand per shoulder (18) --------------
    # Right hand: ipsilateral (vs right-shoulder) xy at 3 timepoints,
    # contralateral (vs left-shoulder) 3-D distance at start/end,
    # signed peak radial rate ipsi.
    'anc_rh_R_dx_s', 'anc_rh_R_dy_s',
    'anc_rh_R_dx_m', 'anc_rh_R_dy_m',
    'anc_rh_R_dx_e', 'anc_rh_R_dy_e',
    'anc_rh_L_d3d_s', 'anc_rh_L_d3d_e',
    'anc_rh_R_radial_peak',
    # Left hand: ipsilateral = left-shoulder, contralateral = right.
    'anc_lh_L_dx_s', 'anc_lh_L_dy_s',
    'anc_lh_L_dx_m', 'anc_lh_L_dy_m',
    'anc_lh_L_dx_e', 'anc_lh_L_dy_e',
    'anc_lh_R_d3d_s', 'anc_lh_R_d3d_e',
    'anc_lh_L_radial_peak',
]
assert len(PHONO_FEATURE_NAMES_V2) == PHONO_DIM_V2, \
    f'{len(PHONO_FEATURE_NAMES_V2)} vs {PHONO_DIM_V2}'
