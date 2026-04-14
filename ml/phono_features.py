"""
Phonological sign descriptor.

Transforms a (T, 171) feature sequence into a fixed-size vector that encodes
the four phonological parameters of a sign (handshape, location, movement,
orientation) plus non-manual markers and bimanuality.

Input is assumed to be the raw MediaPipe-derived features produced by
ml.features.extract_features_from_results (171 dims, no velocity yet). The
sequence should ideally be resampled to SEQUENCE_LENGTH first, but the
extractor is agnostic to length.
"""
import numpy as np

from ml.features import (
    N_POSE_LANDMARKS, HAND_FEATURES, HAND_WEIGHT, PALM_WEIGHT,
    to_right_dominant, hand_activity, both_hands_missing, interpolate_holes,
)

# ---------------------------------------------------------------------------
# Layout references for the 171-dim frame vector (same order used everywhere)
# ---------------------------------------------------------------------------
POSE_END = N_POSE_LANDMARKS * 3                       # 39
LH_SHAPE = slice(POSE_END, POSE_END + 63)             # 39:102   (21 × 3 × HAND_WEIGHT)
LH_PALM = slice(POSE_END + 63, POSE_END + 66)         # 102:105  (3 × PALM_WEIGHT)
RH_SHAPE = slice(POSE_END + HAND_FEATURES,
                 POSE_END + HAND_FEATURES + 63)       # 105:168
RH_PALM = slice(POSE_END + HAND_FEATURES + 63,
                POSE_END + HAND_FEATURES + 66)        # 168:171

# Pose landmark indices (within UPPER_BODY_LANDMARKS):
# 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear,
# 5 l_sh, 6 r_sh, 7 l_el, 8 r_el, 9 l_wrist, 10 r_wrist, 11 l_pinky, 12 r_pinky
POSE_NOSE = 0
POSE_L_SH, POSE_R_SH = 5, 6
POSE_L_WRIST, POSE_R_WRIST = 9, 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pose_xyz(frame, idx):
    """Return 3D position of pose landmark `idx` (already body-normalized)."""
    return frame[idx * 3: idx * 3 + 3]


def _wrist_track(seq, side='right'):
    """Return (T, 3) wrist trajectory in body frame."""
    idx = POSE_R_WRIST if side == 'right' else POSE_L_WRIST
    return seq[:, idx * 3: idx * 3 + 3]


def _hand_landmarks(seq, side='right'):
    """Return (T, 21, 3) hand-shape landmarks, undoing HAND_WEIGHT."""
    sl = RH_SHAPE if side == 'right' else LH_SHAPE
    block = seq[:, sl] / HAND_WEIGHT
    return block.reshape(block.shape[0], 21, 3)


def _palm_vec(seq, side='right'):
    """Return (T, 3) palm normal, undoing PALM_WEIGHT."""
    sl = RH_PALM if side == 'right' else LH_PALM
    return seq[:, sl] / PALM_WEIGHT


def _nonzero_frames(seq, side='right'):
    sl = RH_SHAPE if side == 'right' else LH_SHAPE
    return np.any(seq[:, sl] != 0, axis=1)


def _safe_path_length(track):
    """Sum of step-to-step distances."""
    diffs = np.diff(track, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _fft_peak(signal, n_fft=32):
    """Return (peak_freq_bin, peak_magnitude) for the 1-D signal."""
    signal = signal - signal.mean()
    if len(signal) < 4 or np.all(signal == 0):
        return 0.0, 0.0
    f = np.fft.rfft(signal, n=n_fft)
    mag = np.abs(f)
    # Ignore DC
    if len(mag) <= 1:
        return 0.0, 0.0
    peak = int(np.argmax(mag[1:]) + 1)
    return float(peak) / n_fft, float(mag[peak] / len(signal))


def _count_velocity_peaks(speed, rel_height=0.3):
    """Count local maxima in the speed profile above `rel_height * max`."""
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
# Main extractor
# ---------------------------------------------------------------------------
PHONO_DIM = 34


def phonological_descriptor(seq):
    """
    Compute a 34-dim phonological descriptor from a (T, 171) feature sequence.

    The sequence is first canonicalised so the dominant hand is on the right.
    If both hands are inactive (no detections) the descriptor is all zeros.
    """
    seq = np.asarray(seq, dtype='float32')
    if seq.ndim != 2 or seq.shape[1] < 171:
        return np.zeros(PHONO_DIM, dtype='float32')
    seq = interpolate_holes(seq[:, :171])
    seq = to_right_dominant(seq)

    # --- Extract per-frame building blocks ----------------------------------
    rh_active = _nonzero_frames(seq, 'right')
    lh_active = _nonzero_frames(seq, 'left')
    any_hand = rh_active | lh_active

    # Dominant hand = right (after canonicalization)
    rh_landmarks = _hand_landmarks(seq, 'right')       # (T, 21, 3)
    rh_palm = _palm_vec(seq, 'right')                  # (T, 3)

    wrist_r = _wrist_track(seq, 'right')               # (T, 3) body-frame
    wrist_l = _wrist_track(seq, 'left')

    # --- HANDSHAPE (6 dims) -------------------------------------------------
    if rh_active.sum() == 0:
        handshape_feats = np.zeros(6, dtype='float32')
    else:
        fingertips = rh_landmarks[:, [4, 8, 12, 16, 20], :]      # (T, 5, 3)
        wrist_hand = rh_landmarks[:, 0:1, :]                     # (T, 1, 3)
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

    # --- LOCATION (5 dims) --------------------------------------------------
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

    # --- MOVEMENT (12 dims) -------------------------------------------------
    if rh_active.sum() < 2:
        movement_feats = np.zeros(12, dtype='float32')
    else:
        w = wrist_r[rh_active]     # (n_active, 3)
        path_len = _safe_path_length(w)
        displacement = float(np.linalg.norm(w[-1] - w[0]))
        straightness = displacement / (path_len + 1e-6)
        bbox_x = float(w[:, 0].max() - w[:, 0].min())
        bbox_y = float(w[:, 1].max() - w[:, 1].min())
        # PCA on trajectory (2D using xy)
        xy = w[:, :2] - w[:, :2].mean(axis=0)
        if len(xy) >= 2:
            cov = xy.T @ xy / len(xy)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # eigvals ascending
            main_angle = float(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))
            eccent = float(eigvals[-1] / (eigvals.sum() + 1e-6))
        else:
            main_angle, eccent = 0.0, 0.0
        # Velocity profile (on full active trajectory in 3D)
        v = np.linalg.norm(np.diff(w, axis=0), axis=1)
        v_peak = float(v.max()) if len(v) else 0.0
        v_mean = float(v.mean()) if len(v) else 0.0
        n_peaks = float(_count_velocity_peaks(v))
        # FFT on raw (unmasked) x and y of right wrist
        fft_x = _fft_peak(wrist_r[:, 0])
        fft_y = _fft_peak(wrist_r[:, 1])
        movement_feats = np.array([
            path_len, straightness, bbox_x, bbox_y,
            main_angle, eccent,
            v_peak, v_mean, n_peaks,
            fft_x[1], fft_y[0], fft_y[1],
        ], dtype='float32')

    # --- ORIENTATION (6 dims) -----------------------------------------------
    if rh_active.sum() == 0:
        orient_feats = np.zeros(6, dtype='float32')
    else:
        p = rh_palm[rh_active]
        orient_feats = np.concatenate([
            p.mean(axis=0), p.std(axis=0),
        ]).astype('float32')

    # --- NON-MANUAL (2 dims) ------------------------------------------------
    nose = seq[:, POSE_NOSE * 3: POSE_NOSE * 3 + 3]
    if len(nose) >= 4:
        nose_x = nose[:, 0]
        nose_y_arr = nose[:, 1]
        head_osc_vert = float(np.abs(np.diff(nose_y_arr)).sum())
        head_osc_horiz = float(np.abs(np.diff(nose_x)).sum())
    else:
        head_osc_vert, head_osc_horiz = 0.0, 0.0
    nonmanual_feats = np.array([head_osc_vert, head_osc_horiz], dtype='float32')

    # --- BIMANUALITY (3 dims) -----------------------------------------------
    lh_n = int(lh_active.sum())
    rh_n = int(rh_active.sum())
    is_bimanual = 1.0 if (lh_n >= 5 and rh_n >= 5) else 0.0
    both = lh_active & rh_active
    if both.sum() >= 2:
        dist = np.linalg.norm(wrist_l[both] - wrist_r[both], axis=1)
        hands_dist_mean = float(dist.mean())
        # Symmetry: correlation of the two trajectories' motion magnitudes
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

    descriptor = np.concatenate([
        handshape_feats,   # 6
        location_feats,    # 5
        movement_feats,    # 12
        orient_feats,      # 6
        nonmanual_feats,   # 2
        bimanual_feats,    # 3
    ])
    assert descriptor.shape == (PHONO_DIM,), descriptor.shape
    return descriptor.astype('float32')


# Handy names for debugging
PHONO_FEATURE_NAMES = [
    # Handshape (6)
    'hs_openness_avg', 'hs_openness_range', 'hs_finger_spread', 'hs_thumb_idx',
    'hs_openness_start', 'hs_openness_end',
    # Location (5)
    'loc_d_face', 'loc_d_chest', 'loc_d_forehead', 'loc_avg_y', 'loc_avg_abs_x',
    # Movement (12)
    'mv_path_len', 'mv_straightness', 'mv_bbox_x', 'mv_bbox_y',
    'mv_pca_angle', 'mv_pca_eccent',
    'mv_v_peak', 'mv_v_mean', 'mv_n_peaks',
    'mv_fft_x_mag', 'mv_fft_y_freq', 'mv_fft_y_mag',
    # Orientation (6)
    'palm_x', 'palm_y', 'palm_z', 'palm_std_x', 'palm_std_y', 'palm_std_z',
    # Non-manual (2)
    'head_osc_vert', 'head_osc_horiz',
    # Bimanuality (3)
    'is_bimanual', 'hands_dist', 'hands_symm',
]
assert len(PHONO_FEATURE_NAMES) == PHONO_DIM, f'{len(PHONO_FEATURE_NAMES)} vs {PHONO_DIM}'
