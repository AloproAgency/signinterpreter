"""Inference engine wrapping DTW + FAISS + KNN."""
import numpy as np
import os
import json
import faiss
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from ml.features import (
    FEATURE_DIM, add_wrist_velocity, to_right_dominant, both_hands_missing,
    interpolate_holes,
)

# Templates with too many frames missing both hands are filtered out at load time.
MAX_BOTH_HANDS_MISSING = 3
from ml.constants import (
    INDEX_PATH, METADATA_PATH, TEMPLATE_DIR,
    SEQUENCE_LENGTH, PREFILTER_TOP, K, THRESHOLD,
    SMOOTH_ALPHA, MOTION_START_THRESHOLD, MOTION_END_THRESHOLD,
    MOTION_END_FRAMES, MIN_SIGN_FRAMES, MAX_SIGN_FRAMES,
    PREDICT_AFTER_N_FRAMES, PREDICT_EVERY_N_FRAMES, WEIGHT_POWER, DATA_DIR,
)

# Uniform temporal weights (no frame is privileged)
FRAME_WEIGHTS = np.ones((SEQUENCE_LENGTH, 1), dtype='float32')

# Per-channel weights — boost finger shape so it dominates the DTW distance.
# Layout (FEATURE_DIM = 177): pose[0:39], LH shape[39:102], LH palm[102:105],
# RH shape[105:168], RH palm[168:171], wrist velocity[171:177].
HAND_SHAPE_BOOST = 3.0
CHANNEL_WEIGHTS = np.ones(FEATURE_DIM, dtype='float32')
CHANNEL_WEIGHTS[39:102] = HAND_SHAPE_BOOST   # left hand shape
CHANNEL_WEIGHTS[105:168] = HAND_SHAPE_BOOST  # right hand shape

# Drop the Z coordinate of finger landmarks (depth from single camera is unreliable).
# Hand shape = 21 landmarks × (x, y, z) flattened. Z indices = 2, 5, 8, ..., 62 within each block.
for base in (39, 105):
    for k in range(21):
        CHANNEL_WEIGHTS[base + k * 3 + 2] = 0.0

CHANNEL_SQRT = np.sqrt(CHANNEL_WEIGHTS)


def compute_summary(template):
    return np.concatenate([
        template.mean(axis=0),
        template.std(axis=0),
        template[0],
        template[-1],
    ]).astype('float32')


def resample_sequence(sequence, target_length):
    seq = np.asarray(sequence, dtype='float32')
    n = len(seq)
    if n == target_length:
        return seq
    if n < 2:
        return np.tile(seq[:1], (target_length, 1))
    old_idx = np.linspace(0, n - 1, n)
    new_idx = np.linspace(0, n - 1, target_length)
    out = np.zeros((target_length, seq.shape[1]), dtype='float32')
    for f in range(seq.shape[1]):
        out[:, f] = np.interp(new_idx, old_idx, seq[:, f])
    return out


class InferenceEngine:
    def __init__(self):
        self.index = None
        self.metadata = None
        self.templates = None
        self.words = []
        self.loaded = False

    def load(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
            print("WARNING: FAISS index not found. Build it first.")
            self.loaded = False
            return

        try:
            self.index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, 'r') as f:
                self.metadata = json.load(f)

            if not self.metadata:
                print("WARNING: No templates in metadata.")
                self.loaded = False
                return

            self.templates = []
            valid_metadata = []
            n_filtered_quality = 0
            n_mirrored = 0
            for m in self.metadata:
                path = m['path']
                if not os.path.isabs(path):
                    path = os.path.join(DATA_DIR, path)
                if not os.path.exists(path):
                    continue  # skip missing files
                tpl = np.load(path)
                # A) quality filter — drop templates with too many frames where both hands are missing
                if both_hands_missing(tpl) > MAX_BOTH_HANDS_MISSING:
                    n_filtered_quality += 1
                    continue
                # B) fill mid-sequence detection holes via interpolation
                tpl = interpolate_holes(tpl[:, :FEATURE_DIM] if tpl.shape[1] > 171 else tpl)
                # C) canonicalize laterality — always put the dominant hand on the right
                tpl_canon = to_right_dominant(tpl)
                if not np.array_equal(tpl_canon, tpl):
                    n_mirrored += 1
                self.templates.append(add_wrist_velocity(tpl_canon))
                valid_metadata.append(m)

            self.metadata = valid_metadata
            print(f'  filtered (low quality): {n_filtered_quality}, mirrored to right-dominant: {n_mirrored}')

            if not self.templates:
                print("WARNING: No valid templates found.")
                self.loaded = False
                return

            self.words = sorted(set(m['word'] for m in self.metadata))
            self.loaded = True
            print(f"InferenceEngine loaded: {len(self.words)} words, {len(self.templates)} templates")
        except Exception as e:
            print(f"ERROR loading engine: {e}")
            self.loaded = False

    def reload(self):
        self.load()

    def predict(self, query_sequence):
        """
        Predict sign from a sequence of features.
        query_sequence: list of feature vectors (variable length)
        Returns: (best_word, best_distance, top_k_list)
        """
        if not self.loaded:
            return None, None, []

        # Resample to SEQUENCE_LENGTH then add wrist velocity
        query = resample_sequence(query_sequence, SEQUENCE_LENGTH)
        query = interpolate_holes(query)
        query = to_right_dominant(query)
        query = add_wrist_velocity(query)
        query_summary = compute_summary(query).reshape(1, -1)

        # FAISS pre-filter
        n_search = min(PREFILTER_TOP, self.index.ntotal)
        _, indices = self.index.search(query_summary, n_search)

        # Weighted DTW (frame × channel weighting)
        frame_sqrt = np.sqrt(FRAME_WEIGHTS)
        weighted_query = query * frame_sqrt * CHANNEL_SQRT
        dtw_results = []
        for idx in indices[0]:
            if idx < 0:
                continue
            weighted_template = self.templates[idx] * frame_sqrt * CHANNEL_SQRT
            d, _ = fastdtw(weighted_query, weighted_template, dist=euclidean)
            dtw_results.append((d, self.metadata[idx]['word']))

        if not dtw_results:
            return None, None, []

        dtw_results.sort(key=lambda x: x[0])
        top_k = dtw_results[:K]

        votes = {}
        for d, w in top_k:
            votes[w] = votes.get(w, 0) + 1
        best_word = max(votes, key=votes.get)
        best_distance = min(d for d, w in top_k if w == best_word)
        return best_word, best_distance, top_k


class SignSegmenter:
    """State machine for detecting sign start/end from motion energy."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.is_signing = False
        self.sign_buffer = []
        self.low_motion_count = 0
        self.smoothed_features = None
        self.prev_features = None
        self.motion_energy = 0.0
        self.peak_energy = 0.0
        self.energy_history = []
        self.start_streak = 0
        self._just_recovered = False  # skip motion calc on first frame after a blind period

    def notify_hand_lost(self):
        """Called by the WS layer when MediaPipe drops the hand. Next good
        frame must be treated as a continuation point, not a new delta."""
        self._just_recovered = True

    def process_features(self, raw_features):
        # Smooth
        if self.smoothed_features is None:
            self.smoothed_features = raw_features.astype('float32')
        else:
            self.smoothed_features = (
                SMOOTH_ALPHA * raw_features +
                (1 - SMOOTH_ALPHA) * self.smoothed_features
            ).astype('float32')

        # Motion energy — restrict to HAND dims (39:171). Skip the computation
        # for one frame after a blind period to avoid a spurious spike from
        # the hand having moved during the missing frames.
        if self._just_recovered:
            self.motion_energy = 0.0
            self._just_recovered = False
        elif self.prev_features is not None:
            hand_delta = np.abs(
                self.smoothed_features[39:171] - self.prev_features[39:171]
            )
            self.motion_energy = float(hand_delta.mean())
        else:
            self.motion_energy = 0.0
        self.prev_features = self.smoothed_features.copy()

        sign_ended = False
        ended_buffer = None

        if not self.is_signing:
            # Require MOTION_START_THRESHOLD for 2 consecutive frames before arming.
            # Catches slow/subtle signs (lower threshold) while rejecting single-frame spikes.
            if self.motion_energy > MOTION_START_THRESHOLD:
                self.start_streak += 1
            else:
                self.start_streak = 0

            if self.start_streak >= 2:
                self.is_signing = True
                self.sign_buffer = [self.smoothed_features.copy()]
                self.low_motion_count = 0
                self.peak_energy = self.motion_energy
                self.energy_history = [self.motion_energy]
                self.start_streak = 0
        else:
            self.sign_buffer.append(self.smoothed_features.copy())
            self.energy_history.append(self.motion_energy)

            # Track peak energy
            if self.motion_energy > self.peak_energy:
                self.peak_energy = self.motion_energy

            # Method 1: classic idle detection (energy stays low)
            if self.motion_energy < MOTION_END_THRESHOLD:
                self.low_motion_count += 1
            else:
                self.low_motion_count = 0
            end_by_idle = self.low_motion_count >= MOTION_END_FRAMES

            # Method 2: deceleration to < 30% of peak (global)
            end_by_deceleration = (
                len(self.sign_buffer) >= MIN_SIGN_FRAMES
                and self.peak_energy > MOTION_START_THRESHOLD * 2
                and self.motion_energy < self.peak_energy * 0.3
            )

            # Method 3: motion VALLEY — current frame is a local minimum vs
            # the recent peak (last 6 frames). Catches transitions between two
            # consecutive signs where motion dips but never reaches the idle
            # threshold. Requires a substantial dip (≤ 50 % of recent peak).
            end_by_valley = False
            if len(self.energy_history) >= 6 and len(self.sign_buffer) >= MIN_SIGN_FRAMES:
                recent_peak = max(self.energy_history[-6:])
                if (recent_peak > MOTION_START_THRESHOLD * 2
                        and self.motion_energy <= recent_peak * 0.5
                        and self.motion_energy < self.energy_history[-2]
                        and self.energy_history[-2] < self.energy_history[-3]):
                    # Falling edge into a valley → cut
                    end_by_valley = True

            end_by_max = len(self.sign_buffer) >= MAX_SIGN_FRAMES

            if end_by_idle or end_by_deceleration or end_by_valley or end_by_max:
                trimmed = list(self.sign_buffer)
                if end_by_idle:
                    # Trim trailing idle frames
                    trim_idx = len(trimmed)
                    for j in range(len(self.energy_history) - 1, -1, -1):
                        if self.energy_history[j] > MOTION_END_THRESHOLD:
                            trim_idx = j + 1
                            break
                    trimmed = trimmed[:trim_idx]

                if len(trimmed) >= MIN_SIGN_FRAMES:
                    sign_ended = True
                    ended_buffer = trimmed

                # If we cut on a valley, immediately arm a new sign with the
                # current frame so we don't miss the start of the next one.
                if end_by_valley and self.motion_energy > MOTION_START_THRESHOLD:
                    self.sign_buffer = [self.smoothed_features.copy()]
                    self.low_motion_count = 0
                    self.peak_energy = self.motion_energy
                    self.energy_history = [self.motion_energy]
                    # stay is_signing = True
                else:
                    self.is_signing = False
                    self.sign_buffer = []
                    self.low_motion_count = 0
                    self.peak_energy = 0.0
                    self.energy_history = []

        # Intermediate prediction
        should_predict = (
            self.is_signing and
            len(self.sign_buffer) >= PREDICT_AFTER_N_FRAMES and
            len(self.sign_buffer) % PREDICT_EVERY_N_FRAMES == 0
        )

        return {
            'motion_energy': self.motion_energy,
            'is_signing': self.is_signing,
            'buffer_length': len(self.sign_buffer),
            'sign_ended': sign_ended,
            'ended_buffer': ended_buffer,
            'should_predict_intermediate': should_predict,
            'current_buffer': list(self.sign_buffer) if should_predict else None,
        }
