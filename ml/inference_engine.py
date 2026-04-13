"""Inference engine wrapping DTW + FAISS + KNN."""
import numpy as np
import os
import json
import faiss
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from ml.features import FEATURE_DIM, add_wrist_velocity
from ml.constants import (
    INDEX_PATH, METADATA_PATH, TEMPLATE_DIR,
    SEQUENCE_LENGTH, PREFILTER_TOP, K, THRESHOLD,
    SMOOTH_ALPHA, MOTION_START_THRESHOLD, MOTION_END_THRESHOLD,
    MOTION_END_FRAMES, MIN_SIGN_FRAMES, MAX_SIGN_FRAMES,
    PREDICT_AFTER_N_FRAMES, PREDICT_EVERY_N_FRAMES, WEIGHT_POWER, DATA_DIR,
)

# Precompute temporal weights: start=0.1, middle=0.8, end=1.0 (piecewise linear)
FRAME_WEIGHTS = np.interp(
    np.linspace(0, 1, SEQUENCE_LENGTH),
    [0.0, 0.5, 1.0],
    [0.1, 0.8, 1.0],
).astype('float32')[:, None]


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
            for m in self.metadata:
                path = m['path']
                if not os.path.isabs(path):
                    path = os.path.join(DATA_DIR, path)
                if not os.path.exists(path):
                    continue  # skip missing files
                tpl = np.load(path)
                self.templates.append(add_wrist_velocity(tpl))
                valid_metadata.append(m)

            self.metadata = valid_metadata

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
        query = add_wrist_velocity(query)
        query_summary = compute_summary(query).reshape(1, -1)

        # FAISS pre-filter
        n_search = min(PREFILTER_TOP, self.index.ntotal)
        _, indices = self.index.search(query_summary, n_search)

        # Weighted DTW
        weighted_query = query * np.sqrt(FRAME_WEIGHTS)
        dtw_results = []
        for idx in indices[0]:
            if idx < 0:
                continue
            weighted_template = self.templates[idx] * np.sqrt(FRAME_WEIGHTS)
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
        self.peak_energy = 0.0  # track the max energy during this sign
        self.energy_history = []  # recent energy values for deceleration detection

    def process_features(self, raw_features):
        # Smooth
        if self.smoothed_features is None:
            self.smoothed_features = raw_features.astype('float32')
        else:
            self.smoothed_features = (
                SMOOTH_ALPHA * raw_features +
                (1 - SMOOTH_ALPHA) * self.smoothed_features
            ).astype('float32')

        # Motion energy
        if self.prev_features is not None:
            self.motion_energy = float(np.mean(np.abs(
                self.smoothed_features - self.prev_features
            )))
        else:
            self.motion_energy = 0.0
        self.prev_features = self.smoothed_features.copy()

        sign_ended = False
        ended_buffer = None

        if not self.is_signing:
            if self.motion_energy > MOTION_START_THRESHOLD:
                self.is_signing = True
                self.sign_buffer = [self.smoothed_features.copy()]
                self.low_motion_count = 0
                self.peak_energy = self.motion_energy
                self.energy_history = [self.motion_energy]
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

            # Method 2: deceleration detection (energy dropped to < 30% of peak)
            # Only after we've had at least MIN_SIGN_FRAMES and a significant peak
            end_by_deceleration = (
                len(self.sign_buffer) >= MIN_SIGN_FRAMES
                and self.peak_energy > MOTION_START_THRESHOLD * 2
                and self.motion_energy < self.peak_energy * 0.3
            )

            end_by_max = len(self.sign_buffer) >= MAX_SIGN_FRAMES

            if end_by_idle or end_by_deceleration or end_by_max:
                if len(self.sign_buffer) >= MIN_SIGN_FRAMES:
                    # Trim trailing idle frames (keep only active movement)
                    trim_idx = len(self.sign_buffer)
                    for j in range(len(self.energy_history) - 1, -1, -1):
                        if self.energy_history[j] > MOTION_END_THRESHOLD:
                            trim_idx = j + 1
                            break
                    trimmed = self.sign_buffer[:trim_idx]

                    if len(trimmed) >= MIN_SIGN_FRAMES:
                        sign_ended = True
                        ended_buffer = trimmed

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
