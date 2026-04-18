"""Sign segmentation state-machine + utility for temporal resampling.

Previously lived in ml/inference_engine.py alongside the now-removed KNN+DTW
fallback. Extracted here so the production path (SignSegmenter + resample)
has no KNN/FAISS/fastdtw dependency.
"""
import numpy as np

from ml.constants import (
    SMOOTH_ALPHA, MOTION_START_THRESHOLD, MOTION_END_THRESHOLD,
    MOTION_END_FRAMES, MIN_SIGN_FRAMES, MAX_SIGN_FRAMES,
    PREDICT_AFTER_N_FRAMES, PREDICT_EVERY_N_FRAMES,
)


def resample_sequence(sequence, target_length):
    """Linearly resample a (T, D) sequence to exactly `target_length` frames."""
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
