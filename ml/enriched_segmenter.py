"""
Enriched sign segmenter — uses linguistic & biomechanical priors.

At every frame we compute 7 signals (see ml/segmenter_features.py) and
combine them into a scalar "sign_score" ∈ [0, 1]. A two-state machine
with hysteresis drives the start/end decisions:

   OUT  ── score ≥ SCORE_START for START_STREAK frames ──►  IN
   IN   ── score ≤ SCORE_END   for END_STREAK   frames ──►  OUT

Exposes exactly the same `process_features` API as SignSegmenter so it is
a drop-in replacement: the return dict keys are identical.
"""
import collections
import numpy as np

from ml.constants import (
    SMOOTH_ALPHA, MIN_SIGN_FRAMES, MAX_SIGN_FRAMES,
    PREDICT_AFTER_N_FRAMES, PREDICT_EVERY_N_FRAMES,
)
from ml.segmenter_features import compute_sign_score

# Hysteresis thresholds on the composite sign_score
SCORE_START = 0.55
SCORE_END = 0.35
START_STREAK = 2        # ~67 ms at 30 FPS — snappy onset
END_STREAK = 3          # ~100 ms at 30 FPS — snappy offset

HISTORY_WINDOW = 10     # frames kept for velocity / trajectory / bimanual


class EnrichedSignSegmenter:
    """Drop-in replacement for SignSegmenter driven by a composite
    linguistic/biomechanical score instead of raw motion energy."""

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.is_signing = False
        self.sign_buffer = []
        self.smoothed_features = None
        self.prev_features = None
        self.motion_energy = 0.0

        self.start_streak = 0
        self.end_streak = 0

        # Rolling histories (short, bounded)
        self.energy_history = collections.deque(maxlen=HISTORY_WINDOW)
        self.pose_history = collections.deque(maxlen=HISTORY_WINDOW)
        self.wrist_history = collections.deque(maxlen=HISTORY_WINDOW)
        self.lh_history = collections.deque(maxlen=HISTORY_WINDOW)
        self.rh_history = collections.deque(maxlen=HISTORY_WINDOW)

        self._just_recovered = False
        self.last_score = 0.0
        self.last_signals = {}

    # ------------------------------------------------------------------
    def notify_hand_lost(self):
        """Called by the WS layer when MediaPipe drops the hand. Next good
        frame skips the motion delta (else we'd see a spurious spike)."""
        self._just_recovered = True

    # ------------------------------------------------------------------
    def _update_histories(self):
        """Push a few derived vectors into rolling histories for the signal
        functions (velocity fit, trajectory, bimanual correlation)."""
        f = self.smoothed_features
        self.energy_history.append(self.motion_energy)
        self.pose_history.append(f[0:39].copy())                # upper-body pose
        # Pick dominant wrist: whichever has higher visibility-proxy (non-zero
        # hand block). Default to right. Used for trajectory signal.
        rh_block = f[105:168]
        lh_block = f[39:102]
        if np.any(rh_block != 0):
            self.wrist_history.append(f[30:33].copy())          # right wrist xyz
        elif np.any(lh_block != 0):
            self.wrist_history.append(f[27:30].copy())          # left wrist xyz
        else:
            self.wrist_history.append(np.zeros(3, dtype='float32'))
        self.lh_history.append(lh_block.copy())
        self.rh_history.append(rh_block.copy())

    # ------------------------------------------------------------------
    def process_features(self, raw_features):
        # Smooth
        if self.smoothed_features is None:
            self.smoothed_features = raw_features.astype('float32')
        else:
            self.smoothed_features = (
                SMOOTH_ALPHA * raw_features
                + (1 - SMOOTH_ALPHA) * self.smoothed_features
            ).astype('float32')

        # Motion on the hand dimensions only
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

        # Update rolling histories
        self._update_histories()

        # Composite sign_score from the 7 signals
        score, signals = compute_sign_score(
            self.motion_energy,
            self.smoothed_features,
            list(self.energy_history),
            list(self.pose_history),
            list(self.wrist_history),
            list(self.lh_history),
            list(self.rh_history),
        )
        self.last_score = score
        self.last_signals = signals

        sign_ended = False
        ended_buffer = None

        # ─── State machine ────────────────────────────────────────────
        if not self.is_signing:
            if score >= SCORE_START:
                self.start_streak += 1
            else:
                self.start_streak = 0
            if self.start_streak >= START_STREAK:
                self.is_signing = True
                self.sign_buffer = [self.smoothed_features.copy()]
                self.start_streak = 0
                self.end_streak = 0
        else:
            self.sign_buffer.append(self.smoothed_features.copy())

            if score <= SCORE_END:
                self.end_streak += 1
            else:
                self.end_streak = 0

            end_by_score = self.end_streak >= END_STREAK
            end_by_max = len(self.sign_buffer) >= MAX_SIGN_FRAMES

            if end_by_score or end_by_max:
                # Drop trailing low-score frames (we know at least END_STREAK
                # of them triggered the end)
                trim = len(self.sign_buffer)
                if end_by_score:
                    trim = max(MIN_SIGN_FRAMES, len(self.sign_buffer) - END_STREAK)
                trimmed = self.sign_buffer[:trim]
                if len(trimmed) >= MIN_SIGN_FRAMES:
                    sign_ended = True
                    ended_buffer = trimmed
                self.is_signing = False
                self.sign_buffer = []
                self.end_streak = 0

        # ─── Intermediate prediction cadence (same contract as SignSegmenter)
        should_predict = (
            self.is_signing
            and len(self.sign_buffer) >= PREDICT_AFTER_N_FRAMES
            and len(self.sign_buffer) % PREDICT_EVERY_N_FRAMES == 0
        )

        return {
            'motion_energy': self.motion_energy,
            'is_signing': self.is_signing,
            'buffer_length': len(self.sign_buffer),
            'sign_ended': sign_ended,
            'ended_buffer': ended_buffer,
            'should_predict_intermediate': should_predict,
            'current_buffer': list(self.sign_buffer) if should_predict else None,
            # Diagnostics — useful for debug HUD but ignored by inference_ws
            'sign_score': self.last_score,
            'signals': self.last_signals,
        }
