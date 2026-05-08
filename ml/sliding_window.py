"""Continuous sign classification via a sliding 30-frame window.

Replaces the EnrichedSignSegmenter + HypothesisTracker combo.

Why this beats explicit segmentation
─────────────────────────────────────
Boundary errors (start too early / end too early) cut signs in half,
producing feature vectors that look like different signs.  The sliding
window always feeds a full WINDOW-frame buffer to the LSTM, so the model
sees a complete (or near-complete) sign regardless of boundary precision.

Adaptive K_STABLE
─────────────────
Different signs have different speeds.  Rather than a single fixed
stability requirement, K_STABLE scales with confidence:

  dist < FAST_THRESH  (prob > 0.70) → K_STABLE = 2  (~180 ms)  fast commit
  dist < MID_THRESH   (prob > 0.58) → K_STABLE = 3  (~270 ms)  normal
  otherwise                         → not committed  (below threshold)

High-confidence rapid signs (letters, single-movement signs) emit in 2
strides; ambiguous signs must hold for 3 strides before committing.
The post-emit gap lock prevents the same window from double-emitting.
"""
from __future__ import annotations
import collections
import numpy as np

WINDOW       = 30    # matches LSTM training length
STRIDE       = 3     # run classifier every N frames (~10 Hz at 30 fps)
K_GAP        = 2     # strides of different/uncertain word to release post-emit lock
FAST_THRESH  = 0.35  # dist < this → prob > 0.70 → emit after K_FAST strides
MID_THRESH   = 0.55  # dist < this → prob > 0.58 → emit after K_MID strides
K_FAST       = 2     # strides needed at high confidence  (~180 ms)
K_MID        = 3     # strides needed at mid  confidence  (~270 ms)


class SlidingWindowClassifier:
    """Feed one frame at a time via process(); get committed words back."""

    def __init__(self):
        self._engine    = None
        self._buf: collections.deque = collections.deque(maxlen=WINDOW)
        self._frame_n   = 0

        self._streak_word: str | None = None
        self._streak: int = 0

        self._last_emitted: str | None = None
        self._gap_count: int = 0

        # Exposed for UI
        self.motion_energy: float = 0.0
        self.last_word: str | None = None
        self.last_dist: float = float('inf')
        self.last_top_k: list = []

    def set_engine(self, engine) -> None:
        self._engine = engine

    # ── public API ────────────────────────────────────────────────────

    def process(self, features: np.ndarray) -> str | None:
        """Feed one frame. Returns the committed word or None."""
        self._buf.append(np.asarray(features, dtype='float32'))
        self._frame_n += 1

        # Motion proxy for UI
        if len(self._buf) >= 2:
            arr = np.asarray(list(self._buf)[-2:])
            self.motion_energy = float(np.mean(np.abs(arr[1] - arr[0])))

        if self._engine is None or len(self._buf) < WINDOW:
            return None
        if self._frame_n % STRIDE != 0:
            return None

        word, dist, top_k = self._engine.predict(list(self._buf))
        self.last_word  = word
        self.last_dist  = dist
        self.last_top_k = top_k

        # Determine required stability for this confidence level
        if dist < FAST_THRESH:
            k_needed = K_FAST
        elif dist < MID_THRESH:
            k_needed = K_MID
        else:
            # Below threshold — reset streak, count as gap
            self._streak_word = None
            self._streak      = 0
            if self._last_emitted is not None:
                self._gap_count += 1
                if self._gap_count >= K_GAP:
                    self._last_emitted = None
                    self._gap_count    = 0
            return None

        # ── post-emit gap lock ────────────────────────────────────────
        if self._last_emitted is not None:
            if word != self._last_emitted:
                self._gap_count += 1
            else:
                self._gap_count = 0
            if self._gap_count >= K_GAP:
                self._last_emitted = None
                self._gap_count    = 0

        # ── streak tracking ───────────────────────────────────────────
        if word == self._streak_word:
            self._streak += 1
        else:
            self._streak_word = word
            self._streak      = 1

        # ── emit ──────────────────────────────────────────────────────
        if self._streak >= k_needed and self._last_emitted is None:
            emitted            = self._streak_word
            self._last_emitted = emitted
            self._gap_count    = 0
            self._streak       = 0
            self._streak_word  = None
            return emitted

        return None

    @property
    def is_active(self) -> bool:
        return self._streak > 0

    @property
    def streak_progress(self) -> float:
        k = K_FAST if self.last_dist < FAST_THRESH else K_MID
        return min(self._streak / k, 1.0) if self._streak > 0 else 0.0

    def reset(self) -> None:
        self._buf.clear()
        self._frame_n      = 0
        self._streak_word  = None
        self._streak       = 0
        self._last_emitted = None
        self._gap_count    = 0
        self.motion_energy = 0.0
        self.last_word     = None
        self.last_dist     = float('inf')
        self.last_top_k    = []

    def reset_streak(self) -> None:
        """Reset only the streak/emit state — keeps the frame buffer warm.
        Use between signs so the 30-frame warmup period is not lost."""
        self._streak_word  = None
        self._streak       = 0
        self._last_emitted = None
        self._gap_count    = 0
