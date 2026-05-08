"""LSTM-based sign classifier — V11 inference engine.

Drops in as a direct replacement for EnsembleEngine:
  engine.loaded        bool
  engine.words         list[str]
  engine.load()
  engine.reload()
  engine.predict(frames) -> (best_word, best_dist, top_k)
    best_dist  = -log(prob),  lower = more confident
    top_k      = [(dist, word), …] sorted ascending
"""
from __future__ import annotations
import json
import os
import numpy as np

_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(_ROOT, 'data', 'lstm', 'model.keras')
CLASSES_PATH = os.path.join(_ROOT, 'data', 'lstm', 'classes.json')
SEQUENCE_LENGTH = 30
TOP_K = 10


def _resample(frames, n: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Linear resampling to exactly *n* frames (same logic as segmenter.py)."""
    arr = np.asarray(frames, dtype='float32')
    if len(arr) == n:
        return arr
    if len(arr) < 2:
        return np.tile(arr[0:1], (n, 1)) if len(arr) else np.zeros((n, arr.shape[-1]))
    idx = np.linspace(0, len(arr) - 1, n)
    left  = np.floor(idx).astype(int)
    right = np.minimum(left + 1, len(arr) - 1)
    t     = (idx - left)[:, None]
    return (arr[left] * (1 - t) + arr[right] * t).astype('float32')


class LSTMEngine:
    """BiLSTM + temporal-attention classifier trained on V8 templates."""

    def __init__(self):
        self._model  = None
        self._words: list[str] = []
        self.loaded  = False

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
            return
        import tensorflow as tf          # lazy — avoids startup cost at import time
        # safe_mode=False needed for Lambda layer used in attention pooling
        self._model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
        with open(CLASSES_PATH, encoding='utf-8') as f:
            self._words = json.load(f)
        self.loaded = True

    def reload(self) -> None:
        self.loaded  = False
        self._model  = None
        self._words  = []
        self.load()

    @property
    def words(self) -> list[str]:
        return self._words

    # -------------------------------------------------------------- predict
    def predict(
        self,
        frames,
    ) -> tuple[str | None, float, list]:
        """Classify a variable-length sign sequence.

        Parameters
        ----------
        frames : list or ndarray of shape (T, 171)

        Returns
        -------
        (best_word, best_dist, top_k)
        """
        if not self.loaded or self._model is None:
            return None, float('inf'), []

        seq = _resample(frames)[:, :171]    # (30, 171)
        inp = seq[np.newaxis, ...]           # (1, 30, 171)

        probs = self._model.predict(inp, verbose=0)[0]  # (n_classes,)
        dists = -np.log(np.clip(probs, 1e-9, 1.0))

        order  = np.argsort(dists)
        top_k  = [(float(dists[i]), self._words[i]) for i in order[:TOP_K]]
        return top_k[0][1], top_k[0][0], top_k
