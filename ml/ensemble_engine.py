"""
Ensemble of PhonoEngine (34 semantic features) + RawRfEngine (5310 raw joint features).

Both classifiers produce softmax probabilities over the same set of classes.
We average (or weight-average) them and pick the argmax.

Drop-in replacement for the V8 InferenceEngine API.
"""
import numpy as np

from ml.phono_engine import PhonoEngine
from ml.raw_rf_engine import RawRfEngine
from ml.constants import SEQUENCE_LENGTH
from ml.inference_engine import resample_sequence, SignSegmenter  # noqa: F401 reused
from ml.phono_features import phonological_descriptor


class EnsembleEngine:
    def __init__(self, phono_weight=0.5, raw_weight=0.5):
        self.phono = PhonoEngine()
        self.raw = RawRfEngine()
        self.phono_weight = float(phono_weight)
        self.raw_weight = float(raw_weight)
        self.classes = []
        self.loaded = False

    def load(self):
        self.phono.load()
        self.raw.load()
        if not (self.phono.loaded and self.raw.loaded):
            print('WARNING: at least one member of the ensemble failed to load.')
            self.loaded = self.phono.loaded or self.raw.loaded
            # Use whichever loaded for class list
            self.classes = list(self.phono.classes or self.raw.classes)
            return
        # Align classes: both must agree
        if list(self.phono.classes) != list(self.raw.classes):
            print('WARNING: phono/raw class lists disagree; using phono order.')
        self.classes = list(self.phono.classes)
        self.loaded = True
        print(f'EnsembleEngine loaded: {len(self.classes)} classes '
              f'(weights phono={self.phono_weight:.2f} raw={self.raw_weight:.2f})')

    def reload(self):
        self.load()

    @property
    def words(self):
        return list(self.classes)

    def _phono_probs(self, query_sequence):
        q = resample_sequence(query_sequence, SEQUENCE_LENGTH)
        desc = phonological_descriptor(q)
        x = ((desc - self.phono.scaler_mean) / self.phono.scaler_std).reshape(1, -1)
        return self.phono.clf.predict_proba(x)[0]

    def _raw_probs(self, query_sequence):
        x = ((self.raw._vectorise(query_sequence) - self.raw.mu) / self.raw.sd).reshape(1, -1)
        return self.raw.clf.predict_proba(x)[0]

    def predict(self, query_sequence):
        if not self.loaded:
            return None, None, []

        # Fall back to whichever single engine is loaded
        if not self.phono.loaded:
            return self.raw.predict(query_sequence)
        if not self.raw.loaded:
            return self.phono.predict(query_sequence)

        p_phono = self._phono_probs(query_sequence)
        p_raw = self._raw_probs(query_sequence)

        # Align class indices (both should match; be defensive)
        if list(self.phono.classes) == list(self.raw.classes):
            probs = self.phono_weight * p_phono + self.raw_weight * p_raw
        else:
            # Map raw probs into phono's class ordering
            mapped = np.zeros_like(p_phono)
            for i, c in enumerate(self.raw.classes):
                if c in self.phono.classes:
                    mapped[self.phono.classes.index(c)] = p_raw[i]
            probs = self.phono_weight * p_phono + self.raw_weight * mapped

        order = np.argsort(probs)[::-1]
        eps = 1e-9
        top_k = []
        for idx in order[:5]:
            if probs[idx] <= 0:
                continue
            word = self.classes[idx]
            dist = float(-np.log(max(eps, probs[idx])))
            top_k.append((dist, word))
        if not top_k:
            return None, None, []
        return top_k[0][1], top_k[0][0], top_k
