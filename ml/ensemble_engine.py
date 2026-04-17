"""
3-way ensemble: PhonoEngine (34-dim) + PhonoV2Engine (60-dim) + RawRfEngine (5310-dim).

The three classifiers produce softmax probabilities over the same class set.
We sum weighted probabilities and pick the argmax.

Default weights from Round 2 sweep (92.0% WLASL cross-signer):
  phono_v1 = 0.20, phono_v2 = 0.40, raw = 0.40
"""
import numpy as np

from ml.phono_engine import PhonoEngine
from ml.phono_v2_engine import PhonoV2Engine
from ml.raw_rf_engine import RawRfEngine
from ml.constants import SEQUENCE_LENGTH
from ml.inference_engine import resample_sequence, SignSegmenter  # noqa: F401 reused
from ml.phono_features import phonological_descriptor
from ml.phono_features_v2 import phonological_descriptor_v2


class EnsembleEngine:
    def __init__(self, phono_weight=0.20, phono_v2_weight=0.40, raw_weight=0.40):
        self.phono = PhonoEngine()
        self.phono_v2 = PhonoV2Engine()
        self.raw = RawRfEngine()
        self.phono_weight = float(phono_weight)
        self.phono_v2_weight = float(phono_v2_weight)
        self.raw_weight = float(raw_weight)
        self.classes = []
        self.loaded = False

    def load(self):
        self.phono.load()
        self.phono_v2.load()
        self.raw.load()
        loaded_members = [m for m in (self.phono, self.phono_v2, self.raw) if m.loaded]
        if not loaded_members:
            print('WARNING: no ensemble member could load.')
            self.loaded = False
            return
        # Align on whatever class list is available
        self.classes = list(loaded_members[0].classes)
        for m in loaded_members[1:]:
            if list(m.classes) != self.classes:
                print('WARNING: ensemble members have different class lists.')
        self.loaded = True
        print(f'EnsembleEngine loaded: {len(self.classes)} classes '
              f'(weights phono={self.phono_weight:.2f} '
              f'phono_v2={self.phono_v2_weight:.2f} '
              f'raw={self.raw_weight:.2f})')

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

    def _phono_v2_probs(self, query_sequence):
        q = resample_sequence(query_sequence, SEQUENCE_LENGTH)
        desc = phonological_descriptor_v2(q)
        x = ((desc - self.phono_v2.scaler_mean) / self.phono_v2.scaler_std).reshape(1, -1)
        return self.phono_v2.clf.predict_proba(x)[0]

    def _raw_probs(self, query_sequence):
        x = ((self.raw._vectorise(query_sequence) - self.raw.mu) / self.raw.sd).reshape(1, -1)
        return self.raw.clf.predict_proba(x)[0]

    def predict(self, query_sequence):
        if not self.loaded:
            return None, None, []

        probs = np.zeros(len(self.classes), dtype='float32')
        weight_total = 0.0

        if self.phono.loaded:
            probs += self.phono_weight * self._phono_probs(query_sequence)
            weight_total += self.phono_weight
        if self.phono_v2.loaded:
            probs += self.phono_v2_weight * self._phono_v2_probs(query_sequence)
            weight_total += self.phono_v2_weight
        if self.raw.loaded:
            probs += self.raw_weight * self._raw_probs(query_sequence)
            weight_total += self.raw_weight

        if weight_total <= 0:
            return None, None, []
        probs = probs / weight_total

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
