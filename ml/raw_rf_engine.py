"""RF classifier on flattened raw joint features (30 × 177 = 5310 dims)."""
import json
import os
import pickle
import numpy as np

from ml.constants import DATA_DIR, SEQUENCE_LENGTH
from ml.inference_engine import resample_sequence, SignSegmenter  # noqa: F401 reused
from ml.features import (
    FRAME_FEATURE_DIM, interpolate_holes, to_right_dominant, add_wrist_velocity,
)

RF_DIR = os.path.join(DATA_DIR, 'rf_raw')
MODEL_PATH = os.path.join(RF_DIR, 'rf.pkl')
META_PATH = os.path.join(RF_DIR, 'meta.json')


class RawRfEngine:
    def __init__(self):
        self.loaded = False
        self.clf = None
        self.mu = None
        self.sd = None
        self.classes = []

    def load(self):
        if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
            print(f'WARNING: raw-RF model not found at {MODEL_PATH}.')
            self.loaded = False
            return
        with open(MODEL_PATH, 'rb') as f:
            self.clf = pickle.load(f)
        with open(META_PATH) as f:
            meta = json.load(f)
        self.classes = list(meta['classes'])
        self.mu = np.asarray(meta['mu'], dtype='float32')
        self.sd = np.asarray(meta['sd'], dtype='float32')
        self.sd = np.where(self.sd < 1e-6, 1.0, self.sd)
        self.loaded = True
        print(f'RawRfEngine loaded: {len(self.classes)} classes -> {self.classes}')

    def reload(self):
        self.load()

    @property
    def words(self):
        return list(self.classes)

    def _vectorise(self, seq):
        q = resample_sequence(seq, SEQUENCE_LENGTH)
        if q.shape[1] >= FRAME_FEATURE_DIM:
            q = q[:, :FRAME_FEATURE_DIM]
        q = interpolate_holes(q)
        q = to_right_dominant(q)
        q = add_wrist_velocity(q)
        return q.flatten()

    def predict(self, query_sequence):
        if not self.loaded:
            return None, None, []
        x = ((self._vectorise(query_sequence) - self.mu) / self.sd).reshape(1, -1)
        probs = self.clf.predict_proba(x)[0]
        order = np.argsort(probs)[::-1]
        eps = 1e-9
        top_k = []
        for idx in order[:5]:
            word = self.classes[idx]
            dist = float(-np.log(max(eps, probs[idx])))
            top_k.append((dist, word))
        return top_k[0][1], top_k[0][0], top_k
