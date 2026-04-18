"""
Phonological classifier: Random Forest on a 34-dim phonological descriptor.

Drop-in replacement for V8's InferenceEngine. Provides:
  predict(query_sequence) -> (best_word, distance, top_k_list)
  load(), reload()
"""
import json
import os
import pickle
import numpy as np

from ml.constants import DATA_DIR, SEQUENCE_LENGTH
from ml.segmenter import resample_sequence, SignSegmenter  # noqa: F401  reused
from ml.features import both_hands_missing
from ml.phono_features import phonological_descriptor, PHONO_DIM

PHONO_DIR = os.path.join(DATA_DIR, 'phono')
MODEL_PATH = os.path.join(PHONO_DIR, 'rf.pkl')
META_PATH = os.path.join(PHONO_DIR, 'meta.json')


class PhonoEngine:
    def __init__(self):
        self.loaded = False
        self.clf = None
        self.scaler_mean = None
        self.scaler_std = None
        self.classes = []

    def load(self):
        if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
            print(f'WARNING: phono model not found at {MODEL_PATH}.')
            self.loaded = False
            return
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.clf = pickle.load(f)
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
            self.classes = list(meta['classes'])
            self.scaler_mean = np.asarray(meta['scaler_mean'], dtype='float32')
            self.scaler_std = np.asarray(meta['scaler_std'], dtype='float32')
            self.scaler_std = np.where(self.scaler_std < 1e-6, 1.0, self.scaler_std)
            self.loaded = True
            print(f'PhonoEngine loaded: {len(self.classes)} classes -> {self.classes}')
        except Exception as e:
            print(f'ERROR loading PhonoEngine: {e}')
            self.loaded = False

    def reload(self):
        self.load()

    @property
    def words(self):
        return list(self.classes)

    def predict(self, query_sequence):
        if not self.loaded:
            return None, None, []

        q = resample_sequence(query_sequence, SEQUENCE_LENGTH)
        desc = phonological_descriptor(q)
        x = ((desc - self.scaler_mean) / self.scaler_std).reshape(1, -1)

        probs = self.clf.predict_proba(x)[0]
        order = np.argsort(probs)[::-1]
        eps = 1e-9
        top_k = []
        for idx in order[:5]:
            word = self.classes[idx]
            dist = float(-np.log(max(eps, probs[idx])))
            top_k.append((dist, word))
        best_dist, best_word = top_k[0]
        return best_word, best_dist, top_k
