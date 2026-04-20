"""
Phonological classifier v2: Random Forest on a PHONO_DIM_V2-dim
phonological descriptor (currently 72 dims after v3 directional add).

Same structure as PhonoEngine but uses phonological_descriptor_v2
(extended with angular velocities, curvature, autocorrelation, per-
finger curls, bimanual distances, and — v3 — 12 SIGNED directional
movement features that distinguish signs with identical handshape
but different motion direction).

NB. Any model pickle trained before the v3 change was fitted with
60 features.  The engine detects this mismatch at load time and
refuses to use a stale model rather than crash on predict_proba.
"""
import json
import os
import pickle
import numpy as np

from ml.constants import DATA_DIR, SEQUENCE_LENGTH
from ml.segmenter import resample_sequence  # reused
from ml.phono_features_v2 import phonological_descriptor_v2, PHONO_DIM_V2  # noqa: F401

PHONO_V2_DIR = os.path.join(DATA_DIR, 'phono_v2')
MODEL_PATH = os.path.join(PHONO_V2_DIR, 'rf.pkl')
META_PATH = os.path.join(PHONO_V2_DIR, 'meta.json')


class PhonoV2Engine:
    def __init__(self):
        self.loaded = False
        self.clf = None
        self.scaler_mean = None
        self.scaler_std = None
        self.classes = []

    def load(self):
        if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
            print(f'WARNING: phono v2 model not found at {MODEL_PATH}.')
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

            # Dim-mismatch guard: the extractor may have grown (e.g. v3
            # added 12 directional dims → 60→72) since this pickle was
            # trained.  If so, refuse to serve predictions instead of
            # crashing inside predict_proba on a shape mismatch.
            expected = getattr(self.clf, 'n_features_in_', None)
            if expected is not None and expected != PHONO_DIM_V2:
                print(f'WARNING: phono v2 model was trained with {expected} '
                      f'features but extractor now emits {PHONO_DIM_V2}. '
                      f'Retrain with scripts/train_phono.py before using.')
                self.loaded = False
                return

            self.loaded = True
            print(f'PhonoV2Engine loaded: {len(self.classes)} classes -> {self.classes}')
        except Exception as e:
            print(f'ERROR loading PhonoV2Engine: {e}')
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
        desc = phonological_descriptor_v2(q)
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
