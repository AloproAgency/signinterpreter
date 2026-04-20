"""
3-way ensemble: PhonoEngine (34-dim) + PhonoV2Engine (72-dim, v3) + RawRfEngine (5310-dim).

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
from ml.segmenter import resample_sequence, SignSegmenter  # noqa: F401 reused
from ml.phono_features import phonological_descriptor
from ml.phono_features_v2 import phonological_descriptor_v2
from ml.rerank import DtwReranker


class EnsembleEngine:
    def __init__(self, phono_weight=0.20, phono_v2_weight=0.40, raw_weight=0.40,
                 use_rerank: bool = True, rerank_alpha: float = 0.4):
        self.phono = PhonoEngine()
        self.phono_v2 = PhonoV2Engine()
        self.raw = RawRfEngine()
        self.phono_weight = float(phono_weight)
        self.phono_v2_weight = float(phono_v2_weight)
        self.raw_weight = float(raw_weight)
        self.classes = []
        self.loaded = False
        # Top-K DTW re-ranker — leverages the ensemble's high top-5 recall
        # by recomparing the query to the prototypes of the top-K candidate
        # classes using trajectory-level DTW.  See ml/rerank.py.
        self.use_rerank = bool(use_rerank)
        self.reranker: DtwReranker | None = (
            DtwReranker(alpha=rerank_alpha) if use_rerank else None
        )

    def load(self):
        self.phono.load()
        self.phono_v2.load()
        self.raw.load()
        loaded_members = [m for m in (self.phono, self.phono_v2, self.raw) if m.loaded]
        if not loaded_members:
            print('WARNING: no ensemble member could load.')
            self.loaded = False
            return
        # Union of all class lists across loaded members. If one member is
        # missing a class (e.g. rebuild filtered a bad-quality class), we
        # still accept it — predictions are aligned by class name at
        # predict-time (see _aligned_probs) so the shape mismatch that
        # caused historic crashes can no longer happen.
        seen = []
        seen_set = set()
        for m in loaded_members:
            for c in m.classes:
                if c not in seen_set:
                    seen.append(c)
                    seen_set.add(c)
        self.classes = seen
        for m in loaded_members:
            if list(m.classes) != self.classes:
                missing = [c for c in self.classes if c not in m.classes]
                name = type(m).__name__
                print(f'WARNING: {name} is missing {len(missing)} '
                      f'class(es): {missing}  (will contribute 0 proba there)')
        self._class_index = {c: i for i, c in enumerate(self.classes)}
        self.loaded = True
        print(f'EnsembleEngine loaded: {len(self.classes)} classes '
              f'(weights phono={self.phono_weight:.2f} '
              f'phono_v2={self.phono_v2_weight:.2f} '
              f'raw={self.raw_weight:.2f})')
        if self.reranker is not None:
            try:
                self.reranker.load()
                print(f'DtwReranker loaded: {len(self.reranker._by_class)} '
                      f'class prototype banks.')
            except Exception as e:
                print(f'WARNING: DtwReranker disabled ({e})')
                self.reranker = None
                self.use_rerank = False

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

    def _aligned_probs(self, member, probs_raw):
        """Project a member's (n_member,) probability vector onto the
        ensemble's canonical class list. Classes the member does not
        know get 0. Returns an (n_canonical,) vector plus a mask of
        which positions the member actually contributes to."""
        out = np.zeros(len(self.classes), dtype='float32')
        mask = np.zeros(len(self.classes), dtype=bool)
        for j, c in enumerate(member.classes):
            i = self._class_index.get(c)
            if i is None:
                continue
            out[i] = probs_raw[j]
            mask[i] = True
        return out, mask

    def predict(self, query_sequence):
        if not self.loaded:
            return None, None, []

        K = len(self.classes)
        probs = np.zeros(K, dtype='float32')
        weight_per_class = np.zeros(K, dtype='float32')

        if self.phono.loaded:
            p, m = self._aligned_probs(self.phono, self._phono_probs(query_sequence))
            probs += self.phono_weight * p
            weight_per_class[m] += self.phono_weight
        if self.phono_v2.loaded:
            p, m = self._aligned_probs(self.phono_v2, self._phono_v2_probs(query_sequence))
            probs += self.phono_v2_weight * p
            weight_per_class[m] += self.phono_v2_weight
        if self.raw.loaded:
            p, m = self._aligned_probs(self.raw, self._raw_probs(query_sequence))
            probs += self.raw_weight * p
            weight_per_class[m] += self.raw_weight

        # Per-class normalisation: a class seen only by a subset of the
        # members is still comparable because we divide by the weight
        # that actually voted for it.
        valid = weight_per_class > 1e-9
        if not valid.any():
            return None, None, []
        probs = np.where(valid, probs / np.maximum(weight_per_class, 1e-9), 0.0)

        order = np.argsort(probs)[::-1]
        eps = 1e-9
        top_k = []
        ensemble_probs_dict: dict[str, float] = {}
        for idx in order[:5]:
            if probs[idx] <= 0:
                continue
            word = self.classes[idx]
            p = float(probs[idx])
            ensemble_probs_dict[word] = p
            dist = float(-np.log(max(eps, p)))
            top_k.append((dist, word))
        if not top_k:
            return None, None, []

        # DTW re-ranking on the top-5 candidates.  Only kicks in when
        # the reranker is loaded AND when there's a genuine disambiguation
        # to do (margin between top-1 and top-2 is small enough that a
        # trajectory-level second opinion is worth the compute).
        if (self.use_rerank and self.reranker is not None
                and self.reranker.loaded and len(top_k) >= 2):
            top_k = self.reranker.rerank(query_sequence, top_k,
                                         ensemble_probs_dict)

        return top_k[0][1], top_k[0][0], top_k
