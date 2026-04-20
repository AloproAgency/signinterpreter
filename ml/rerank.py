"""
Top-K re-ranker: exploit the 80 %+ top-5 recall of the ensemble.

Rationale
---------
The ensemble (PhonoEngine + PhonoV2Engine + RawRfEngine) reaches ~40 %
top-1 but ~81 % top-5 on cross-corpus SL.  The 41-point gap is recoverable
if we RE-COMPARE the query to the prototypes of just the top-K candidate
classes using a more expensive but more accurate distance than aggregated
feature vectors — namely, Dynamic Time Warping (DTW) on wrist + palm
trajectories.  The lattice of candidates is tiny (5 classes × ~30
templates = ~150 comparisons), so even an O(T²) pairwise distance fits in
comfortably under 100 ms on an M1.

Distance used
-------------
Per-prototype multi-stream DTW combining:
    * right wrist xyz       (weighted 1.0)
    * right palm normal xyz (weighted 0.5) — captures rotation that
                                              the wrist translation block
                                              misses
Each stream is resampled to the same length as the query so warp costs
are comparable.  The final score is 1 - exp(-dtw_cost / sigma).  σ is
the median intra-prototype DTW cost of the class (computed once at
load time) — a prototype that is an outlier within its class gets
down-weighted automatically.

Final re-ranked score per candidate class c:
    score_c = α · p_ensemble(c) + (1-α) · max_{p ∈ templates_c} sim(query, p)

A template-presence floor prevents a class with <3 templates from
dominating just because its distance is undefined (falls back to the
ensemble probability).
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml.constants import DATA_DIR, TEMPLATE_DIR, SEQUENCE_LENGTH
from ml.features import both_hands_missing, FRAME_FEATURE_DIM
from ml.segmenter import resample_sequence


# ── Stream extraction ───────────────────────────────────────────────────


_RH_WRIST = slice(10 * 3, 10 * 3 + 3)
_RH_PALM  = slice(105 + 63, 105 + 66)
_LH_WRIST = slice(9 * 3,  9 * 3  + 3)
_LH_PALM  = slice(39 + 63, 39 + 66)


def _extract_streams(seq: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        'rh_xyz':  seq[:, _RH_WRIST].astype(np.float32),
        'rh_palm': seq[:, _RH_PALM ].astype(np.float32),
        'lh_xyz':  seq[:, _LH_WRIST].astype(np.float32),
        'lh_palm': seq[:, _LH_PALM ].astype(np.float32),
    }


# ── DTW (pure numpy, fast enough for T=30) ──────────────────────────────


def _dtw_multi(a: Dict[str, np.ndarray],
               b: Dict[str, np.ndarray],
               weights: Dict[str, float]) -> float:
    """Weighted multi-stream DTW between two sign representations.
    Per-cell cost = Σ_s w_s · ‖a_s[i] - b_s[j]‖².  Returns cost
    normalised by max(Ta, Tb)."""
    ref_stream = 'rh_xyz'
    Ta = a[ref_stream].shape[0]
    Tb = b[ref_stream].shape[0]

    # Per-stream pairwise distance matrices (Ta × Tb), summed weighted
    M = np.zeros((Ta, Tb), dtype=np.float64)
    for s, w in weights.items():
        if w <= 0.0:
            continue
        As, Bs = a[s], b[s]
        # L2 distances
        diff = As[:, None, :] - Bs[None, :, :]
        M += w * (diff * diff).sum(axis=-1)

    # DTW DP with 3-neighbour rule
    D = np.full((Ta + 1, Tb + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, Ta + 1):
        for j in range(1, Tb + 1):
            D[i, j] = M[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[Ta, Tb] / max(Ta, Tb))


# ── Prototype store ─────────────────────────────────────────────────────


@dataclass
class _ClassPrototypes:
    label: str
    streams: List[Dict[str, np.ndarray]] = field(default_factory=list)
    sigma:  float = 1.0


class DtwReranker:
    """Loads per-class prototype trajectories and exposes rerank()."""

    def __init__(self,
                 template_dir: str = TEMPLATE_DIR,
                 alpha: float = 0.4,           # weight of ensemble proba
                 stream_weights: Optional[Dict[str, float]] = None,
                 max_templates_per_class: int = 30,
                 cache_path: Optional[str] = None):
        self.template_dir = template_dir
        self.alpha = float(alpha)
        self.stream_weights = stream_weights or {
            'rh_xyz':  1.0,
            'rh_palm': 0.5,
            'lh_xyz':  0.4,
            'lh_palm': 0.2,
        }
        self.max_templates = int(max_templates_per_class)
        self.cache_path = cache_path or os.path.join(
            DATA_DIR, 'rerank_prototypes.pkl')
        self._by_class: Dict[str, _ClassPrototypes] = {}
        self.loaded = False

    # --------------------------------------------------------- load
    def load(self, force_rebuild: bool = False) -> None:
        if self.loaded:
            return
        if not force_rebuild and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    self._by_class = pickle.load(f)
                self.loaded = True
                return
            except Exception:
                pass
        self._build_from_templates()
        self._save_cache()
        self.loaded = True

    def _build_from_templates(self) -> None:
        if not os.path.isdir(self.template_dir):
            return
        for word in sorted(os.listdir(self.template_dir)):
            word_dir = os.path.join(self.template_dir, word)
            if not os.path.isdir(word_dir):
                continue
            cp = _ClassPrototypes(label=word)
            for fn in sorted(os.listdir(word_dir))[: self.max_templates]:
                if not fn.endswith('.npy'):
                    continue
                try:
                    arr = np.load(os.path.join(word_dir, fn))
                except Exception:
                    continue
                if arr.ndim != 2 or arr.shape[0] != SEQUENCE_LENGTH:
                    continue
                raw = arr[:, :FRAME_FEATURE_DIM]
                if both_hands_missing(raw) > 3:
                    continue
                cp.streams.append(_extract_streams(raw))
            if len(cp.streams) < 2:
                # Can't compute intra-class sigma reliably with one; skip.
                continue
            # Per-class sigma = median intra-class DTW cost
            costs = []
            n = len(cp.streams)
            for i in range(min(n, 8)):
                for j in range(i + 1, min(n, 8)):
                    costs.append(_dtw_multi(cp.streams[i], cp.streams[j],
                                            self.stream_weights))
            cp.sigma = float(np.median(costs)) if costs else 1.0
            cp.sigma = max(cp.sigma, 1e-4)
            self._by_class[word] = cp

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self._by_class, f, protocol=4)
        except Exception:
            pass

    # --------------------------------------------------------- rerank
    def rerank(self,
               query_sequence: np.ndarray,
               ensemble_top_k: List[Tuple[float, str]],
               ensemble_probs: Optional[Dict[str, float]] = None,
               ) -> List[Tuple[float, str]]:
        """Re-rank the ensemble's top-K candidates using per-class DTW.

        Args:
            query_sequence : (T, 171) query sequence.
            ensemble_top_k : list of (dist, word) from EnsembleEngine.predict,
                             where dist = -log(prob) and the list is top-K.
            ensemble_probs : optional {word: prob} for α-blending.  If
                             missing, we convert `dist` back via exp(-dist).

        Returns a list of (dist_reranked, word) sorted ascending.
        """
        if not self.loaded or not ensemble_top_k:
            return ensemble_top_k

        q_streams = _extract_streams(
            resample_sequence(query_sequence, SEQUENCE_LENGTH)
            if query_sequence.shape[0] != SEQUENCE_LENGTH
            else query_sequence
        )

        rescored: List[Tuple[float, str]] = []
        for dist, word in ensemble_top_k:
            p_ensemble = (ensemble_probs.get(word, float(np.exp(-dist)))
                          if ensemble_probs else float(np.exp(-dist)))
            cp = self._by_class.get(word)
            if cp is None or not cp.streams:
                # Unknown class in prototypes — keep ensemble score as-is
                combined = p_ensemble
            else:
                best = min(
                    _dtw_multi(q_streams, pr, self.stream_weights)
                    for pr in cp.streams
                )
                sim_dtw = float(np.exp(-best / cp.sigma))
                combined = self.alpha * p_ensemble + (1.0 - self.alpha) * sim_dtw
            rescored.append((float(-np.log(max(1e-9, combined))), word))

        rescored.sort(key=lambda t: t[0])
        return rescored
