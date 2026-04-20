"""
Streaming sign commit via hypothesis tracking.

The classifier fires once every 3 frames while the user is signing. Naive
rules ("append if different from last committed") turn oscillations like
`oui → moi → oui → moi` into a polluted transcript because the dedup
rule only compares against the *last committed* word, not against the
pending competing candidate.

This module implements the classic **hypothesis tracker** pattern (speech
recognition, streaming OCR, gesture detection): rather than commit
per-frame, we maintain a per-candidate *support score*, determine which
candidate is the current "winner" (the hypothesis), and commit a word
only when the winner *transitions* to a different stable hypothesis.

Consequences
------------
- A sustained sign (same prediction for 20 frames) produces **one** commit
  the moment the next sign's hypothesis takes over.
- Oscillations `A ↔ B` cancel each other's support — no transition is
  ever stable enough to commit.
- A real `A B C` sequence produces three commits in order because each
  signs dominates in turn.
- There is **no hardcoded cooldown** and **no dedup window**: the commit
  rule is purely structural (transition of the winning hypothesis).

Hyperparameters are interpretable:
  promote_threshold — support needed to become the winner (≈ # of frames).
  decay — how fast a non-winning candidate forgets itself.
  min_dominance — how long a hypothesis must reign before it becomes
                  commit-eligible. This prevents a brief spurious class
                  (classifier confusion during a transition) from being
                  committed as a real sign.

Defaults are chosen so a perfect 50/50 A↔B alternation has equilibrium
support 1/(1−decay) = 1/0.4 = 2.5, strictly below promote_threshold=3.0:
no transition ever fires while the classifier flip-flops. A truly
sustained sign (no decay on self) grows linearly 1, 2, 3 and commits on
the 3rd observation (~100 ms at 30 FPS / 3-frame cadence).

With min_dominance=5 (≈500 ms at the 3-frame prediction cadence), a
transient hypothesis that briefly wins during a sign-to-sign transition
is NOT re-committed — only hypotheses that persisted through ≥5
predictions count as "real signs".
"""


import math


def _safe_exp_neg(dist: float) -> float:
    """exp(-dist) clamped to avoid overflow."""
    if dist > 50.0:
        return 0.0
    return math.exp(-dist)


class HypothesisTracker:
    """Track streaming classifier predictions and emit a word only when a
    clear hypothesis *transition* occurs.

    Typical usage in the WebSocket inference loop::

        tracker = HypothesisTracker()

        # on every intermediate / final prediction
        committed = tracker.observe(best_word)
        if committed is not None:
            current_signs.append(committed)

        # at phrase boundary (hand lost, user pressed Enter, etc.)
        committed = tracker.flush()
        if committed is not None:
            current_signs.append(committed)
        tracker.reset()
    """

    def __init__(self, promote_threshold: float = 3.0, decay: float = 0.6,
                 prune_below: float = 0.3, min_dominance: int = 5):
        self.promote_threshold = float(promote_threshold)
        self.decay = float(decay)
        self.prune_below = float(prune_below)
        self.min_dominance = int(min_dominance)
        self.support: dict[str, float] = {}
        self.current_hypothesis: str | None = None
        self.last_committed: str | None = None
        # How many consecutive observations have had current_hypothesis as
        # the winner. A short streak means the hypothesis is likely a
        # transition artefact and must NOT be committed.
        self.hypothesis_streak: int = 0

    # ------------------------------------------------------------------
    def observe(self, predicted_word: str | None,
                top_k: list[tuple[float, str]] | None = None) -> str | None:
        """Feed one prediction.

        Two modes:
          1. Legacy (top-1 only): pass `predicted_word` and leave
             `top_k=None`.  The observed candidate gets +1.0 support,
             all others decay by `self.decay`.
          2. Top-K soft mode: pass `top_k = [(dist, word), ...]` as
             returned by EnsembleEngine.predict (and optionally the
             reranker).  Each word receives support proportional to
             its probability (prob = exp(-dist), re-normalised over
             the top-K list), then all OTHER tracked words decay.
             The winner uses its probability-weighted support instead
             of a hard +1.0, so a persistent #2 that never wins top-1
             can still outpace a noisy top-1 that keeps flipping.

        Returns a word to commit NOW (push to current_signs) if a
        transition from a *stable* previous hypothesis was detected,
        else None.
        """
        if top_k:
            # --- top-K soft mode ---
            # Convert distances (-log p) back to probabilities and
            # re-normalise over the candidate list so the top-K mass
            # sums to 1.0.  Classes not in the top-K are implicitly
            # zero for this frame and will decay.
            probs = {w: float(_safe_exp_neg(d)) for d, w in top_k}
            total = sum(probs.values())
            if total <= 1e-9:
                return None
            probs = {w: p / total for w, p in probs.items()}
            # Reinforce each observed candidate by its share, decay others
            observed_set = set(probs.keys())
            for w, p in probs.items():
                self.support[w] = self.support.get(w, 0.0) + p
            for w in list(self.support):
                if w not in observed_set:
                    self.support[w] *= self.decay
                    if self.support[w] < self.prune_below:
                        del self.support[w]
        else:
            if not predicted_word:
                return None
            # Reinforce the observed candidate, decay every other
            self.support[predicted_word] = self.support.get(predicted_word, 0.0) + 1.0
            for w in list(self.support):
                if w != predicted_word:
                    self.support[w] *= self.decay
                    if self.support[w] < self.prune_below:
                        del self.support[w]

        # Who is winning?
        winner = max(self.support, key=self.support.get)
        if self.support[winner] < self.promote_threshold:
            return None  # nobody dominant enough yet

        # The hypothesis is already this word → reinforce its dominance streak.
        if winner == self.current_hypothesis:
            self.hypothesis_streak += 1
            return None

        # Transition detected. Commit the previous hypothesis ONLY if it
        # persisted long enough to count as a real sign (not a flash
        # between two real signs).
        to_commit: str | None = None
        if (self.current_hypothesis is not None
                and self.current_hypothesis != self.last_committed
                and self.hypothesis_streak >= self.min_dominance):
            to_commit = self.current_hypothesis
            self.last_committed = self.current_hypothesis

        self.current_hypothesis = winner
        self.hypothesis_streak = 1
        return to_commit

    # ------------------------------------------------------------------
    def flush(self) -> str | None:
        """Force commit the current hypothesis if it hasn't been committed
        yet. Call at phrase boundaries (user pressed Enter, hand lost for
        good, etc.). Returns the word to push, or None.

        Flush intentionally ignores min_dominance: when the user explicitly
        ends the phrase, we prefer to commit the last signal even if it was
        brief rather than silently drop it."""
        if (self.current_hypothesis is not None
                and self.current_hypothesis != self.last_committed):
            committed = self.current_hypothesis
            self.last_committed = self.current_hypothesis
            return committed
        return None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Full reset — call when starting a new phrase."""
        self.support.clear()
        self.current_hypothesis = None
        self.last_committed = None
        self.hypothesis_streak = 0
