"""WebSocket endpoint for real-time inference.

MediaPipe runs client-side (browser). The client sends 171-float feature vectors
as JSON messages; the server only handles segmentation, kNN lookup, and translation.
"""
import json
import os
import time
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ml.inference_engine import InferenceEngine, SignSegmenter
from ml.phono_engine import PhonoEngine
from ml.translator import get_translator
from ml.constants import THRESHOLD
from ml.features import FRAME_FEATURE_DIM

router = APIRouter()

# Use phonological RF when trained; otherwise fall back to KNN+DTW.
ENGINE_KIND = os.environ.get('ENGINE', 'ensemble').lower()
if ENGINE_KIND == 'ensemble':
    from ml.ensemble_engine import EnsembleEngine
    # Round 2 optimal: 3-way mix reaches 92% on WLASL cross-signer eval.
    engine = EnsembleEngine(phono_weight=0.20, phono_v2_weight=0.40, raw_weight=0.40)
elif ENGINE_KIND == 'raw_rf':
    from ml.raw_rf_engine import RawRfEngine
    engine = RawRfEngine()
elif ENGINE_KIND == 'dtw':
    engine = InferenceEngine()
else:
    engine = PhonoEngine()

PAUSE_THRESHOLD = 1.2        # seconds without hand = failsafe finalize
REST_WRIST_Y = 0.8           # wrist y (in body frame, shoulder-normalised) ≥ this
                             # means the wrist is clearly below the shoulders
REST_MOTION_MAX = 0.02       # hand motion must be below this to count as "rest"
REST_FRAMES = 5              # consecutive at-rest frames before finalising (≈170 ms)
# Indices of the wrist y coordinate in the 171-dim feature vector
# (UPPER_BODY_LANDMARKS layout: l_wrist is pose-index 9, r_wrist is 10 → y = idx*3+1)
LW_Y_IDX, RW_Y_IDX = 9 * 3 + 1, 10 * 3 + 1


def _is_at_rest(features, motion_energy):
    """True when both wrists sit below the shoulder line AND motion is low.
    Handles one-handed rest too: if only one wrist is active (the other was
    already resting), the active one dropping below shoulders is enough."""
    lw_y = float(features[LW_Y_IDX])
    rw_y = float(features[RW_Y_IDX])
    # Both wrists visible and both below shoulder line → rest
    wrists_low = (lw_y >= REST_WRIST_Y) and (rw_y >= REST_WRIST_Y)
    return wrists_low and motion_energy < REST_MOTION_MAX


def get_engine():
    if not engine.loaded:
        engine.load()
    return engine


@router.websocket('/ws/inference')
async def inference_websocket(ws: WebSocket):
    await ws.accept()

    segmenter = SignSegmenter()
    eng = get_engine()
    translator = get_translator()

    # State
    current_signs = []             # signs for the current phrase being built
    completed_phrases = []         # list of translated phrases (finalized)
    completed_signs = []           # list of raw sign sequences (one list per phrase)
    completed_scores = []          # translator confidence per completed phrase
    threshold = THRESHOLD

    def translation_confidence(logp_per_sign: float) -> float:
        """Map translator log-prob (in [−3, 0]) to a 0..1 confidence."""
        import math
        return max(0.0, min(1.0, math.exp(max(-3.0, logp_per_sign))))

    async def _do_finalize():
        """Prune outlier signs, translate the cleaned sequence, push to
        completed_phrases and broadcast the update."""
        nonlocal current_signs
        if not current_signs:
            return
        signs = list(current_signs)
        original = list(current_signs)
        if translator.loaded and len(signs) >= 3:
            signs, removed = translator.prune_intruders(signs, min_gain=0.15, min_keep=1)
            if removed:
                print(f'[prune] dropped {removed} from {original} -> {signs}')
        if translator.loaded and len(signs) >= 2:
            # Search for a better ordering that maximises sentence fluency.
            reordered, swapped = translator.reorder_signs(signs, min_gain=0.1)
            if swapped:
                print(f'[reorder] {signs} -> {reordered}')
                signs = reordered
        if translator.loaded and signs:
            phrase = translator.translate(signs)
            score = translation_confidence(translator.score_signs(signs))
        else:
            phrase = ' '.join(signs)
            score = 1.0
        completed_phrases.append(phrase)
        completed_signs.append(signs)
        completed_scores.append(round(score, 2))
        current_signs = []
        await ws.send_json({
            'type': 'sentence_update',
            'sentence': [],
            'translated': '',
            'translated_score': 0,
            'phrases': completed_phrases,
            'phrase_signs': completed_signs,
            'phrase_scores': completed_scores,
        })

    last_hand_time = time.time()   # stamped when the hand first disappears
    hand_missing_streak = 0         # consecutive frames without a hand
    rest_streak = 0                 # consecutive at-rest frames (hands low + idle)
    HAND_LOSS_GRACE = 10            # ~330 ms at 30 FPS (tolerate motion blur)

    try:
        while True:
            try:
                data = await ws.receive()
            except RuntimeError:
                break

            if data.get('type') == 'websocket.disconnect':
                break

            # All messages are JSON (features + actions)
            if 'text' not in data:
                continue

            msg = json.loads(data['text'])
            action = msg.get('action')
            msg_type = msg.get('type')

            if action == 'clear_sentence':
                current_signs = []
                completed_phrases = []
                completed_signs = []
                completed_scores = []
                await ws.send_json({
                    'type': 'sentence_update',
                    'sentence': [],
                    'translated': '',
                    'translated_score': 0,
                    'phrases': [],
                    'phrase_signs': [],
                    'phrase_scores': [],
                })
                continue
            elif action == 'set_threshold':
                threshold = float(msg.get('value', THRESHOLD))
                continue
            elif action == 'reload_index':
                eng.reload()
                await ws.send_json({'type': 'index_reloaded', 'loaded': eng.loaded})
                continue
            elif action == 'finalize_sentence':
                if current_signs:
                    await _do_finalize()
                continue

            if msg_type != 'features':
                continue

            raw_features = msg.get('features') or []
            if len(raw_features) != FRAME_FEATURE_DIM:
                continue

            features = np.asarray(raw_features, dtype='float32')
            hand_visible = bool(msg.get('hand_visible', False))

            now = time.time()

            if not hand_visible:
                if hand_missing_streak == 0:
                    # Hand just disappeared → stamp the start of the pause once
                    # and tell the segmenter to ignore the next motion delta.
                    last_hand_time = now
                    segmenter.notify_hand_lost()
                hand_missing_streak += 1

                # Finalize the current phrase once the pause is long enough.
                if current_signs:
                    time_since_hand = now - last_hand_time
                    if time_since_hand >= PAUSE_THRESHOLD:
                        await _do_finalize()

                # Grace period: wait a few blank frames before resetting the
                # segmenter (MediaPipe often drops the hand for 1–2 frames mid-sign).
                if hand_missing_streak >= HAND_LOSS_GRACE:
                    segmenter.reset()
                    await ws.send_json({
                        'type': 'status',
                        'hand_visible': False,
                        'is_signing': False,
                        'buffer_length': 0,
                        'motion_energy': 0.0,
                    })
                continue

            # Hand is visible — reset pause bookkeeping
            hand_missing_streak = 0

            seg_result = segmenter.process_features(features)

            # Rest-posture phrase boundary: when both wrists drop below the
            # shoulder line and motion is low, the signer is at rest → end
            # of phrase. Triggers only after we actually have some signs and
            # the segmenter isn't mid-sign.
            if (not seg_result['is_signing']
                    and _is_at_rest(features, seg_result['motion_energy'])):
                rest_streak += 1
            else:
                rest_streak = 0

            if current_signs and rest_streak >= REST_FRAMES:
                await _do_finalize()
                rest_streak = 0

            await ws.send_json({
                'type': 'status',
                'hand_visible': True,
                'is_signing': seg_result['is_signing'],
                'buffer_length': seg_result['buffer_length'],
                'motion_energy': round(seg_result['motion_energy'], 4),
            })

            # Intermediate prediction
            if seg_result['should_predict_intermediate'] and seg_result['current_buffer']:
                t0 = time.time()
                best_word, best_dist, top_k = eng.predict(seg_result['current_buffer'])
                inference_ms = int((time.time() - t0) * 1000)

                if best_word:
                    confidence = max(0, 1 - (best_dist / threshold)) if best_dist else 0
                    await ws.send_json({
                        'type': 'prediction',
                        'word': best_word,
                        'distance': round(best_dist, 1),
                        'threshold': threshold,
                        'confidence': round(confidence, 2),
                        'top_k': [{'word': w, 'distance': round(d, 1)} for d, w in top_k],
                        'inference_ms': inference_ms,
                        'is_final': False,
                    })

            # Final prediction (sign ended)
            if seg_result['sign_ended'] and seg_result['ended_buffer']:
                t0 = time.time()
                best_word, best_dist, top_k = eng.predict(seg_result['ended_buffer'])
                inference_ms = int((time.time() - t0) * 1000)

                # ---- Context rerank via the seq2seq translator (implicit LM) ----
                # Only rerank when the classifier is uncertain (prob < 0.7) and
                # we already have at least one prior sign. Rescored log-score:
                #     score = log(P_clf) + LM_LAMBDA * log(P_translator)
                LM_LAMBDA = 0.3
                CERTAINTY_CUTOFF = 0.7
                if (best_word and translator.loaded and current_signs
                        and len(top_k) >= 2 and np.exp(-best_dist) < CERTAINTY_CUTOFF):
                    rerank_t0 = time.time()
                    rescored = []
                    for cand_dist, cand_word in top_k[:3]:
                        clf_logp = -cand_dist
                        lm_logp = translator.score_signs(current_signs + [cand_word])
                        rescored.append((clf_logp + LM_LAMBDA * lm_logp, cand_dist, cand_word))
                    rescored.sort(key=lambda x: -x[0])
                    new_best = rescored[0]
                    if new_best[2] != best_word:
                        best_word = new_best[2]
                        best_dist = new_best[1]
                        inference_ms += int((time.time() - rerank_t0) * 1000)

                if best_word and best_dist is not None and best_dist < threshold:
                    if not current_signs or current_signs[-1] != best_word:
                        current_signs.append(best_word)

                        # Retranslate after every newly-added sign
                        if translator.loaded:
                            translated = translator.translate(current_signs)
                            translated_score = round(
                                translation_confidence(translator.score_signs(current_signs)), 2
                            )
                        else:
                            translated = ' '.join(current_signs)
                            translated_score = 1.0

                        await ws.send_json({
                            'type': 'sentence_update',
                            'sentence': list(current_signs),
                            'translated': translated,
                            'translated_score': translated_score,
                            'phrases': completed_phrases,
                            'phrase_signs': completed_signs,
                            'phrase_scores': completed_scores,
                        })

                if best_word:
                    confidence = max(0, 1 - (best_dist / threshold)) if best_dist else 0
                    await ws.send_json({
                        'type': 'prediction',
                        'word': best_word,
                        'distance': round(best_dist, 1),
                        'threshold': threshold,
                        'confidence': round(confidence, 2),
                        'top_k': [{'word': w, 'distance': round(d, 1)} for d, w in top_k],
                        'inference_ms': inference_ms,
                        'is_final': True,
                    })

    except WebSocketDisconnect:
        pass
