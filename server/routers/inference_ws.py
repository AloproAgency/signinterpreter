"""WebSocket endpoint for real-time inference — V11 (BiLSTM + sliding window).

MediaPipe runs client-side (browser). The client sends 171-float feature
vectors as JSON messages; the server classifies continuously via a sliding
30-frame window and translates committed sign sequences.

No explicit segmentation: the LSTM always sees a full 30-frame buffer, so
partial-sign errors are eliminated.  A sign is emitted only when the same
class is predicted stably for K_STABLE consecutive strides.
"""
import json
import os
import time
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ml.lstm_engine import LSTMEngine
from ml.sliding_window import SlidingWindowClassifier
from ml.translator import get_translator
from ml.features import FRAME_FEATURE_DIM

router = APIRouter()

engine = LSTMEngine()

PAUSE_THRESHOLD   = 0.8   # seconds without hand → finalize phrase
REST_WRIST_Y      = 0.8   # wrist y ≥ this (shoulder-normalised) → hands low
REST_MOTION_MAX   = 0.02
REST_FRAMES       = 8     # ~270 ms of rest posture → phrase boundary
REST_MIN_GAP      = 0.3   # s — don't finalize if a sign was just committed
IDLE_TIMEOUT      = 3.0   # s — absolute safety-net finalize

LW_Y_IDX, RW_Y_IDX = 9 * 3 + 1, 10 * 3 + 1


def _is_at_rest(features, motion_energy):
    lw_y = float(features[LW_Y_IDX])
    rw_y = float(features[RW_Y_IDX])
    return (lw_y >= REST_WRIST_Y) and (rw_y >= REST_WRIST_Y) and motion_energy < REST_MOTION_MAX


def get_engine():
    if not engine.loaded:
        engine.load()
    return engine


@router.websocket('/ws/inference')
async def inference_websocket(ws: WebSocket):
    await ws.accept()

    eng        = get_engine()
    classifier = SlidingWindowClassifier()
    classifier.set_engine(eng)
    translator = get_translator()

    current_signs:     list[str]   = []
    completed_phrases: list[str]   = []
    completed_signs:   list[list]  = []
    completed_scores:  list[float] = []

    last_hand_time  = time.time()
    last_sign_time  = time.time()
    hand_miss_count = 0
    rest_streak     = 0
    HAND_GRACE      = 25   # frames of missing hand tolerated before streak reset

    def _translation_confidence(logp: float) -> float:
        import math
        return max(0.0, min(1.0, math.exp(max(-3.0, logp))))

    async def push_sign(word: str):
        nonlocal last_sign_time
        if not word:
            return
        if current_signs and current_signs[-1] == word:
            return
        current_signs.append(word)
        last_sign_time = time.time()
        await ws.send_json({
            'type': 'sentence_update',
            'sentence': list(current_signs),
            'translated': '',
            'translated_score': 0,
            'phrases': completed_phrases,
            'phrase_signs': completed_signs,
            'phrase_scores': completed_scores,
        })

    async def _do_finalize():
        nonlocal current_signs
        if not current_signs:
            return
        signs = list(current_signs)
        if translator.loaded:
            phrase = translator.translate(signs)
            score  = _translation_confidence(translator.score_signs(signs))
        else:
            phrase, score = ' '.join(signs), 1.0
        completed_phrases.append(phrase)
        completed_signs.append(signs)
        completed_scores.append(round(score, 2))
        current_signs.clear()
        classifier.reset()
        await ws.send_json({
            'type': 'sentence_update',
            'sentence': [],
            'translated': '',
            'translated_score': 0,
            'phrases': completed_phrases,
            'phrase_signs': completed_signs,
            'phrase_scores': completed_scores,
        })

    try:
        while True:
            try:
                data = await ws.receive()
            except RuntimeError:
                break
            if data.get('type') == 'websocket.disconnect':
                break
            if 'text' not in data:
                continue

            msg    = json.loads(data['text'])
            action = msg.get('action')

            if action == 'clear_sentence':
                current_signs.clear()
                completed_phrases.clear()
                completed_signs.clear()
                completed_scores.clear()
                classifier.reset()
                await ws.send_json({
                    'type': 'sentence_update', 'sentence': [],
                    'translated': '', 'translated_score': 0,
                    'phrases': [], 'phrase_signs': [], 'phrase_scores': [],
                })
                continue
            elif action == 'finalize_sentence':
                if current_signs:
                    await _do_finalize()
                continue
            elif action == 'reload_index':
                eng.reload()
                classifier.set_engine(eng)
                classifier.reset()
                await ws.send_json({'type': 'index_reloaded', 'loaded': eng.loaded})
                continue

            if msg.get('type') != 'features':
                continue

            raw = msg.get('features') or []
            if len(raw) != FRAME_FEATURE_DIM:
                continue

            features     = np.asarray(raw, dtype='float32')
            hand_visible = bool(msg.get('hand_visible', False))
            now          = time.time()

            # Absolute idle safety-net
            if current_signs and (now - last_sign_time) >= IDLE_TIMEOUT:
                await _do_finalize()

            # ── hand not visible ──────────────────────────────────────
            if not hand_visible:
                if hand_miss_count == 0:
                    last_hand_time = now
                hand_miss_count += 1

                if current_signs and (now - last_hand_time) >= PAUSE_THRESHOLD:
                    await _do_finalize()

                if hand_miss_count >= HAND_GRACE:
                    classifier.reset_streak()   # keep buffer warm — only clear streak state
                    await ws.send_json({
                        'type': 'status', 'hand_visible': False,
                        'is_signing': False, 'buffer_length': len(classifier._buf),
                        'motion_energy': 0.0, 'streak_progress': 0.0,
                    })
                continue

            # ── hand visible ──────────────────────────────────────────
            hand_miss_count = 0

            word = classifier.process(features)
            if word:
                await push_sign(word)

            # Rest-posture phrase boundary
            if (not classifier.is_active
                    and _is_at_rest(features, classifier.motion_energy)):
                rest_streak += 1
            else:
                rest_streak = 0

            if (current_signs
                    and rest_streak >= REST_FRAMES
                    and (now - last_sign_time) >= REST_MIN_GAP):
                await _do_finalize()
                rest_streak = 0

            # Live prediction feedback for UI (even before emit)
            if classifier.last_word and classifier.last_dist < float('inf'):
                await ws.send_json({
                    'type': 'prediction',
                    'word': classifier.last_word,
                    'distance': round(classifier.last_dist, 3),
                    'confidence': round(max(0.0, 1.0 - classifier.last_dist / 0.55), 2),
                    'top_k': [{'word': w, 'distance': round(d, 3)} for d, w in classifier.last_top_k[:5]],
                    'is_final': False,
                    'streak_progress': round(classifier.streak_progress, 2),
                })

            await ws.send_json({
                'type': 'status',
                'hand_visible': True,
                'is_signing': classifier.is_active,
                'buffer_length': len(classifier._buf),
                'motion_energy': round(classifier.motion_energy, 4),
                'streak_progress': round(classifier.streak_progress, 2),
            })

    except WebSocketDisconnect:
        pass
