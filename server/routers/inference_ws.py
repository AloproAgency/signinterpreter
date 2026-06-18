"""WebSocket endpoint for real-time inference — V11 (CTC + sliding window fallback).

MediaPipe runs client-side (browser). The client sends 171-float feature
vectors as JSON messages.

Sign recognition strategy
─────────────────────────
• CTC model (primary): accumulates all frames since the last phrase boundary
  and re-decodes the growing buffer every CTC_STRIDE frames.  `current_signs`
  is updated in-place — newly detected signs appear progressively.
  A final CTC pass runs at every phrase boundary to emit any trailing signs.

• SlidingWindowClassifier (fallback / UI feedback): still runs every frame to
  provide live prediction, motion energy, and streak_progress for the UI.
  Its sign-emit return value is used only when CTC is not loaded.
"""
import json
import os
import time
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ml.lstm_engine import LSTMEngine
from ml.sliding_window import SlidingWindowClassifier
from ml.ctc_model import CTCEngine
from ml.translator import get_translator
from ml.features import FRAME_FEATURE_DIM

router = APIRouter()

engine     = LSTMEngine()
ctc_engine = CTCEngine()

PAUSE_THRESHOLD   = 0.8
REST_WRIST_Y      = 0.8
REST_MOTION_MAX   = 0.02
REST_FRAMES       = 8
REST_MIN_GAP      = 0.3
IDLE_TIMEOUT      = 3.0

CTC_STRIDE     = 20   # run CTC every N frames (~0.67 s at 30 fps)
CTC_MIN_FRAMES = 30   # minimum buffer before first CTC run

LW_Y_IDX, RW_Y_IDX = 9 * 3 + 1, 10 * 3 + 1


def _is_at_rest(features, motion_energy):
    lw_y = float(features[LW_Y_IDX])
    rw_y = float(features[RW_Y_IDX])
    return (lw_y >= REST_WRIST_Y) and (rw_y >= REST_WRIST_Y) and motion_energy < REST_MOTION_MAX


def get_engine():
    if not engine.loaded:
        engine.load()
    return engine


def get_ctc_engine():
    if not ctc_engine.loaded:
        ctc_engine.load()
    return ctc_engine


@router.websocket('/ws/inference')
async def inference_websocket(ws: WebSocket):
    await ws.accept()

    eng        = get_engine()
    ctc_eng    = get_ctc_engine()
    classifier = SlidingWindowClassifier()
    classifier.set_engine(eng)
    translator = get_translator()

    use_ctc = ctc_eng.loaded
    print(f'[WS] CTC model {"ACTIVE" if use_ctc else "not available — sliding window mode"}')
    await ws.send_json({'type': 'model_info', 'ctc': use_ctc})

    current_signs:     list[str]   = []
    completed_phrases: list[str]   = []
    completed_signs:   list[list]  = []
    completed_scores:  list[float] = []

    # CTC state (per phrase)
    ctc_buf:          list         = []
    ctc_stride_count: int          = 0

    last_hand_time  = time.time()
    last_sign_time  = time.time()
    hand_miss_count = 0
    rest_streak     = 0
    HAND_GRACE      = 25

    def _translation_confidence(logp: float) -> float:
        import math
        return max(0.0, min(1.0, math.exp(max(-3.0, logp))))

    async def _send_sentence():
        await ws.send_json({
            'type': 'sentence_update',
            'sentence': list(current_signs),
            'translated': '',
            'translated_score': 0,
            'phrases': completed_phrases,
            'phrase_signs': completed_signs,
            'phrase_scores': completed_scores,
        })

    async def push_sign(word: str):
        nonlocal last_sign_time
        if not word:
            return
        if current_signs and current_signs[-1] == word:
            return
        current_signs.append(word)
        last_sign_time = time.time()
        await _send_sentence()

    async def _run_ctc_update():
        """Re-decode the growing buffer; add any newly detected signs."""
        nonlocal last_sign_time
        if len(ctc_buf) < CTC_MIN_FRAMES:
            return
        frames  = np.stack(ctc_buf)
        decoded = ctc_eng.predict_sequence(frames)

        # Déduplique les signes consécutifs identiques dans le décodage
        deduped = [s for i, s in enumerate(decoded) if i == 0 or s != decoded[i-1]]
        if len(deduped) > len(current_signs):
            for sign in deduped[len(current_signs):]:
                if not current_signs or current_signs[-1] != sign:
                    current_signs.append(sign)
                    last_sign_time = time.time()
            await _send_sentence()

    async def _do_finalize():
        nonlocal current_signs, ctc_buf, ctc_stride_count, rest_streak, last_sign_time
        if not current_signs and not ctc_buf:
            return
        rest_streak    = 0
        last_sign_time = time.time()

        # Final CTC pass — only if nothing was caught during streaming
        if use_ctc and ctc_buf and not current_signs:
            frames  = np.stack(ctc_buf)
            decoded = ctc_eng.predict_sequence(frames)
            if decoded:
                current_signs = decoded

        signs = list(current_signs)
        if not signs:
            ctc_buf.clear()
            ctc_stride_count = 0
            return

        from ml.translator import rule_translate
        phrase = rule_translate(signs)
        score  = _translation_confidence(translator.score_signs(signs)) if translator.loaded else 1.0

        completed_phrases.append(phrase)
        completed_signs.append(signs)
        completed_scores.append(round(score, 2))
        current_signs = []
        ctc_buf.clear()
        ctc_stride_count = 0
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
                current_signs = []
                ctc_buf.clear()
                ctc_stride_count = 0
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
                if current_signs or ctc_buf:
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

            if current_signs and (now - last_sign_time) >= IDLE_TIMEOUT:
                await _do_finalize()

            # ── hand not visible ──────────────────────────────────────
            if not hand_visible:
                if hand_miss_count == 0:
                    last_hand_time = now
                hand_miss_count += 1

                if (current_signs or ctc_buf) and (now - last_hand_time) >= PAUSE_THRESHOLD:
                    await _do_finalize()

                if hand_miss_count >= HAND_GRACE:
                    classifier.reset_streak()
                    await ws.send_json({
                        'type': 'status', 'hand_visible': False,
                        'is_signing': False, 'buffer_length': len(ctc_buf if use_ctc else classifier._buf),
                        'motion_energy': 0.0, 'streak_progress': 0.0,
                    })
                continue

            # ── hand visible ──────────────────────────────────────────
            hand_miss_count = 0

            # Sliding window always runs for motion energy + UI prediction
            sw_word = classifier.process(features)

            if use_ctc:
                # ── CTC mode ──────────────────────────────────────────
                ctc_buf.append(features)
                ctc_stride_count += 1
                if ctc_stride_count >= CTC_STRIDE:
                    ctc_stride_count = 0
                    await _run_ctc_update()
            else:
                # ── Fallback: sliding window emits signs ───────────────
                if sw_word:
                    await push_sign(sw_word)

            # Rest-posture phrase boundary
            if not classifier.is_active and _is_at_rest(features, classifier.motion_energy):
                rest_streak += 1
            else:
                rest_streak = 0

            if ((current_signs or ctc_buf)
                    and rest_streak >= REST_FRAMES
                    and (now - last_sign_time) >= REST_MIN_GAP):
                await _do_finalize()
                rest_streak = 0

            # Live prediction feedback
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

            buf_len = len(ctc_buf) if use_ctc else len(classifier._buf)
            await ws.send_json({
                'type': 'status',
                'hand_visible': True,
                'is_signing': classifier.is_active,
                'buffer_length': buf_len,
                'motion_energy': round(classifier.motion_energy, 4),
                'streak_progress': round(classifier.streak_progress, 2),
            })

    except WebSocketDisconnect:
        pass
