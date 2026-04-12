"""WebSocket endpoint for real-time inference.

MediaPipe runs client-side (browser). The client sends 171-float feature vectors
as JSON messages; the server only handles segmentation, kNN lookup, and translation.
"""
import json
import time
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ml.inference_engine import InferenceEngine, SignSegmenter
from ml.translator import get_translator
from ml.constants import THRESHOLD
from ml.features import FRAME_FEATURE_DIM

router = APIRouter()

engine = InferenceEngine()

PAUSE_THRESHOLD = 2.0  # seconds without hand = new sentence


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
    current_signs = []        # signs for the current phrase being built
    completed_phrases = []    # list of translated phrases (finalized)
    threshold = THRESHOLD
    last_hand_time = time.time()
    hand_was_visible = False

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
                await ws.send_json({
                    'type': 'sentence_update',
                    'sentence': [],
                    'translated': '',
                    'phrases': [],
                })
                continue
            elif action == 'set_threshold':
                threshold = float(msg.get('value', THRESHOLD))
                continue
            elif action == 'reload_index':
                eng.reload()
                await ws.send_json({'type': 'index_reloaded', 'loaded': eng.loaded})
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
                # Check if we should finalize current phrase (pause detected)
                if hand_was_visible and current_signs:
                    time_since_hand = now - last_hand_time
                    if time_since_hand >= PAUSE_THRESHOLD:
                        # Finalize current phrase
                        if translator.loaded:
                            translated = translator.translate(current_signs)
                        else:
                            translated = ' '.join(current_signs)
                        completed_phrases.append(translated)
                        current_signs = []

                        await ws.send_json({
                            'type': 'sentence_update',
                            'sentence': [],
                            'translated': '',
                            'phrases': completed_phrases,
                        })

                if hand_was_visible and not hand_visible:
                    last_hand_time = now

                hand_was_visible = False
                segmenter.reset()
                await ws.send_json({
                    'type': 'status',
                    'hand_visible': False,
                    'is_signing': False,
                    'buffer_length': 0,
                    'motion_energy': 0.0,
                })
                continue

            # Hand is visible
            hand_was_visible = True
            last_hand_time = now

            seg_result = segmenter.process_features(features)

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

                if best_word and best_dist is not None and best_dist < threshold:
                    if not current_signs or current_signs[-1] != best_word:
                        current_signs.append(best_word)

                        # Translate current phrase in real-time
                        if translator.loaded and len(current_signs) >= 1:
                            translated = translator.translate(current_signs)
                        else:
                            translated = ' '.join(current_signs)

                        await ws.send_json({
                            'type': 'sentence_update',
                            'sentence': list(current_signs),
                            'translated': translated,
                            'phrases': completed_phrases,
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
