"""WebSocket endpoint for template recording (contributions)."""
import cv2
import json
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ml.mediapipe_service import MediaPipeService
from ml.template_manager import save_contribution, list_templates
from server.database import SessionLocal, Word, Contribution
from ml.constants import SEQUENCE_LENGTH

router = APIRouter()


def extract_landmark_points(results, frame_w, frame_h):
    """Extract landmark pixel coordinates for client-side drawing."""
    landmarks = {'pose': [], 'left_hand': [], 'right_hand': []}
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks['pose'].append({
                'x': round(lm.x * frame_w), 'y': round(lm.y * frame_h), 'v': round(lm.visibility, 2),
            })
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks['left_hand'].append({'x': round(lm.x * frame_w), 'y': round(lm.y * frame_h)})
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks['right_hand'].append({'x': round(lm.x * frame_w), 'y': round(lm.y * frame_h)})
    return landmarks


@router.websocket('/ws/collect')
async def collection_websocket(ws: WebSocket):
    await ws.accept()

    mp_service = MediaPipeService()
    recording = False
    features_buffer = []
    current_word = None
    contributor = 'anonymous'

    try:
        while True:
            try:
                data = await ws.receive()
            except RuntimeError:
                break

            if data.get('type') == 'websocket.disconnect':
                break

            if 'text' in data:
                msg = json.loads(data['text'])
                action = msg.get('action')

                if action == 'start':
                    current_word = msg.get('word', '').strip()
                    contributor = msg.get('contributor', 'anonymous').strip()
                    if not current_word:
                        await ws.send_json({'type': 'error', 'message': 'Word is required'})
                        continue
                    existing = list_templates(current_word)
                    recording = True
                    features_buffer = []
                    await ws.send_json({
                        'type': 'ready',
                        'word': current_word,
                        'existing_count': len(existing),
                    })

                elif action == 'stop':
                    if recording and len(features_buffer) >= 4:
                        # Save features template
                        template = np.array(features_buffer[:SEQUENCE_LENGTH], dtype='float32')
                        if len(template) < SEQUENCE_LENGTH:
                            pad = np.tile(template[-1:], (SEQUENCE_LENGTH - len(template), 1))
                            template = np.concatenate([template, pad])

                        db = SessionLocal()
                        try:
                            word_obj = db.query(Word).filter(Word.name == current_word).first()
                            if not word_obj:
                                word_obj = Word(name=current_word)
                                db.add(word_obj)
                                db.commit()
                                db.refresh(word_obj)

                            contrib = Contribution(
                                word_id=word_obj.id,
                                contributor=contributor,
                                file_path='',
                            )
                            db.add(contrib)
                            db.commit()
                            db.refresh(contrib)

                            # Save features as .npy
                            features_path = save_contribution(contrib.id, template)
                            contrib.file_path = features_path

                            db.commit()

                            await ws.send_json({
                                'type': 'saved',
                                'contribution_id': contrib.id,
                                'word': current_word,
                                'frames_recorded': len(features_buffer),
                                'status': 'pending',
                            })
                        finally:
                            db.close()
                    else:
                        await ws.send_json({
                            'type': 'error',
                            'message': f'Not enough frames ({len(features_buffer)})',
                        })

                    recording = False
                    features_buffer = []

                elif action == 'cancel':
                    recording = False
                    features_buffer = []
                    await ws.send_json({'type': 'cancelled'})

                continue

            if 'bytes' not in data:
                continue

            frame_bytes = data['bytes']
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            features, results, hand_visible = mp_service.process_frame(frame)
            landmarks = extract_landmark_points(results, w, h)

            if recording:
                features_buffer.append(features)

                await ws.send_json({
                    'type': 'recording',
                    'frame_count': len(features_buffer),
                    'total': SEQUENCE_LENGTH,
                    'hand_visible': hand_visible,
                    'landmarks': landmarks,
                })

                if len(features_buffer) >= SEQUENCE_LENGTH:
                    await ws.send_json({
                        'type': 'buffer_full',
                        'message': 'Buffer full, send stop to save',
                    })
            else:
                await ws.send_json({
                    'type': 'preview',
                    'hand_visible': hand_visible,
                    'landmarks': landmarks,
                })

    except WebSocketDisconnect:
        pass
    finally:
        mp_service.close()
