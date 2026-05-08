"""REST API endpoints for vocabulary, contributions, admin."""
import os
import time
import json
import queue
import threading
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from server.database import get_db, Word, Contribution, IndexBuild
from sqlalchemy.orm import Session
from ml import template_manager
from ml.template_manager import normalize_word
from server.routers.inference_ws import get_engine
from server.config import ADMIN_PASSWORD, TEMPLATE_DIR, REFERENCE_DATASET

router = APIRouter(prefix='/api')


# ============================================================
# SCHEMAS
# ============================================================
class WordCreate(BaseModel):
    name: str
    description: str = ''

class WordUpdate(BaseModel):
    description: Optional[str] = None
    is_active: Optional[bool] = None

class ContributionReview(BaseModel):
    status: str  # 'approved' or 'rejected'
    notes: str = ''
    reviewed_by: str = 'admin'

class AdminLogin(BaseModel):
    password: str


# ============================================================
# VOCABULARY
# ============================================================
@router.get('/vocabulary')
def list_vocabulary(db: Session = Depends(get_db)):
    """List all words with template counts."""
    # Sync from filesystem
    fs_words = template_manager.list_words()
    fs_names = {w['name'] for w in fs_words}

    # Ensure DB has all filesystem words
    for fw in fs_words:
        existing = db.query(Word).filter(Word.name == fw['name']).first()
        if existing:
            existing.template_count = fw['template_count']
        else:
            db.add(Word(name=fw['name'], template_count=fw['template_count']))
    db.commit()

    words = db.query(Word).order_by(Word.name).all()
    return [{
        'id': w.id,
        'name': w.name,
        'description': w.description or '',
        'template_count': w.template_count,
        'is_active': w.is_active,
        'created_at': w.created_at.isoformat() if w.created_at else None,
    } for w in words]


@router.get('/vocabulary/{word_name}')
def get_word_detail(word_name: str, db: Session = Depends(get_db)):
    word_name = normalize_word(word_name)
    word = db.query(Word).filter(Word.name == word_name).first()
    if not word:
        raise HTTPException(404, 'Word not found')

    templates = template_manager.list_templates(word_name)
    contributions = db.query(Contribution).filter(Contribution.word_id == word.id).all()

    # Check if reference video exists
    has_reference = False
    if os.path.isdir(REFERENCE_DATASET):
        ref_dir = os.path.join(REFERENCE_DATASET, word_name)
        has_reference = os.path.isdir(ref_dir) and any(
            f.endswith('.mp4') for f in os.listdir(ref_dir)
        )

    return {
        'word': {
            'id': word.id,
            'name': word.name,
            'description': word.description or '',
            'template_count': len(templates),
            'is_active': word.is_active,
        },
        'templates': templates,
        'has_reference': has_reference,
        'contributions': [{
            'id': c.id,
            'contributor': c.contributor,
            'status': c.status,
            'recorded_at': c.recorded_at.isoformat() if c.recorded_at else None,
        } for c in contributions],
    }


@router.post('/vocabulary')
def create_word(data: WordCreate, db: Session = Depends(get_db)):
    name = normalize_word(data.name)
    if not name:
        raise HTTPException(400, 'Word name is empty')
    existing = db.query(Word).filter(Word.name == name).first()
    if existing:
        raise HTTPException(400, 'Word already exists')
    word = Word(name=name, description=data.description)
    db.add(word)
    db.commit()
    os.makedirs(os.path.join(TEMPLATE_DIR, name), exist_ok=True)
    return {'id': word.id, 'name': word.name}


@router.patch('/vocabulary/{word_name}')
def update_word(word_name: str, data: WordUpdate, db: Session = Depends(get_db)):
    word = db.query(Word).filter(Word.name == normalize_word(word_name)).first()
    if not word:
        raise HTTPException(404, 'Word not found')
    if data.description is not None:
        word.description = data.description
    if data.is_active is not None:
        word.is_active = data.is_active
    db.commit()
    return {'ok': True}


@router.delete('/vocabulary/{word_name}')
def delete_word(word_name: str, db: Session = Depends(get_db)):
    word_name = normalize_word(word_name)
    word = db.query(Word).filter(Word.name == word_name).first()
    if not word:
        raise HTTPException(404, 'Word not found')
    template_manager.delete_word(word_name)
    db.delete(word)
    db.commit()
    return {'ok': True}


# ============================================================
# CONTRIBUTIONS
# ============================================================
@router.get('/contributions')
def list_contributions(
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    q = db.query(Contribution).join(Word)
    if status:
        q = q.filter(Contribution.status == status)
    contribs = q.order_by(Contribution.recorded_at.desc()).all()
    return [{
        'id': c.id,
        'word': c.word.name,
        'contributor': c.contributor,
        'status': c.status,
        'recorded_at': c.recorded_at.isoformat() if c.recorded_at else None,
        'reviewed_at': c.reviewed_at.isoformat() if c.reviewed_at else None,
        'notes': c.notes,
    } for c in contribs]


@router.patch('/contributions/{contribution_id}')
def review_contribution(
    contribution_id: int,
    review: ContributionReview,
    db: Session = Depends(get_db)
):
    contrib = db.query(Contribution).get(contribution_id)
    if not contrib:
        raise HTTPException(404, 'Contribution not found')

    if review.status == 'approved':
        # Move template from pending to templates/.  We use the
        # contribution's DB id as the destination filename so that
        # concurrent approvals can never overwrite each other (the
        # previous sequential-index scheme had a race condition that
        # silently lost templates — see template_manager.save_template
        # docstring).
        word = db.query(Word).get(contrib.word_id)
        if word and os.path.exists(contrib.file_path):
            new_path, idx = template_manager.approve_contribution(
                contrib.file_path, word.name, unique_id=contrib.id,
            )
            contrib.file_path = new_path
            word.template_count = len(template_manager.list_templates(word.name))

    elif review.status == 'rejected':
        # Delete pending features file
        if os.path.exists(contrib.file_path):
            os.remove(contrib.file_path)

    contrib.status = review.status
    contrib.notes = review.notes
    contrib.reviewed_by = review.reviewed_by
    contrib.reviewed_at = datetime.utcnow()
    db.commit()
    return {'ok': True, 'status': review.status}


@router.delete('/contributions/{contribution_id}')
def delete_contribution(contribution_id: int, db: Session = Depends(get_db)):
    contrib = db.query(Contribution).get(contribution_id)
    if not contrib:
        raise HTTPException(404)
    if os.path.exists(contrib.file_path):
        os.remove(contrib.file_path)
    db.delete(contrib)
    db.commit()
    return {'ok': True}


# ============================================================
# ADMIN
# ============================================================
@router.post('/admin/login')
def admin_login(data: AdminLogin):
    if data.password != ADMIN_PASSWORD:
        raise HTTPException(401, 'Wrong password')
    return {'ok': True, 'token': 'admin-session'}


class BuildIndexRequest(BaseModel):
    templates_dir: Optional[str] = None  # custom path; None = auto-detect V11 then V8
    epochs: int = 200


@router.post('/admin/build-index')
def build_index(req: BuildIndexRequest = BuildIndexRequest(), db: Session = Depends(get_db)):
    import time
    t0 = time.time()
    result = {'status': 'success'}

    # Train the V11 BiLSTM classifier on the available templates.
    from ml import lstm_trainer
    try:
        result['lstm'] = lstm_trainer.fit_and_save(
            templates_dir=req.templates_dir,
            epochs=req.epochs,
        )
    except Exception as e:
        result['lstm_error'] = str(e)
        result['status'] = 'failed'

    n_templates = (result.get('lstm') or {}).get('n_templates', 0)
    n_classes   = (result.get('lstm') or {}).get('n_classes', 0)
    build = IndexBuild(
        n_words=n_classes,
        n_templates=n_templates,
        summary_dim=0,
        status=result['status'],
        duration_ms=int((time.time() - t0) * 1000),
    )
    db.add(build)
    db.commit()

    # Hot-reload the live LSTM engine (picks up the newly saved model.keras).
    eng = get_engine()
    eng.reload()

    return result


@router.get('/admin/train-stream')
def train_stream(templates_dir: Optional[str] = None, epochs: int = 200):
    """SSE endpoint — streams BiLSTM training progress epoch by epoch."""
    q: queue.Queue = queue.Queue()

    def run():
        try:
            from ml import lstm_trainer
            meta = lstm_trainer.fit_and_save(
                templates_dir=templates_dir or None,
                epochs=epochs,
                progress_queue=q,
            )
            # Hot-reload engine after training
            eng = get_engine()
            eng.reload()
            q.put({'type': 'done', 'meta': meta})
        except Exception as e:
            q.put({'type': 'error', 'message': str(e)})

    threading.Thread(target=run, daemon=True).start()

    def event_stream():
        while True:
            try:
                item = q.get(timeout=2)
                yield f"data: {json.dumps(item)}\n\n"
                if item.get('type') in ('done', 'error'):
                    break
            except queue.Empty:
                yield ": keepalive\n\n"

    return StreamingResponse(
        event_stream(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@router.get('/admin/stats')
def get_stats(db: Session = Depends(get_db)):
    n_words = db.query(Word).count()
    n_templates = template_manager.count_total_templates()
    n_pending = db.query(Contribution).filter(Contribution.status == 'pending').count()
    n_approved = db.query(Contribution).filter(Contribution.status == 'approved').count()
    n_rejected = db.query(Contribution).filter(Contribution.status == 'rejected').count()

    last_build = db.query(IndexBuild).order_by(IndexBuild.built_at.desc()).first()

    eng = get_engine()

    return {
        'words': n_words,
        'templates': n_templates,
        'contributions': {
            'pending': n_pending,
            'approved': n_approved,
            'rejected': n_rejected,
        },
        'last_build': {
            'built_at': last_build.built_at.isoformat() if last_build else None,
            'n_templates': last_build.n_templates if last_build else 0,
            'duration_ms': last_build.duration_ms if last_build else 0,
            'status': last_build.status if last_build else 'never',
        },
        'engine_loaded': eng.loaded,
        'engine_words': len(eng.words) if eng.loaded else 0,
    }


# ============================================================
# TRANSLATION
# ============================================================
class TranslateRequest(BaseModel):
    signs: str  # space-separated sign words

@router.post('/translate')
def translate_signs(data: TranslateRequest):
    """Translate sign words to a French sentence."""
    from ml.translator import get_translator
    translator = get_translator()
    signs_list = data.signs.strip().split()
    if not signs_list:
        return {'signs': '', 'sentence': ''}
    sentence = translator.translate(signs_list)
    return {'signs': data.signs, 'sentence': sentence}


# ============================================================
# REFERENCE VIDEOS
# ============================================================
@router.get('/reference/{word_name}')
def get_reference_info(word_name: str):
    """Check if a reference video exists for this word in the SL dataset."""
    if not os.path.isdir(REFERENCE_DATASET):
        return {'exists': False}
    ref_dir = os.path.join(REFERENCE_DATASET, word_name)
    if not os.path.isdir(ref_dir):
        return {'exists': False}
    videos = sorted([f for f in os.listdir(ref_dir) if f.endswith('.mp4')])
    if not videos:
        return {'exists': False}
    return {'exists': True, 'count': len(videos), 'filename': videos[0]}


@router.get('/reference/{word_name}/video')
def serve_reference_video(word_name: str):
    """Serve the first reference video for a word."""
    if not os.path.isdir(REFERENCE_DATASET):
        raise HTTPException(404, 'Reference dataset not found')
    ref_dir = os.path.join(REFERENCE_DATASET, word_name)
    if not os.path.isdir(ref_dir):
        raise HTTPException(404, 'No reference for this word')
    videos = sorted([f for f in os.listdir(ref_dir) if f.endswith('.mp4')])
    if not videos:
        raise HTTPException(404, 'No video files')
    return FileResponse(
        os.path.join(ref_dir, videos[0]),
        media_type='video/mp4',
        filename=f'{word_name}.mp4'
    )


@router.get('/contributions/{contribution_id}/features')
def serve_contribution_features(contribution_id: int, db: Session = Depends(get_db)):
    """Serve the recorded features array for a contribution (for skeleton playback)."""
    import numpy as np
    contrib = db.query(Contribution).get(contribution_id)
    if not contrib:
        raise HTTPException(404, 'Contribution not found')
    if not contrib.file_path or not os.path.exists(contrib.file_path):
        raise HTTPException(404, 'Features file not found')
    arr = np.load(contrib.file_path)
    # Only keep the raw 171 per-frame features (ignore velocity if present)
    if arr.ndim == 2 and arr.shape[1] >= 171:
        arr = arr[:, :171]
    return {'frames': arr.tolist(), 'shape': list(arr.shape)}


@router.get('/dataset-words')
def list_dataset_words():
    """List all available words in the SL reference dataset."""
    if not os.path.isdir(REFERENCE_DATASET):
        return []
    words = sorted([d for d in os.listdir(REFERENCE_DATASET)
                    if os.path.isdir(os.path.join(REFERENCE_DATASET, d))])
    return words


@router.get('/health')
def health():
    return {'status': 'ok'}
