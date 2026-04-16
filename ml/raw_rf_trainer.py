"""
Train the flat-joint Random Forest from templates. Called by the admin
"Rebuild Index" endpoint (same cadence as the Phono RF).
"""
import json
import os
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from ml.constants import TEMPLATE_DIR, SEQUENCE_LENGTH
from ml.features import (
    FRAME_FEATURE_DIM, both_hands_missing, interpolate_holes,
    to_right_dominant, add_wrist_velocity,
)
from ml.raw_rf_engine import RF_DIR, MODEL_PATH, META_PATH


def _vectorise(raw):
    raw = raw[:, :FRAME_FEATURE_DIM] if raw.shape[1] >= FRAME_FEATURE_DIM else raw
    raw = interpolate_holes(raw)
    raw = to_right_dominant(raw)
    return add_wrist_velocity(raw).flatten()


def fit_and_save(n_estimators=400, max_depth=None, seed=42, max_both_missing=3):
    classes = sorted(d for d in os.listdir(TEMPLATE_DIR)
                     if os.path.isdir(os.path.join(TEMPLATE_DIR, d)))
    X, y = [], []
    dropped = 0
    for ci, w in enumerate(classes):
        wd = os.path.join(TEMPLATE_DIR, w)
        for fn in sorted(os.listdir(wd)):
            if not fn.endswith('.npy'):
                continue
            try:
                arr = np.load(os.path.join(wd, fn))
            except Exception:
                dropped += 1
                continue
            if arr.ndim != 2 or arr.shape[0] != SEQUENCE_LENGTH:
                dropped += 1
                continue
            raw = arr[:, :FRAME_FEATURE_DIM] if arr.shape[1] >= FRAME_FEATURE_DIM else arr
            if both_hands_missing(raw) > max_both_missing:
                dropped += 1
                continue
            X.append(_vectorise(raw))
            y.append(ci)
    if not X:
        raise RuntimeError('No usable templates for raw-RF.')
    X = np.stack(X).astype('float32')
    y = np.array(y, dtype='int32')

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xn = (X - mu) / sd

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight='balanced', random_state=seed, n_jobs=-1,
    )
    clf.fit(Xn, y)

    os.makedirs(RF_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(META_PATH, 'w') as f:
        json.dump({
            'classes': classes,
            'mu': mu.tolist(),
            'sd': sd.tolist(),
            'n_estimators': n_estimators,
            'n_templates': int(len(X)),
            'dropped': int(dropped),
        }, f, ensure_ascii=False, indent=2)

    return {
        'n_classes': len(classes),
        'n_templates': int(len(X)),
        'dropped': int(dropped),
        'classes': classes,
    }
