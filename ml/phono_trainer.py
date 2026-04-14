"""
Programmatic entry-point for training the Phono Random Forest.
Called by scripts/train_phono.py (verbose / with CV) AND by the admin
"rebuild" endpoint (quick / silent).
"""
import json
import os
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from ml.constants import TEMPLATE_DIR, SEQUENCE_LENGTH
from ml.features import both_hands_missing, FRAME_FEATURE_DIM
from ml.phono_features import phonological_descriptor
from ml.phono_engine import PHONO_DIR, MODEL_PATH, META_PATH


def load_descriptors(template_dir=TEMPLATE_DIR, max_both_missing=3):
    """Return (X (n, 34), y (n,), classes). Quality-filtered."""
    classes = sorted(d for d in os.listdir(template_dir)
                     if os.path.isdir(os.path.join(template_dir, d)))
    X, y = [], []
    dropped = 0
    for ci, word in enumerate(classes):
        word_dir = os.path.join(template_dir, word)
        for fn in sorted(os.listdir(word_dir)):
            if not fn.endswith('.npy'):
                continue
            try:
                arr = np.load(os.path.join(word_dir, fn))
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
            X.append(phonological_descriptor(raw))
            y.append(ci)
    if not X:
        raise RuntimeError('No usable templates to train on.')
    return np.stack(X).astype('float32'), np.array(y, dtype='int32'), classes, dropped


def fit_and_save(n_estimators=300, max_depth=None, seed=42):
    """Fit the RF on the full templates set and persist model + metadata."""
    X, y, classes, dropped = load_descriptors()

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xn = (X - mu) / sd

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(Xn, y)

    os.makedirs(PHONO_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(META_PATH, 'w') as f:
        json.dump({
            'classes': classes,
            'scaler_mean': mu.tolist(),
            'scaler_std': sd.tolist(),
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
