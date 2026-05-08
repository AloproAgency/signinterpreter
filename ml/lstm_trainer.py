"""Train the V11 BiLSTM+attention classifier on template sequences.

Called from  server/routers/api.py  (POST /api/admin/build-index)  and
directly via  scripts/train_lstm.py.

Templates must be  (30, 171)  float32  .npy  files, one per sample,
organised in  <templates_dir>/<word>/<id>.npy  folders (same layout as V8).

The script searches for templates in:
  1. <V11>/data/templates/  (V11's own contributions, if any)
  2. <V11>/../V8/data/templates/  (V8 corpus, fallback)
A custom path can be supplied via the  templates_dir  argument.

Outputs (written to  <V11>/data/lstm/):
  model.keras    — trained Keras model
  classes.json   — ordered list of class names
  meta.json      — training statistics
"""
from __future__ import annotations

import json
import os
import numpy as np

_ROOT            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_TMPL    = os.path.join(_ROOT, 'data', 'templates')
_V8_TMPL         = os.path.abspath(os.path.join(_ROOT, '..', 'V8', 'data', 'templates'))
OUT_DIR          = os.path.join(_ROOT, 'data', 'lstm')
SEQUENCE_LENGTH  = 30
FEATURE_DIM      = 171


# ─────────────────────────────────────────── data loading ────────────────────

def _load_templates(templates_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, classes).  X.shape = (N, 30, 171), y.shape = (N,)."""
    classes = sorted(
        d for d in os.listdir(templates_dir)
        if os.path.isdir(os.path.join(templates_dir, d))
    )
    if not classes:
        raise ValueError(f'No word sub-directories found in {templates_dir}')

    idx_of = {c: i for i, c in enumerate(classes)}
    X, y = [], []
    for cls in classes:
        cls_dir = os.path.join(templates_dir, cls)
        for fn in os.listdir(cls_dir):
            if not fn.endswith('.npy'):
                continue
            arr = np.load(os.path.join(cls_dir, fn)).astype('float32')
            if arr.ndim == 2 and arr.shape[0] == SEQUENCE_LENGTH and arr.shape[1] >= FEATURE_DIM:
                X.append(arr[:, :FEATURE_DIM])
                y.append(idx_of[cls])
            else:
                print(f'  [skip] {cls}/{fn}: unexpected shape {arr.shape}')

    if not X:
        raise ValueError('No valid (30, 171) templates found.')

    X = np.stack(X).astype('float32')
    y = np.array(y, dtype='int32')

    print(f'Loaded {len(X)} templates / {len(classes)} classes from {templates_dir}')
    for cls in classes:
        n = int((y == idx_of[cls]).sum())
        print(f'  {cls:<25s} {n:3d}')
    return X, y, classes


# ──────────────────────────────────── augmentation ───────────────────────────

def _resample(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) == n:
        return arr
    t     = np.linspace(0, len(arr) - 1, n)
    left  = np.floor(t).astype(int)
    right = np.minimum(left + 1, len(arr) - 1)
    alpha = (t - left)[:, None]
    return (arr[left] * (1 - alpha) + arr[right] * alpha).astype('float32')


def _augment(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random augmentation of a (30, 171) template sequence."""
    s = seq.copy()

    # Gaussian noise
    if rng.random() < 0.80:
        s += rng.normal(0, 0.008, s.shape).astype('float32')

    # Amplitude scaling
    if rng.random() < 0.70:
        s *= float(rng.uniform(0.88, 1.12))

    # Temporal jitter — resample to ±15 % length then back to 30
    if rng.random() < 0.60:
        nl = max(10, int(SEQUENCE_LENGTH * float(rng.uniform(0.85, 1.15))))
        s  = _resample(s, nl)
        s  = _resample(s, SEQUENCE_LENGTH)

    # Hand-dominance flip (swap LH ↔ RH blocks, mirror pose x)
    if rng.random() < 0.50:
        f                 = s.copy()
        f[:, 39:102]      = s[:, 105:168]   # LH shape ← RH shape
        f[:, 105:168]     = s[:, 39:102]
        f[:, 102:105]     = s[:, 168:171]   # LH palm ← RH palm
        f[:, 168:171]     = s[:, 102:105]
        f[:, 0:39:3]      = -s[:, 0:39:3]  # mirror pose x-coords
        s = f

    # Frame blend — smooth random frames between neighbours
    if rng.random() < 0.40:
        n_blend = int(rng.integers(1, 3))
        for i in rng.choice(SEQUENCE_LENGTH - 2, n_blend, replace=False) + 1:
            alpha = float(rng.uniform(0.3, 0.7))
            s[i]  = alpha * s[i - 1] + (1 - alpha) * s[i + 1]

    return s.astype('float32')


def _build_augmented(X: np.ndarray, y: np.ndarray, multiplier: int):
    rng  = np.random.default_rng(42)
    Xa   = [X]
    ya   = [y]
    for _ in range(multiplier - 1):
        Xa.append(np.stack([_augment(X[i], rng) for i in range(len(X))]))
        ya.append(y)
    Xa = np.concatenate(Xa, axis=0)
    ya = np.concatenate(ya, axis=0)
    perm = rng.permutation(len(Xa))
    return Xa[perm], ya[perm]


# ────────────────────────────────────────── model ────────────────────────────

def _build_model(n_classes: int):
    """BiLSTM + temporal self-attention → softmax.

    Architecture
    ────────────
    Input  (30, 171)
      │
      ▼
    Bidirectional LSTM(64, return_sequences=True)  +  Dropout(0.50)
      │
      ▼   single-head additive attention
    score  = Dense(1, tanh)(h)           (30, 1)   — frame importance
    weight = softmax(score, axis=time)   (30, 1)
    ctx    = Σ weight × h                (128,)    — attended context
      │
      ▼
    Dropout(0.40)
      │
      ▼
    Dense(n_classes, softmax)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    reg = regularizers.l2(1e-4)
    inp = keras.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM), name='sequence')

    h = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, kernel_regularizer=reg),
        name='bilstm',
    )(inp)
    h = layers.Dropout(0.50, name='drop1')(h)

    # Temporal attention — one scalar score per frame, then softmax
    scores  = layers.Dense(1, activation='tanh', use_bias=False, name='attn_score')(h)
    # softmax over the time axis (axis=1) using Permute trick for portability
    scores  = layers.Permute((2, 1))(scores)               # (B, 1, T)
    weights = layers.Activation('softmax', name='attn_softmax')(scores)
    weights = layers.Permute((2, 1))(weights)              # (B, T, 1)
    ctx     = layers.Multiply(name='attn_weight')([h, weights])
    # GlobalAveragePooling1D ≡ sum/T — scale absorbed by the Dense layer.
    # Avoids Lambda which is not safely serializable in Keras 3.
    ctx     = layers.GlobalAveragePooling1D(name='attn_ctx')(ctx)  # (B, 128)

    ctx = layers.Dropout(0.40, name='drop2')(ctx)
    out = layers.Dense(n_classes, activation='softmax', name='logits')(ctx)

    model = keras.Model(inp, out, name='bilstm_attn')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ───────────────────────────────────── training entry point ──────────────────

def fit_and_save(templates_dir: str | None = None, epochs: int = 200,
                 progress_queue=None) -> dict:
    """Train and save the model.  Returns a summary dict for the API response."""
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split

    # ── locate templates ──────────────────────────────────────────────────────
    if templates_dir is None:
        if os.path.isdir(_DEFAULT_TMPL) and os.listdir(_DEFAULT_TMPL):
            templates_dir = _DEFAULT_TMPL
        elif os.path.isdir(_V8_TMPL):
            print(f'[lstm_trainer] Falling back to V8 templates: {_V8_TMPL}')
            templates_dir = _V8_TMPL
        else:
            raise FileNotFoundError(
                'No templates found.  Pass --templates-dir or populate '
                f'{_DEFAULT_TMPL}.'
            )

    X, y, classes = _load_templates(templates_dir)

    # ── augment ───────────────────────────────────────────────────────────────
    multiplier = max(10, 300 // max(len(X), 1))
    print(f'Augmenting ×{multiplier} → {len(X) * multiplier} samples')
    X_aug, y_aug = _build_augmented(X, y, multiplier)

    # ── train / val split (stratified) ────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_aug, y_aug, test_size=0.15, random_state=42, stratify=y_aug,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = _build_model(n_classes=len(classes))
    model.summary()

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt = os.path.join(OUT_DIR, 'model.keras')

    class _ProgressCallback(keras.callbacks.Callback):
        def __init__(self, q, total):
            self._q = q
            self._total = total
        def on_epoch_end(self, epoch, logs=None):
            if self._q is None:
                return
            self._q.put({
                'type': 'epoch',
                'epoch': epoch + 1,
                'total': self._total,
                'loss': round(float(logs.get('loss', 0)), 4),
                'acc': round(float(logs.get('accuracy', 0)), 4),
                'val_loss': round(float(logs.get('val_loss', 0)), 4),
                'val_acc': round(float(logs.get('val_accuracy', 0)), 4),
            })

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt, save_best_only=True, monitor='val_accuracy', verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            patience=25, monitor='val_accuracy', restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=12, min_lr=1e-5, verbose=0,
        ),
        _ProgressCallback(progress_queue, epochs),
    ]

    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=min(32, max(8, len(X_tr) // 8)),
        callbacks=callbacks,
        verbose=2,
    )

    # ── evaluate on original (un-augmented) templates ─────────────────────────
    best_model  = keras.models.load_model(ckpt, safe_mode=False)
    orig_preds  = np.argmax(best_model.predict(X, verbose=0), axis=1)
    orig_acc    = float((orig_preds == y).mean())
    best_val    = float(max(hist.history['val_accuracy']))

    # ── save class list & metadata ────────────────────────────────────────────
    with open(os.path.join(OUT_DIR, 'classes.json'), 'w', encoding='utf-8') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    meta = {
        'n_classes':          len(classes),
        'n_templates':        int(len(X)),
        'n_aug_samples':      int(len(X_aug)),
        'best_val_accuracy':  round(best_val, 4),
        'orig_accuracy':      round(orig_acc, 4),
        'templates_dir':      templates_dir,
    }
    with open(os.path.join(OUT_DIR, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('\n=== Training complete ===')
    print(f'  classes:           {len(classes)}')
    print(f'  templates:         {len(X)}')
    print(f'  val accuracy:      {best_val:.1%}')
    print(f'  orig accuracy:     {orig_acc:.1%}')
    print(f'  model saved:       {ckpt}')
    return meta
