#!/usr/bin/env python3
"""
Train a Random Forest classifier on phonological descriptors extracted from
data/templates/<word>/*.npy.

Outputs:
  data/phono/rf.pkl
  data/phono/meta.json   (classes, scaler_mean, scaler_std)

Also reports:
  - 5-fold stratified CV accuracy
  - 80/20 holdout accuracy (confusion matrix)
  - Feature importance ranking
"""
import argparse
import json
import os
import pickle
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from ml.constants import TEMPLATE_DIR, SEQUENCE_LENGTH
from ml.features import both_hands_missing
from ml.phono_features import phonological_descriptor, PHONO_DIM, PHONO_FEATURE_NAMES
from ml.phono_engine import PHONO_DIR, MODEL_PATH, META_PATH


def load_descriptors(template_dir, max_both_missing=3):
    """Return (X (n, 34), y (n,), words)."""
    words = sorted(d for d in os.listdir(template_dir)
                   if os.path.isdir(os.path.join(template_dir, d)))
    X, y = [], []
    dropped = 0
    for cls_idx, word in enumerate(words):
        word_dir = os.path.join(template_dir, word)
        for fn in sorted(os.listdir(word_dir)):
            if not fn.endswith('.npy'):
                continue
            arr = np.load(os.path.join(word_dir, fn))
            if arr.ndim != 2 or arr.shape[0] != SEQUENCE_LENGTH:
                continue
            raw = arr[:, :171] if arr.shape[1] >= 171 else arr
            if both_hands_missing(raw) > max_both_missing:
                dropped += 1
                continue
            desc = phonological_descriptor(raw)
            X.append(desc)
            y.append(cls_idx)
    X = np.stack(X).astype('float32')
    y = np.array(y, dtype='int32')
    print(f'Loaded {len(X)} descriptors across {len(words)} classes (dropped {dropped} low-quality)')
    return X, y, words


def standardize(X_train, X_test):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--templates', default=TEMPLATE_DIR)
    ap.add_argument('--n-estimators', type=int, default=300)
    ap.add_argument('--max-depth', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--test-size', type=float, default=0.2)
    args = ap.parse_args()

    np.random.seed(args.seed)

    X, y, words = load_descriptors(args.templates)
    per_class = np.bincount(y, minlength=len(words))
    print('Per-class counts:', dict(zip(words, per_class.tolist())))

    # ---- 5-fold stratified CV on the full set -----------------------------
    print('\n=== 5-fold CV ===')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    accs = []
    for i, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte, mu, sd = standardize(X[tr], X[te])
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            class_weight='balanced', random_state=args.seed, n_jobs=-1,
        )
        clf.fit(Xtr, y[tr])
        acc = clf.score(Xte, y[te])
        accs.append(acc)
        print(f'  fold {i+1}  acc={acc:.3f}')
    print(f'  mean CV accuracy: {np.mean(accs):.3f}  (± {np.std(accs):.3f})')

    # ---- 80/20 holdout for confusion matrix -------------------------------
    print('\n=== 80/20 holdout ===')
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed,
    )
    Xtr, Xte, mu, sd = standardize(Xtr, Xte)
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        class_weight='balanced', random_state=args.seed, n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    acc = (yp == yte).mean()
    print(f'  holdout accuracy: {acc:.3f}  ({(yp==yte).sum()}/{len(yte)})')

    print('\nConfusion matrix (rows = truth, cols = predicted):')
    cm = confusion_matrix(yte, yp, labels=list(range(len(words))))
    print('            ' + ''.join(f'{w[:6]:>7s}' for w in words))
    for i, w in enumerate(words):
        row = f'  {w:10s}' + ''.join(f'{cm[i][j]:7d}' for j in range(len(words)))
        print(row)

    print('\nClassification report:')
    print(classification_report(yte, yp, target_names=words, zero_division=0))

    # ---- Feature importance -----------------------------------------------
    print('Top 15 feature importances:')
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1][:15]
    for idx in order:
        print(f'  {PHONO_FEATURE_NAMES[idx]:25s}  {importances[idx]:.3f}')

    # ---- Final fit on all data + save ------------------------------------
    print('\n=== Training final model on all data ===')
    X_all, _, mu_all, sd_all = standardize(X, X)  # use all as "train"
    final = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        class_weight='balanced', random_state=args.seed, n_jobs=-1,
    )
    final.fit(X_all, y)
    os.makedirs(PHONO_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final, f)
    with open(META_PATH, 'w') as f:
        json.dump({
            'classes': words,
            'scaler_mean': mu_all.tolist(),
            'scaler_std': sd_all.tolist(),
            'n_estimators': args.n_estimators,
            'holdout_accuracy': float(acc),
            'cv_accuracy_mean': float(np.mean(accs)),
            'cv_accuracy_std': float(np.std(accs)),
        }, f, ensure_ascii=False, indent=2)
    print(f'  saved {MODEL_PATH}')
    print(f'  saved {META_PATH}')


if __name__ == '__main__':
    main()
