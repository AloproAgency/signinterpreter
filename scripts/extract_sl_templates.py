#!/usr/bin/env python3
"""
Extract MediaPipe features from SL reference videos and save them as .npy
templates in V11/data/templates/<word>/, ready for lstm_trainer.py.

Pipeline per video:
  video → MediaPipe → SignSegmenter → resample to (30, 171) → .npy

Only words already in --words-filter (or all words if omitted) are processed.
Existing templates are left untouched; this script only appends new sl_XXXX.npy.

Usage:
  python3 scripts/extract_sl_templates.py
  python3 scripts/extract_sl_templates.py --sl ../SL --out data/templates --max-videos 0
  python3 scripts/extract_sl_templates.py --words bonjour merci oui
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SEQUENCE_LENGTH = 30
FEATURE_DIM     = 171

_SL_DEFAULT = os.path.join(ROOT, '..', 'SL')


def _resample(arr: np.ndarray, n: int = SEQUENCE_LENGTH) -> np.ndarray:
    if len(arr) == n:
        return arr
    if len(arr) < 2:
        return np.tile(arr[0:1], (n, 1)) if len(arr) else np.zeros((n, arr.shape[-1]))
    idx   = np.linspace(0, len(arr) - 1, n)
    left  = np.floor(idx).astype(int)
    right = np.minimum(left + 1, len(arr) - 1)
    t     = (idx - left)[:, None]
    return (arr[left] * (1 - t) + arr[right] * t).astype('float32')


def extract_from_video(video_path: str, mp_service, segmenter_cls) -> list[np.ndarray]:
    """Return list of (30, 171) arrays, one per detected sign segment."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    seg      = segmenter_cls()
    segments = []

    def _commit(buf):
        if len(buf) >= 3:
            arr = np.stack(buf).astype('float32')[:, :FEATURE_DIM]
            segments.append(_resample(arr, SEQUENCE_LENGTH))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        features, _, hand_visible = mp_service.process_frame(frame)

        if not hand_visible:
            if hasattr(seg, 'notify_hand_lost'):
                seg.notify_hand_lost()
            if getattr(seg, 'is_signing', False):
                _commit(getattr(seg, 'sign_buffer', []))
            seg.reset()
            continue

        result = seg.process_features(features)
        if result['sign_ended'] and result['ended_buffer']:
            _commit(result['ended_buffer'])

    # flush sign still in progress at EOF
    if getattr(seg, 'is_signing', False):
        _commit(getattr(seg, 'sign_buffer', []))

    cap.release()
    return segments


def main() -> int:
    parser = argparse.ArgumentParser(description='Extract SL video templates for V11')
    parser.add_argument('--sl',          default=_SL_DEFAULT,
                        help='SL dataset root (default: ../SL)')
    parser.add_argument('--out',         default=os.path.join(ROOT, 'data', 'templates'),
                        help='Output templates directory (default: V11/data/templates)')
    parser.add_argument('--max-videos',  type=int, default=0,
                        help='Max videos per word (0 = all)')
    parser.add_argument('--words',       nargs='*', default=None,
                        help='Restrict to these words (default: all SL words)')
    parser.add_argument('--min-frames',  type=int, default=10,
                        help='Discard segments shorter than this many raw frames (default 10)')
    args = parser.parse_args()

    sl_dir = os.path.abspath(args.sl)
    if not os.path.isdir(sl_dir):
        print(f'ERROR: SL directory not found: {sl_dir}')
        return 1

    from ml.segmenter import SignSegmenter as SegCls

    from ml.mediapipe_service import MediaPipeService
    mp = MediaPipeService()

    words = args.words or sorted(
        d for d in os.listdir(sl_dir)
        if os.path.isdir(os.path.join(sl_dir, d))
    )

    total_saved = 0
    for word in words:
        word_dir = os.path.join(sl_dir, word)
        if not os.path.isdir(word_dir):
            print(f'[SKIP] {word}: not found in SL')
            continue

        videos = sorted(f for f in os.listdir(word_dir) if f.endswith('.mp4'))
        if args.max_videos:
            videos = videos[:args.max_videos]
        if not videos:
            print(f'[SKIP] {word}: no .mp4 files')
            continue

        out_dir = os.path.join(args.out, word)
        os.makedirs(out_dir, exist_ok=True)

        # Next available index (don't overwrite existing templates)
        existing = [f for f in os.listdir(out_dir) if f.startswith('sl_') and f.endswith('.npy')]
        next_idx = len(existing)

        saved = skipped = 0
        for vf in videos:
            segs = extract_from_video(os.path.join(word_dir, vf), mp, SegCls)
            for seg in segs:
                fname = os.path.join(out_dir, f'sl_{next_idx:04d}.npy')
                np.save(fname, seg)
                next_idx  += 1
                saved     += 1
            if not segs:
                skipped += 1

        total_saved += saved
        print(f'  {word:<22s}  {saved:3d} templates saved  ({skipped} videos yielded nothing)')

    mp.close()
    print(f'\nDone — {total_saved} templates written to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
