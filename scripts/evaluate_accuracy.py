#!/usr/bin/env python3
"""
Evaluate the inference engine on the SL dataset for the words currently
in data/templates/. For each video we replay the same pipeline as the
WebSocket: per-frame MediaPipe → SignSegmenter → InferenceEngine.predict.

Usage:
  python3 scripts/evaluate_accuracy.py [--sl /path/to/SL] [--threshold 150]
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml.mediapipe_service import MediaPipeService
from ml.inference_engine import InferenceEngine, SignSegmenter
from ml.phono_engine import PhonoEngine
from ml.constants import THRESHOLD, TEMPLATE_DIR


def evaluate_video(video_path, mp_service, engine, threshold):
    """Replay one video, return list of (predicted_word, distance) for each detected sign."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    segmenter = SignSegmenter()
    predictions = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        features, _, hand_visible = mp_service.process_frame(frame)
        if not hand_visible:
            segmenter.reset()
            continue

        seg = segmenter.process_features(features)

        if seg['sign_ended'] and seg['ended_buffer']:
            best_word, best_dist, _ = engine.predict(seg['ended_buffer'])
            if best_word is not None:
                predictions.append((best_word, best_dist))

    # Catch a sign still in progress at end of video
    if segmenter.is_signing and len(segmenter.sign_buffer) >= 4:
        best_word, best_dist, _ = engine.predict(segmenter.sign_buffer)
        if best_word is not None:
            predictions.append((best_word, best_dist))

    cap.release()
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sl', default=os.path.join(os.path.dirname(ROOT), 'SL'),
                        help='Path to SL dataset root')
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    parser.add_argument('--max-videos', type=int, default=0,
                        help='Limit videos per word (0 = all)')
    parser.add_argument('--engine', choices=('phono', 'dtw'), default='phono')
    args = parser.parse_args()

    if not os.path.isdir(args.sl):
        print(f'ERROR: SL dataset not found at {args.sl}')
        sys.exit(1)

    words = sorted(d for d in os.listdir(TEMPLATE_DIR)
                   if os.path.isdir(os.path.join(TEMPLATE_DIR, d)))
    print(f'Words in templates: {words}\n')

    mp_service = MediaPipeService()
    engine = PhonoEngine() if args.engine == 'phono' else InferenceEngine()
    engine.load()
    if not engine.loaded:
        print(f'ERROR: {args.engine} engine could not load.')
        sys.exit(1)
    print(f'Using {args.engine.upper()} engine')

    per_word = {}
    confusion = defaultdict(lambda: defaultdict(int))
    total_correct = 0
    total_predictions = 0
    total_below_threshold = 0
    t0 = time.time()

    for word in words:
        word_dir = os.path.join(args.sl, word)
        if not os.path.isdir(word_dir):
            print(f'  [SKIP] {word}: no folder in SL dataset')
            continue
        videos = sorted(f for f in os.listdir(word_dir) if f.endswith('.mp4'))
        if args.max_videos:
            videos = videos[:args.max_videos]
        if not videos:
            print(f'  [SKIP] {word}: no videos')
            continue

        correct = 0
        predicted_any = 0
        below_thresh = 0
        for v in videos:
            preds = evaluate_video(os.path.join(word_dir, v), mp_service, engine, args.threshold)
            if not preds:
                continue
            # Pick the prediction with the smallest distance (most confident)
            preds.sort(key=lambda p: p[1])
            best, dist = preds[0]
            predicted_any += 1
            if dist > args.threshold:
                below_thresh += 1
                continue
            confusion[word][best] += 1
            if best == word:
                correct += 1

        per_word[word] = {
            'videos': len(videos),
            'predicted': predicted_any,
            'below_thresh': below_thresh,
            'correct': correct,
            'accuracy': correct / len(videos) if videos else 0.0,
        }
        total_correct += correct
        total_predictions += len(videos)
        total_below_threshold += below_thresh
        print(f'  {word:10s} {correct:3d}/{len(videos):3d}  '
              f'(predicted {predicted_any}, rejected {below_thresh})  '
              f'acc={per_word[word]["accuracy"]:.0%}')

    elapsed = time.time() - t0
    print(f'\nOverall: {total_correct}/{total_predictions} '
          f'= {total_correct/max(total_predictions,1):.1%}  '
          f'(threshold={args.threshold}, {elapsed:.1f}s)\n')

    print('Confusion (rows = ground truth, cols = predicted):')
    pred_words = sorted({p for d in confusion.values() for p in d})
    header = '            ' + ''.join(f'{w[:6]:>7s}' for w in pred_words)
    print(header)
    for gt in words:
        row = f'  {gt:10s}' + ''.join(f'{confusion[gt].get(p, 0):7d}' for p in pred_words)
        print(row)

    mp_service.close()


if __name__ == '__main__':
    main()
