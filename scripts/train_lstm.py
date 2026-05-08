#!/usr/bin/env python3
"""CLI: train the V11 BiLSTM classifier.

    python scripts/train_lstm.py
    python scripts/train_lstm.py --templates-dir ../V8/data/templates
    python scripts/train_lstm.py --templates-dir /path/to/templates --epochs 300
"""
from __future__ import annotations
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml.lstm_trainer import fit_and_save


def main() -> int:
    parser = argparse.ArgumentParser(description='Train V11 LSTM classifier')
    parser.add_argument('--templates-dir', default=None,
                        help='Path to templates directory (default: auto-detect V11 then V8)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum training epochs (default: 200)')
    args = parser.parse_args()
    fit_and_save(templates_dir=args.templates_dir, epochs=args.epochs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
