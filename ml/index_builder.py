"""FAISS index building service."""
import numpy as np
import os
import json
import time
import faiss
from ml.features import FEATURE_DIM, FRAME_FEATURE_DIM, add_wrist_velocity
from ml.constants import TEMPLATE_DIR, INDEX_PATH, METADATA_PATH, DATA_DIR

SUMMARY_DIM = 4 * FEATURE_DIM


def compute_summary(template):
    return np.concatenate([
        template.mean(axis=0),
        template.std(axis=0),
        template[0],
        template[-1],
    ]).astype('float32')


def build_index():
    """Build FAISS index from templates. Returns stats dict."""
    t0 = time.time()

    words = sorted([d for d in os.listdir(TEMPLATE_DIR)
                    if os.path.isdir(os.path.join(TEMPLATE_DIR, d))])

    summaries = []
    metadata = []

    for word in words:
        word_dir = os.path.join(TEMPLATE_DIR, word)
        files = sorted([f for f in os.listdir(word_dir) if f.endswith('.npy')],
                       key=lambda x: int(x.replace('.npy', '')))

        for f in files:
            path = os.path.join(word_dir, f)
            template = np.load(path)
            if template.shape[1] not in (FRAME_FEATURE_DIM, FEATURE_DIM):
                continue
            # Add wrist velocity if not already present
            template = add_wrist_velocity(template)
            summary = compute_summary(template)
            summaries.append(summary)
            metadata.append({
                'word': word,
                'template_idx': int(f.replace('.npy', '')),
                'path': os.path.relpath(path, os.path.dirname(TEMPLATE_DIR)),
            })

    if not summaries:
        return {'status': 'failed', 'error': 'No valid templates found'}

    summaries_array = np.array(summaries, dtype='float32')
    index = faiss.IndexFlatL2(SUMMARY_DIM)
    index.add(summaries_array)

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    duration_ms = int((time.time() - t0) * 1000)

    return {
        'status': 'success',
        'n_words': len(words),
        'n_templates': len(summaries),
        'summary_dim': SUMMARY_DIM,
        'duration_ms': duration_ms,
    }
