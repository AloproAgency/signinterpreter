"""Template file management."""
import os
import shutil
import numpy as np
from ml.constants import TEMPLATE_DIR, CONTRIBUTIONS_DIR
from ml.features import FEATURE_DIM


def list_words():
    """List all words with template counts."""
    if not os.path.isdir(TEMPLATE_DIR):
        return []
    words = []
    for d in sorted(os.listdir(TEMPLATE_DIR)):
        word_dir = os.path.join(TEMPLATE_DIR, d)
        if not os.path.isdir(word_dir):
            continue
        n = len([f for f in os.listdir(word_dir) if f.endswith('.npy')])
        words.append({'name': d, 'template_count': n})
    return words


def list_templates(word):
    """List template files for a word."""
    word_dir = os.path.join(TEMPLATE_DIR, word)
    if not os.path.isdir(word_dir):
        return []
    files = sorted([f for f in os.listdir(word_dir) if f.endswith('.npy')],
                   key=lambda x: int(x.replace('.npy', '')))
    return [{'index': int(f.replace('.npy', '')), 'file': f} for f in files]


def get_next_index(word):
    """Get next available template index for a word."""
    word_dir = os.path.join(TEMPLATE_DIR, word)
    os.makedirs(word_dir, exist_ok=True)
    existing = [int(f.replace('.npy', ''))
                for f in os.listdir(word_dir) if f.endswith('.npy')]
    return max(existing) + 1 if existing else 0


def save_template(word, template_array):
    """Save a template array to disk. Returns the file path."""
    idx = get_next_index(word)
    word_dir = os.path.join(TEMPLATE_DIR, word)
    os.makedirs(word_dir, exist_ok=True)
    path = os.path.join(word_dir, f'{idx}.npy')
    np.save(path, template_array)
    return path, idx


def save_contribution(contribution_id, template_array):
    """Save a contribution template to pending directory."""
    os.makedirs(CONTRIBUTIONS_DIR, exist_ok=True)
    path = os.path.join(CONTRIBUTIONS_DIR, f'{contribution_id}.npy')
    np.save(path, template_array)
    return path


def approve_contribution(contribution_path, word):
    """Move a contribution from pending to templates."""
    template_array = np.load(contribution_path)
    path, idx = save_template(word, template_array)
    # Remove pending file
    if os.path.exists(contribution_path):
        os.remove(contribution_path)
    return path, idx


def delete_word(word):
    """Delete a word and all its templates."""
    word_dir = os.path.join(TEMPLATE_DIR, word)
    if os.path.isdir(word_dir):
        shutil.rmtree(word_dir)
        return True
    return False


def count_total_templates():
    """Count total templates across all words."""
    total = 0
    if os.path.isdir(TEMPLATE_DIR):
        for d in os.listdir(TEMPLATE_DIR):
            word_dir = os.path.join(TEMPLATE_DIR, d)
            if os.path.isdir(word_dir):
                total += len([f for f in os.listdir(word_dir) if f.endswith('.npy')])
    return total
