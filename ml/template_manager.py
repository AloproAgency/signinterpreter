"""Template file management."""
import os
import shutil
import numpy as np
from ml.constants import TEMPLATE_DIR, CONTRIBUTIONS_DIR
from ml.features import FEATURE_DIM


def normalize_word(name: str) -> str:
    """Canonical form for a vocabulary word.

    Trim whitespace and lowercase so that 'Bon ', 'BON', 'bon' all map
    to the same identifier.  Matters because (1) macOS APFS is case-
    insensitive and silently merged data/templates/{bon,BON}/ folders,
    while the SQLite DB is case-sensitive — leading to duplicate Word
    rows pointing to the same physical directory; (2) signers tend to
    type the word in any case when contributing.
    """
    return (name or '').strip().lower()


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
    word = normalize_word(word)
    word_dir = os.path.join(TEMPLATE_DIR, word)
    if not os.path.isdir(word_dir):
        return []
    files = sorted([f for f in os.listdir(word_dir) if f.endswith('.npy')],
                   key=lambda x: int(x.replace('.npy', '')))
    return [{'index': int(f.replace('.npy', '')), 'file': f} for f in files]


def get_next_index(word):
    """Get next available template index for a word."""
    word = normalize_word(word)
    word_dir = os.path.join(TEMPLATE_DIR, word)
    os.makedirs(word_dir, exist_ok=True)
    existing = [int(f.replace('.npy', ''))
                for f in os.listdir(word_dir) if f.endswith('.npy')]
    return max(existing) + 1 if existing else 0


def save_template(word, template_array, unique_id=None):
    """Save a template array to disk. Returns (path, idx).

    If `unique_id` is provided (typically a Contribution.id from the
    DB), the template is saved as `<word>/<unique_id>.npy`.  This
    eliminates the race condition that the previous sequential-index
    scheme suffered: two concurrent approvals would both call
    `get_next_index(word)`, get the same N, and write `N.npy` —
    silently overwriting one of the two templates.

    If `unique_id` is None we fall back to atomic sequential allocation
    via O_CREAT|O_EXCL with retry on collision (still safe under
    concurrent calls).
    """
    word = normalize_word(word)
    word_dir = os.path.join(TEMPLATE_DIR, word)
    os.makedirs(word_dir, exist_ok=True)

    if unique_id is not None:
        idx = int(unique_id)
        path = os.path.join(word_dir, f'{idx}.npy')
        np.save(path, template_array)
        return path, idx

    # Atomic sequential allocation — retry until O_EXCL succeeds.
    for _ in range(64):
        idx = get_next_index(word)
        path = os.path.join(word_dir, f'{idx}.npy')
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        try:
            with os.fdopen(fd, 'wb') as f:
                np.save(f, template_array)
        except Exception:
            try: os.remove(path)
            except OSError: pass
            raise
        return path, idx
    raise RuntimeError('save_template: too many index collisions')


def save_contribution(contribution_id, template_array):
    """Save a contribution template to pending directory."""
    os.makedirs(CONTRIBUTIONS_DIR, exist_ok=True)
    path = os.path.join(CONTRIBUTIONS_DIR, f'{contribution_id}.npy')
    np.save(path, template_array)
    return path


def approve_contribution(contribution_path, word, unique_id=None):
    """Move a contribution from pending to templates.

    If `unique_id` is supplied (the DB contribution id), the destination
    filename is deterministic and free of race conditions — see
    `save_template` doc for the rationale.
    """
    template_array = np.load(contribution_path)
    path, idx = save_template(word, template_array, unique_id=unique_id)
    if os.path.exists(contribution_path):
        os.remove(contribution_path)
    return path, idx


def delete_word(word):
    """Delete a word and all its templates."""
    word = normalize_word(word)
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
