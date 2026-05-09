#!/usr/bin/env python3
"""
Generate 30,000 synthetic CTC training sequences from real LSF sign sequences.

Source of sign combinations: data/training_pairs.csv (5,801 linguistically
valid sequences). Each sequence is re-sampled ~5× with different template
picks, speed factors, and noise — giving visual diversity while preserving
linguistic correctness.
"""
import csv, json, os, pickle, random
import numpy as np

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(ROOT, 'data', 'templates')
CSV_PATH     = os.path.join(ROOT, 'data', 'training_pairs.csv')
OUT_PKL      = os.path.join(ROOT, 'data', 'ctc_sequences.pkl')
OUT_CLASSES  = os.path.join(ROOT, 'data', 'ctc_classes.json')

N          = 30_000
SPEED_MIN  = 0.75   # templates stretched to 75–125 % of original length
SPEED_MAX  = 1.25
TRANS_N    = 3      # linear-interpolated frames at sign boundaries
NOISE_STD  = 0.005


def parse_signs(raw: str, known: set) -> list[str] | None:
    """Parse a signs string, merging multi-word tokens. Returns None if any sign is unknown."""
    toks = raw.strip().lower().split()
    merged = []
    i = 0
    while i < len(toks):
        if i + 1 < len(toks) and toks[i] == 'au' and toks[i + 1] == 'revoir':
            merged.append('au revoir'); i += 2
        elif i + 1 < len(toks) and toks[i] == 'aujourd' and toks[i + 1] == 'hui':
            merged.append('aujourd hui'); i += 2
        else:
            merged.append(toks[i]); i += 1
    if all(s in known for s in merged):
        return merged
    return None


def stretch(frames: np.ndarray, factor: float) -> np.ndarray:
    """Vectorised linear resampling along the time axis."""
    T    = len(frames)
    new_T = max(12, round(T * factor))
    idx  = np.linspace(0, T - 1, new_T)
    lo   = np.floor(idx).astype(np.int32)
    hi   = np.minimum(lo + 1, T - 1)
    frac = (idx - lo)[:, None]           # (new_T, 1) — broadcast over feature dim
    return (frames[lo] * (1.0 - frac) + frames[hi] * frac).astype('float32')


def make_transition(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """n frames linearly blended between last frame of a and first frame of b."""
    alphas = np.linspace(0, 1, n + 2)[1:-1, None]
    return (a[-1] * (1.0 - alphas) + b[0] * alphas).astype('float32')


def build_sequence(sign_list: list[str], templates: dict,
                   counters: dict) -> tuple[np.ndarray, list[str]]:
    parts = []
    prev  = None
    for sign in sign_list:
        # Stratified round-robin: chaque template est utilisé à tour de rôle
        # avant de recommencer, garantissant une couverture uniforme.
        n = len(templates[sign])
        idx = counters[sign] % n
        counters[sign] += 1
        frames = stretch(templates[sign][idx], random.uniform(SPEED_MIN, SPEED_MAX))
        if prev is not None:
            parts.append(make_transition(prev, frames, TRANS_N))
        parts.append(frames)
        prev = frames
    seq = np.concatenate(parts, axis=0)
    seq += np.random.normal(0, NOISE_STD, seq.shape).astype('float32')
    return seq.astype('float16'), sign_list


def main():
    random.seed(42)
    np.random.seed(42)

    # ── Load templates ────────────────────────────────────────────────────
    classes = sorted(os.listdir(TEMPLATE_DIR))
    known   = set(classes)
    templates = {}
    for c in classes:
        cdir  = os.path.join(TEMPLATE_DIR, c)
        files = sorted(f for f in os.listdir(cdir) if f.endswith('.npy'))
        templates[c] = [np.load(os.path.join(cdir, f)).astype('float32') for f in files]

    n_tmpl = sum(len(v) for v in templates.values())
    print(f"Templates : {n_tmpl} across {len(classes)} classes")

    # ── Load real LSF sequences from training_pairs.csv ───────────────────
    valid_seqs = []
    with open(CSV_PATH, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            parsed = parse_signs(row['signs'], known)
            if parsed:
                valid_seqs.append(parsed)

    print(f"Valid sequences from CSV: {len(valid_seqs)}")
    import collections
    dist = collections.Counter(len(s) for s in valid_seqs)
    for k in sorted(dist):
        print(f"  {k} signes : {dist[k]}")

    # ── Weighted sampling : surreprésenter les séquences contenant des signes rares ─
    # Le poids d'une séquence = inverse de la fréquence de son signe le plus rare.
    # Cela préserve le contexte linguistique tout en donnant plus de visibilité
    # aux signes peu fréquents dans la distribution naturelle du LSF.
    import collections as col
    uniform_counts = col.Counter(sign for seq in valid_seqs for sign in seq)
    mean_count = float(np.mean(list(uniform_counts.values())))

    weights = np.array([
        max(mean_count / uniform_counts[s] for s in seq)
        for seq in valid_seqs
    ], dtype=float)
    weights /= weights.sum()

    indices = np.random.choice(len(valid_seqs), size=N, replace=True, p=weights)
    pool = [valid_seqs[i] for i in indices]

    seqs_out   = []
    labels_out = []
    class_idx  = {c: i for i, c in enumerate(classes)}
    counters   = {c: 0 for c in classes}   # round-robin index per class

    for i, sign_list in enumerate(pool):
        seq, signs = build_sequence(sign_list, templates, counters)
        seqs_out.append(seq)
        labels_out.append([class_idx[s] for s in signs])
        if (i + 1) % 5_000 == 0:
            print(f"  {i + 1:6d} / {N}")

    # ── Stats & CTC constraint check ──────────────────────────────────────
    frame_lens = [len(s) for s in seqs_out]
    print(f"\nFrame lengths — min:{min(frame_lens)}  "
          f"mean:{int(np.mean(frame_lens))}  max:{max(frame_lens)}")

    violations = sum(1 for s, l in zip(seqs_out, labels_out)
                     if len(s) < 2 * len(l) - 1)
    print(f"CTC constraint violations: {violations}  (must be 0)")

    # ── Save ──────────────────────────────────────────────────────────────
    payload = {'classes': classes, 'seqs': seqs_out, 'labels': labels_out}
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(payload, f, protocol=4)

    size_mb = os.path.getsize(OUT_PKL) / 1e6
    print(f"\nSaved {N} sequences → {OUT_PKL}  ({size_mb:.0f} MB)")

    with open(OUT_CLASSES, 'w') as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)
    print(f"Classes        → {OUT_CLASSES}")


if __name__ == '__main__':
    main()
