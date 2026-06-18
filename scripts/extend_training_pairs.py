#!/usr/bin/env python3
"""
Extend training_pairs.csv with linguistically valid sequences
that cover missing sign bigrams.

Uses rule_translate() to generate correct French translations.
Only adds sequences that cover at least one missing bigram.
"""
import csv, os, sys, json, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, 'data', 'training_pairs.csv')
sys.path.insert(0, ROOT)
from ml.translator import rule_translate

# ── Vocabulary ─────────────────────────────────────────────────────────────────
SUBJECTS  = ['moi', 'tu', 'nous', 'ami', 'bebe', 'chat']
VERBS     = ['manger', 'boire', 'aller', 'apprendre', 'comprendre',
             'changer', 'aimer', 'vouloir', 'donner', 'attendre']
OBJECTS   = ['aide', 'ami', 'amour', 'bebe', 'chat', 'moi', 'tu', 'nous']
GREETINGS = ['bonjour', 'au revoir', 'merci', 'oui', 'non']
MODS      = ['bien', 'bientot', 'ici']
AVEC_OBJ  = ['ami', 'moi', 'tu', 'nous', 'bebe', 'chat']
TEMPORAL  = ['aujourd hui']

# ── Load existing + compute missing bigrams ────────────────────────────────────
with open(os.path.join(ROOT, 'data', 'ctc_model', 'classes.json')) as f:
    classes = set(json.load(f))

existing_signs = set()
existing_rows  = []
with open(CSV_PATH, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        existing_rows.append(row)
        existing_signs.add(row['signs'].strip().lower())

def parse_seq(s):
    toks = s.strip().lower().split()
    out, i = [], 0
    while i < len(toks):
        if i+1 < len(toks) and toks[i]=='au' and toks[i+1]=='revoir':
            out.append('au revoir'); i += 2
        elif i+1 < len(toks) and toks[i]=='aujourd' and toks[i+1]=='hui':
            out.append('aujourd hui'); i += 2
        else:
            out.append(toks[i]); i += 1
    return out

seen_bigrams = set()
for row in existing_rows:
    seq = parse_seq(row['signs'])
    for i in range(len(seq)-1):
        if seq[i] in classes and seq[i+1] in classes:
            seen_bigrams.add((seq[i], seq[i+1]))

missing_bigrams = {(a,b) for a in classes for b in classes
                   if a != b and (a,b) not in seen_bigrams}
print(f"Existing: {len(existing_rows)}  |  Missing bigrams: {len(missing_bigrams)}")

# ── Candidate generator ────────────────────────────────────────────────────────
candidates = []
seen_new   = set()

def add(seq):
    s = ' '.join(seq)
    if s in existing_signs or s in seen_new:
        return
    if not all(tok in classes for tok in seq):
        return
    for i in range(len(seq)-1):
        if (seq[i], seq[i+1]) in missing_bigrams:
            fr = rule_translate(seq)
            if fr and fr.strip():
                candidates.append((s, fr))
                seen_new.add(s)
            return

# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1 : [Subj] aller [Verb]  — "moi aller manger" → couvre aller→verb
for subj in SUBJECTS:
    for verb in VERBS:
        if verb != 'aller':
            add([subj, 'aller', verb])

# Pattern 2 : [Subj] aller [Verb] [Obj]
for subj in ['moi', 'tu', 'nous']:
    for verb in ['manger', 'boire', 'apprendre', 'donner', 'comprendre']:
        for obj in ['aide', 'ami', 'amour', 'bebe']:
            add([subj, 'aller', verb, obj])

# Pattern 3 : [Greeting] [Subj] [Verb]  — "bonjour moi manger"
for greet in GREETINGS:
    for subj in SUBJECTS:
        for verb in VERBS:
            add([greet, subj, verb])

# Pattern 4 : [Greeting] [Subj] [Verb] [Obj]
for greet in ['bonjour', 'merci', 'oui']:
    for subj in ['moi', 'tu', 'nous']:
        for verb in VERBS:
            for obj in ['aide', 'ami', 'amour', 'bebe', 'chat']:
                if obj != subj:
                    add([greet, subj, verb, obj])

# Pattern 5 : [oui/non] [Subj] [Verb]  — "oui moi comprendre"
for resp in ['oui', 'non']:
    for subj in SUBJECTS:
        for verb in VERBS:
            add([resp, subj, verb])

# Pattern 6 : [oui/non] [Subj] [Verb] [Obj]
for resp in ['oui', 'non']:
    for subj in ['moi', 'tu', 'nous']:
        for verb in VERBS:
            for obj in ['aide', 'ami', 'amour']:
                add([resp, subj, verb, obj])

# Pattern 7 : [Subj] [Verb] [Mod]  — "moi manger bientot"
for subj in SUBJECTS:
    for verb in VERBS:
        for mod in MODS:
            add([subj, verb, mod])

# Pattern 8 : [Mod] [Subj] [Verb]  — "bientot moi aller"
for mod in MODS:
    for subj in SUBJECTS:
        for verb in VERBS:
            add([mod, subj, verb])

# Pattern 9 : [Temporal] [Subj] [Verb] — "aujourd hui moi comprendre"
for temp in TEMPORAL:
    for subj in SUBJECTS:
        for verb in VERBS:
            add([temp, subj, verb])

# Pattern 10 : [Temporal] [Subj] [non] [Verb]
for temp in TEMPORAL:
    for subj in ['moi', 'tu', 'nous']:
        for verb in VERBS:
            add([temp, subj, 'non', verb])

# Pattern 11 : [Subj] [Verb] [Obj] [Greeting]
for subj in ['moi', 'tu', 'nous']:
    for verb in VERBS:
        for obj in ['aide', 'ami']:
            for greet in ['merci', 'bientot']:
                add([subj, verb, obj, greet])

# Pattern 12 : [Verb] [Subj] — après un verbe isolé (contexte dialogue)
for verb in VERBS:
    for subj in SUBJECTS:
        add([verb, subj])

# Pattern 13 : [Subj] [Verb1] [Subj] [Verb2] — répétition sujet pour chaîner
for subj in ['moi', 'tu', 'nous']:
    for v1 in ['manger', 'boire', 'apprendre', 'comprendre']:
        for v2 in VERBS:
            if v1 != v2:
                add([subj, v1, subj, v2])

# Pattern 14 : [Greeting] [Mod]
for greet in GREETINGS:
    for mod in MODS:
        add([greet, mod])

# Pattern 15 : [Mod] [Greeting]
for mod in MODS:
    for greet in GREETINGS:
        add([mod, greet])

# Pattern 16 : [Subj] [Verb] [avec] [Person] [Mod]
for subj in ['moi', 'tu', 'nous']:
    for verb in ['manger', 'boire', 'aller', 'aimer']:
        for avec_p in ['ami', 'bebe']:
            for mod in ['bien', 'bientot']:
                add([subj, verb, 'avec', avec_p, mod])

# Pattern 17 : [comment] [Subj] [Verb] [Obj]
for subj in SUBJECTS:
    for verb in VERBS:
        for obj in ['aide', 'ami', 'amour']:
            add(['comment', subj, verb, obj])

# Pattern 18 : [Subj] [Verb] [Obj] [Mod] — longer phrases
for subj in SUBJECTS:
    for verb in VERBS:
        for obj in OBJECTS:
            if obj != subj:
                for mod in MODS:
                    add([subj, verb, obj, mod])

# Pattern 19 : [aujourd hui] [Greeting] [Subj] [Verb]
for greet in ['bonjour', 'merci']:
    for subj in SUBJECTS:
        for verb in VERBS:
            add(['aujourd hui', greet, subj, verb])

# Pattern 20 : Séquences avec "avec" en position centrale
for subj in SUBJECTS:
    for avec_p in AVEC_OBJ:
        if avec_p != subj:
            for verb in VERBS:
                add([subj, 'avec', avec_p, verb])

# ── Stats ──────────────────────────────────────────────────────────────────────
new_bigrams = set()
for signs, _ in candidates:
    seq = parse_seq(signs)
    for i in range(len(seq)-1):
        if seq[i] in classes and seq[i+1] in classes:
            new_bigrams.add((seq[i], seq[i+1]))

covered = new_bigrams & missing_bigrams
total_after = len(seen_bigrams) + len(covered)
print(f"New sequences:     {len(candidates)}")
print(f"New bigrams:       {len(covered)} / {len(missing_bigrams)} missing covered")
print(f"Bigram coverage:   {total_after} / {len(classes)**2} "
      f"({total_after / len(classes)**2 * 100:.1f}%)")

# ── Append to CSV ──────────────────────────────────────────────────────────────
with open(CSV_PATH, 'a', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    for signs, fr in candidates:
        writer.writerow({'signs': signs, 'sentence': fr})

print(f"\nAppended {len(candidates)} sequences → {CSV_PATH}")
print(f"Total: {len(existing_rows) + len(candidates)} sequences")
