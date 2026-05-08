#!/usr/bin/env python3
"""
Train a neural slot-filling model for LSF→French translation.

Architecture: BiLSTM → multi-task classification heads
  - greeting  : 4 classes (none + bonjour/au revoir/merci)
  - subject   : 7 classes (none + 6 subjects)
  - verb       : 11 classes (none + 10 verbs)
  - inf_verb  : 11 classes (none + 10 verbs, for modal + infinitive)
  - negated    : binary
  - objects    : multi-label (8 classes)
  - mods       : multi-label (5 classes)
  - avec       : 7 classes (none + 6 persons)

Labels are auto-extracted from sign sequences (deterministic).
At inference, predicted slots are fed to the rule-based grammar.
"""
import csv, json, os, pickle, sys
import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    Dense, Dropout, GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(ROOT, 'data', 'training_pairs.csv')
OUT_DIR  = os.path.join(ROOT, 'data', 'slot_model')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Slot vocabularies ──────────────────────────────────────────────────────────
GREETINGS = ['<none>', 'bonjour', 'au revoir', 'merci']
SUBJECTS  = ['<none>', 'moi', 'tu', 'nous', 'ami', 'bebe', 'chat']
VERBS     = ['<none>', 'manger', 'boire', 'aller', 'apprendre', 'comprendre',
             'changer', 'aimer', 'vouloir', 'donner', 'attendre']
INF_VERBS = VERBS  # same vocab reused for the infinitive slot
OBJECTS   = ['aide', 'ami', 'amour', 'bebe', 'chat', 'moi', 'tu', 'nous']
MODS      = ['bien', 'bon', 'ici', 'bientot', 'aujourd hui']
AVEC      = ['<none>', 'moi', 'tu', 'nous', 'ami', 'bebe', 'chat']

_GREET_SET = set(GREETINGS[1:])
_SUBJ_SET  = set(SUBJECTS[1:])
_VERB_SET  = set(VERBS[1:])
_OBJ_SET   = set(OBJECTS)
_MOD_SET   = set(MODS)
_AVEC_SET  = set(AVEC[1:])
GREET_IDX  = {g: i for i, g in enumerate(GREETINGS)}
SUBJ_IDX   = {s: i for i, s in enumerate(SUBJECTS)}
VERB_IDX   = {v: i for i, v in enumerate(VERBS)}
AVEC_IDX   = {a: i for i, a in enumerate(AVEC)}


def _tokenise(sign_str: str) -> list[str]:
    s = sign_str.strip().lower()
    for mw in ('au revoir', 'aujourd hui'):
        s = s.replace(mw, mw.replace(' ', '\x00'))
    return [t.replace('\x00', ' ') for t in s.split()]


def extract_slots(sign_str: str) -> dict:
    toks = _tokenise(sign_str)

    greeting = next((t for t in toks if t in _GREET_SET), None)
    subject  = next((t for t in toks if t in _SUBJ_SET), None)
    verb     = next((t for t in toks if t in _VERB_SET), None)
    negated  = 'non' in toks

    verb_pos  = toks.index(verb) if verb else len(toks)
    post_verb = toks[verb_pos + 1:]

    # Infinitive verb: second verb after the main verb (e.g. vouloir + apprendre)
    inf_verb = next((t for t in post_verb if t in _VERB_SET), None)

    # avec person
    avec_person = None
    if 'avec' in toks:
        av = toks.index('avec')
        if av + 1 < len(toks) and toks[av + 1] in _AVEC_SET:
            avec_person = toks[av + 1]

    # objects: exclude subject, verbs, and avec_person to avoid double-counting
    objects = [t for t in post_verb
               if t in _OBJ_SET
               and t != subject
               and t not in _VERB_SET
               and t != avec_person]

    mods = [t for t in toks if t in _MOD_SET]

    return {
        'greeting': GREET_IDX.get(greeting, 0),
        'subject':  SUBJ_IDX.get(subject, 0),
        'verb':     VERB_IDX.get(verb, 0),
        'inf_verb': VERB_IDX.get(inf_verb, 0),
        'negated':  int(negated),
        'objects':  [int(o in objects) for o in OBJECTS],
        'mods':     [int(m in mods)    for m in MODS],
        'avec':     AVEC_IDX.get(avec_person, 0),
    }


# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading training_pairs.csv …")
signs_raw, slots_all = [], []
with open(DATA_CSV, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        s = row['signs'].strip().lower()
        signs_raw.append(s)
        slots_all.append(extract_slots(s))

print(f"  {len(signs_raw)} pairs")

# ── Tokenise signs ─────────────────────────────────────────────────────────────
tok = Tokenizer(filters='', lower=True, oov_token='<unk>')
tok.fit_on_texts(signs_raw)
VOCAB   = len(tok.word_index) + 1
seqs    = tok.texts_to_sequences(signs_raw)
MAX_LEN = max(len(s) for s in seqs)
X       = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
print(f"  Vocab: {VOCAB}  |  Max seq len: {MAX_LEN}")

# ── Label arrays ───────────────────────────────────────────────────────────────
Y_greet = np.array([s['greeting'] for s in slots_all])
Y_subj  = np.array([s['subject']  for s in slots_all])
Y_verb  = np.array([s['verb']     for s in slots_all])
Y_inf   = np.array([s['inf_verb'] for s in slots_all])
Y_neg   = np.array([s['negated']  for s in slots_all], dtype=np.float32)
Y_obj   = np.array([s['objects']  for s in slots_all], dtype=np.float32)
Y_mods  = np.array([s['mods']     for s in slots_all], dtype=np.float32)
Y_avec  = np.array([s['avec']     for s in slots_all])

# ── Model ──────────────────────────────────────────────────────────────────────
EMBED, HIDDEN = 32, 64

inp  = Input(shape=(MAX_LEN,), name='signs')
emb  = Embedding(VOCAB, EMBED, mask_zero=True)(inp)
x    = Bidirectional(LSTM(HIDDEN, return_sequences=True))(emb)
x    = GlobalAveragePooling1D()(x)
x    = Dropout(0.3)(x)

out_greet = Dense(len(GREETINGS), activation='softmax', name='greeting')(x)
out_subj  = Dense(len(SUBJECTS),  activation='softmax', name='subject')(x)
out_verb  = Dense(len(VERBS),     activation='softmax', name='verb')(x)
out_inf   = Dense(len(INF_VERBS), activation='softmax', name='inf_verb')(x)
out_neg   = Dense(1,              activation='sigmoid', name='negated')(x)
out_obj   = Dense(len(OBJECTS),   activation='sigmoid', name='objects')(x)
out_mods  = Dense(len(MODS),      activation='sigmoid', name='mods')(x)
out_avec  = Dense(len(AVEC),      activation='softmax', name='avec')(x)

model = Model(inp, [out_greet, out_subj, out_verb, out_inf, out_neg, out_obj, out_mods, out_avec])
model.compile(
    optimizer='adam',
    loss={
        'greeting': 'sparse_categorical_crossentropy',
        'subject':  'sparse_categorical_crossentropy',
        'verb':     'sparse_categorical_crossentropy',
        'inf_verb': 'sparse_categorical_crossentropy',
        'negated':  'binary_crossentropy',
        'objects':  'binary_crossentropy',
        'mods':     'binary_crossentropy',
        'avec':     'sparse_categorical_crossentropy',
    },
    metrics={'greeting': 'accuracy', 'subject': 'accuracy',
             'verb': 'accuracy', 'negated': 'accuracy', 'avec': 'accuracy'},
)
model.summary()

# ── Train ──────────────────────────────────────────────────────────────────────
print(f"\nTraining …")
model.fit(
    X,
    {'greeting': Y_greet, 'subject': Y_subj, 'verb': Y_verb, 'inf_verb': Y_inf,
     'negated': Y_neg, 'objects': Y_obj, 'mods': Y_mods, 'avec': Y_avec},
    epochs=60,
    batch_size=64,
    validation_split=0.05,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    ],
    verbose=1,
)

# ── Save ───────────────────────────────────────────────────────────────────────
model.save(os.path.join(OUT_DIR, 'model.keras'))
with open(os.path.join(OUT_DIR, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tok, f)
with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump({'max_len': MAX_LEN, 'vocab': VOCAB,
               'greetings': GREETINGS, 'subjects': SUBJECTS,
               'verbs': VERBS, 'objects': OBJECTS,
               'mods': MODS, 'avec': AVEC}, f, indent=2)
print(f"\nSaved → {OUT_DIR}")

# ── End-to-end inference test ──────────────────────────────────────────────────
print("\n── End-to-end test (neural slots → rule grammar) ─────────────────────")
sys.path.insert(0, ROOT)
from ml.translator import rule_translate

def neural_translate(signs_list: list) -> str:
    raw = ' '.join(s.lower() for s in signs_list)
    seq = tok.texts_to_sequences([raw])
    seq = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    preds = model.predict(seq, verbose=0)
    p_greet, p_subj, p_verb, p_inf, p_neg, p_obj, p_mods, p_avec = preds

    greet_str = GREETINGS[int(np.argmax(p_greet[0]))]
    subj_str  = SUBJECTS[int(np.argmax(p_subj[0]))]
    verb_str  = VERBS[int(np.argmax(p_verb[0]))]
    inf_str   = INF_VERBS[int(np.argmax(p_inf[0]))]
    negated   = float(p_neg[0][0]) > 0.5
    objs      = [OBJECTS[i] for i, v in enumerate(p_obj[0]) if v > 0.5]
    mods      = [MODS[i]    for i, v in enumerate(p_mods[0]) if v > 0.5]
    avec_str  = AVEC[int(np.argmax(p_avec[0]))]

    # Reconstruct canonical sign list for rule_translate
    reconstructed = []
    if greet_str != '<none>': reconstructed.append(greet_str)
    if subj_str  != '<none>': reconstructed.append(subj_str)
    if negated:               reconstructed.append('non')
    if verb_str  != '<none>': reconstructed.append(verb_str)
    if inf_str   != '<none>': reconstructed.append(inf_str)
    reconstructed.extend(objs)
    if avec_str  != '<none>': reconstructed.extend(['avec', avec_str])
    reconstructed.extend(mods)
    return rule_translate(reconstructed) if reconstructed else '(rien)'


tests = [
    ["moi", "manger"],
    ["tu", "boire"],
    ["nous", "apprendre"],
    ["moi", "non", "comprendre"],
    ["tu", "non", "manger"],
    ["moi", "vouloir", "apprendre"],
    ["moi", "aimer", "chat"],
    ["moi", "donner", "aide", "ami"],
    ["ami", "attendre", "moi"],
    ["moi", "manger", "avec", "ami"],
    ["nous", "apprendre", "aujourd hui"],
    ["moi", "non", "vouloir", "aide"],
    ["moi", "non", "aimer", "chat"],
    ["bonjour"],
    ["au revoir", "ami"],
]

ok = 0
for signs in tests:
    rule_out   = rule_translate(signs)
    neural_out = neural_translate(signs)
    match = "✓" if rule_out == neural_out else "✗"
    print(f"  {match}  {' '.join(signs):30s}  →  {neural_out}")
    if rule_out == neural_out:
        ok += 1
    else:
        print(f"      (rule: {rule_out})")

print(f"\n{ok}/{len(tests)} match rule-based output")
