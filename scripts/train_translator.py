#!/usr/bin/env python3
"""
Train the LSF→French seq2seq translator on training_pairs.csv.

Architecture: Encoder LSTM + Decoder LSTM + Bahdanau Attention
Compatible with V11/ml/translator.py inference code.

Usage:
    python scripts/train_translator.py [--epochs 80] [--hidden 256] [--embed 64]
"""
import argparse
import csv
import json
import os
import pickle
import sys
import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Concatenate, Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV  = os.path.join(ROOT, 'data', 'training_pairs.csv')
OUT_DIR   = os.path.join(ROOT, 'data', 'sentence_model')
os.makedirs(OUT_DIR, exist_ok=True)

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',  type=int, default=80)
parser.add_argument('--hidden',  type=int, default=256)
parser.add_argument('--embed',   type=int, default=64)
parser.add_argument('--batch',   type=int, default=128)
args = parser.parse_args()

HIDDEN_DIM = args.hidden
EMBED_DIM  = args.embed
EPOCHS     = args.epochs
BATCH_SIZE = args.batch

print(f"Config: hidden={HIDDEN_DIM}, embed={EMBED_DIM}, epochs={EPOCHS}, batch={BATCH_SIZE}")

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading training_pairs.csv …")
signs_raw, sents_raw = [], []
with open(DATA_CSV, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        signs_raw.append(row['signs'].strip().lower())
        sents_raw.append(row['sentence'].strip().lower())

print(f"  {len(signs_raw)} pairs loaded")

# ── tokenizers ────────────────────────────────────────────────────────────────
sign_tok = Tokenizer(filters='', lower=True, oov_token='<unk>')
sign_tok.fit_on_texts(signs_raw)

# Keep < > so that <start> and <end> tokens are NOT split/stripped
sent_tok = Tokenizer(filters='!"#$%&()*+,-./:;?@[\\]^_`{|}~\t\n',
                     lower=True, oov_token='<unk>')
sents_with_markers = [f"<start> {s} <end>" for s in sents_raw]
sent_tok.fit_on_texts(sents_with_markers)

SIGN_VOCAB = len(sign_tok.word_index) + 1
SENT_VOCAB = len(sent_tok.word_index) + 1
print(f"  Sign vocab: {SIGN_VOCAB}  |  French vocab: {SENT_VOCAB}")

# ── encode sequences ──────────────────────────────────────────────────────────
enc_seqs = sign_tok.texts_to_sequences(signs_raw)
dec_seqs = sent_tok.texts_to_sequences(sents_with_markers)

MAX_SIGN_LEN = max(len(s) for s in enc_seqs)
MAX_SENT_LEN = max(len(s) for s in dec_seqs)
print(f"  Max sign len: {MAX_SIGN_LEN}  |  Max sent len: {MAX_SENT_LEN}")

enc_padded = pad_sequences(enc_seqs, maxlen=MAX_SIGN_LEN, padding='post')

# decoder input:  <start> tok1 tok2 ...
# decoder target: tok1 tok2 ... <end>
dec_input  = pad_sequences([s[:-1] for s in dec_seqs], maxlen=MAX_SENT_LEN - 1, padding='post')
dec_target = pad_sequences([s[1:]  for s in dec_seqs], maxlen=MAX_SENT_LEN - 1, padding='post')

# one-hot targets for sparse_categorical_crossentropy
dec_target_exp = np.expand_dims(dec_target, -1)

print(f"  Encoder input:  {enc_padded.shape}")
print(f"  Decoder input:  {dec_input.shape}")
print(f"  Decoder target: {dec_target_exp.shape}")

# ── model ─────────────────────────────────────────────────────────────────────
# Encoder
enc_in   = Input(shape=(None,), name='encoder_inputs')
enc_emb  = Embedding(SIGN_VOCAB, EMBED_DIM, mask_zero=True)(enc_in)
enc_lstm = LSTM(HIDDEN_DIM, return_sequences=True, return_state=True, name='encoder_lstm')
enc_out, enc_h, enc_c = enc_lstm(enc_emb)

# Decoder
dec_in   = Input(shape=(None,), name='decoder_inputs')
dec_emb  = Embedding(SENT_VOCAB, EMBED_DIM, mask_zero=True)(dec_in)
dec_lstm = LSTM(HIDDEN_DIM, return_sequences=True, return_state=True, name='decoder_lstm')
dec_out, _, _ = dec_lstm(dec_emb, initial_state=[enc_h, enc_c])

# Attention (Bahdanau-style: query=decoder, value=encoder outputs)
attn   = Attention(name='attention')
ctx    = attn([dec_out, enc_out])
merged = Concatenate()([dec_out, ctx])

dense_tanh = Dense(HIDDEN_DIM, activation='tanh', name='dense_tanh')
hidden = dense_tanh(merged)

out_layer = Dense(SENT_VOCAB, activation='softmax', name='output')
output = out_layer(hidden)

model = Model([enc_in, dec_in], output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

# ── callbacks ─────────────────────────────────────────────────────────────────
ckpt_path = os.path.join(OUT_DIR, 'best_model.keras')
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
    ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
]

# ── train ─────────────────────────────────────────────────────────────────────
print(f"\nTraining for up to {EPOCHS} epochs …")
history = model.fit(
    [enc_padded, dec_input],
    dec_target_exp,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.05,
    callbacks=callbacks,
    verbose=1,
)

# ── save ──────────────────────────────────────────────────────────────────────
model_path = os.path.join(OUT_DIR, 'model.keras')
model.save(model_path)
print(f"\nModel saved → {model_path}")

with open(os.path.join(OUT_DIR, 'sign_tokenizer.pkl'), 'wb') as f:
    pickle.dump(sign_tok, f)
with open(os.path.join(OUT_DIR, 'sentence_tokenizer.pkl'), 'wb') as f:
    pickle.dump(sent_tok, f)

config = {
    'hidden_dim':    HIDDEN_DIM,
    'embed_dim':     EMBED_DIM,
    'sign_vocab':    SIGN_VOCAB,
    'sent_vocab':    SENT_VOCAB,
    'max_sign_len':  MAX_SIGN_LEN,
    'max_sent_len':  MAX_SENT_LEN,
    'final_acc':     float(history.history['accuracy'][-1]),
    'final_val_acc': float(history.history['val_accuracy'][-1]),
}
with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"Config saved → {os.path.join(OUT_DIR, 'config.json')}")
print(f"\nFinal accuracy : {config['final_acc']:.4f}")
print(f"Final val acc  : {config['final_val_acc']:.4f}")
print("\nDone. Reload the server to use the new translator.")

# ── quick sanity check ────────────────────────────────────────────────────────
print("\n── Quick inference test ───────────────────────────────────────────────")
sys.path.insert(0, os.path.join(ROOT))
from ml.translator import Translator
t = Translator()
t.load()
if t.loaded:
    tests = [
        ["moi", "manger"],
        ["tu", "vouloir", "boire"],
        ["nous", "apprendre", "aujourd hui"],
        ["moi", "non", "comprendre"],
        ["bonjour", "ami"],
        ["moi", "aimer", "chat"],
        ["tu", "aller", "avec", "ami"],
    ]
    for signs in tests:
        result = t.translate(signs)
        print(f"  {' '.join(signs):35s} → {result}")
else:
    print("  Translator could not load (check errors above).")
