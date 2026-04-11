#!/usr/bin/env python3
"""
Seq2Seq model with Attention for translating sign sequences to French sentences.
Architecture: LSTM Encoder + Attention + LSTM Decoder

Usage:
  Train:  python sentence_model.py --train --data data/training_pairs.csv
  Test:   python sentence_model.py --test "moi manger pomme"
"""

import os
import sys
import csv
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, Concatenate,
    Attention, Layer, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'sentence_model')


# ============================================================
# DATA PREPARATION
# ============================================================
def load_data(csv_path):
    """Load and prepare training data."""
    pairs = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            signs = row['signs'].strip().lower()
            sentence = row['sentence'].strip()
            if signs and sentence and len(signs.split()) >= 2:
                # Add start/end tokens to target
                sentence = '<start> ' + sentence + ' <end>'
                pairs.append((signs, sentence))

    print(f"Loaded {len(pairs)} pairs")
    return pairs


def create_tokenizers(pairs):
    """Create tokenizers for source (signs) and target (sentences)."""
    sign_texts = [p[0] for p in pairs]
    sentence_texts = [p[1] for p in pairs]

    # Source tokenizer (signs)
    sign_tokenizer = Tokenizer(filters='', oov_token='<unk>')
    sign_tokenizer.fit_on_texts(sign_texts)

    # Target tokenizer (French sentences)
    sentence_tokenizer = Tokenizer(filters='', oov_token='<unk>')
    sentence_tokenizer.fit_on_texts(sentence_texts)

    return sign_tokenizer, sentence_tokenizer


def prepare_sequences(pairs, sign_tok, sent_tok, max_sign_len, max_sent_len):
    """Convert text pairs to padded sequences."""
    sign_texts = [p[0] for p in pairs]
    sent_texts = [p[1] for p in pairs]

    # Encode
    sign_seqs = sign_tok.texts_to_sequences(sign_texts)
    sent_seqs = sent_tok.texts_to_sequences(sent_texts)

    # Pad
    encoder_input = pad_sequences(sign_seqs, maxlen=max_sign_len, padding='post')
    decoder_input = pad_sequences([s[:-1] for s in sent_seqs], maxlen=max_sent_len, padding='post')
    decoder_target = pad_sequences([s[1:] for s in sent_seqs], maxlen=max_sent_len, padding='post')

    return encoder_input, decoder_input, decoder_target


# ============================================================
# MODEL
# ============================================================
def build_model(sign_vocab_size, sent_vocab_size, max_sign_len, max_sent_len,
                embedding_dim=128, hidden_dim=256):
    """Build seq2seq model with attention."""

    # ENCODER
    encoder_inputs = Input(shape=(max_sign_len,), name='encoder_input')
    encoder_embedding = Embedding(sign_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

    # DECODER
    decoder_inputs = Input(shape=(max_sent_len,), name='decoder_input')
    decoder_embedding = Embedding(sent_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

    # ATTENTION
    attention = Attention(name='attention')
    context = attention([decoder_outputs, encoder_outputs])

    # CONCAT + OUTPUT
    decoder_combined = Concatenate()([decoder_outputs, context])
    decoder_dense = Dense(hidden_dim, activation='tanh', name='dense_tanh')(decoder_combined)
    decoder_dropout = Dropout(0.3)(decoder_dense)
    output_layer = Dense(sent_vocab_size, activation='softmax', name='output')(decoder_dropout)

    model = Model([encoder_inputs, decoder_inputs], output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# INFERENCE
# ============================================================
def build_inference_models(model, hidden_dim=256):
    """Build separate encoder and decoder models for inference."""

    # Encoder model
    encoder_inputs = model.input[0]
    encoder_lstm = model.get_layer('encoder_lstm')
    # Get encoder embedding layer
    encoder_emb_layer = model.layers[2]  # first embedding
    encoder_emb_out = encoder_emb_layer(encoder_inputs)
    enc_outputs, enc_h, enc_c = encoder_lstm(encoder_emb_out)
    encoder_model = Model(encoder_inputs, [enc_outputs, enc_h, enc_c])

    # Decoder model - new inputs for step-by-step decoding
    decoder_inputs = Input(shape=(1,), name='decoder_inf_input')
    decoder_state_h = Input(shape=(hidden_dim,), name='decoder_inf_state_h')
    decoder_state_c = Input(shape=(hidden_dim,), name='decoder_inf_state_c')
    encoder_outputs_inf = Input(shape=(None, hidden_dim), name='encoder_outputs_inf')

    # Reuse weights from trained model
    # Find decoder embedding by name pattern
    decoder_emb_layer = None
    for layer in model.layers:
        if 'embedding' in layer.name and layer != encoder_emb_layer:
            decoder_emb_layer = layer
            break
    if decoder_emb_layer is None:
        decoder_emb_layer = model.layers[4]

    decoder_emb = decoder_emb_layer(decoder_inputs)

    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_out, dec_h, dec_c = decoder_lstm(
        decoder_emb, initial_state=[decoder_state_h, decoder_state_c]
    )

    attention_layer = model.get_layer('attention')
    context = attention_layer([decoder_out, encoder_outputs_inf])

    concat = Concatenate()([decoder_out, context])
    dense_out = model.get_layer('dense_tanh')(concat)
    output = model.get_layer('output')(dense_out)

    decoder_model = Model(
        [decoder_inputs, decoder_state_h, decoder_state_c, encoder_outputs_inf],
        [output, dec_h, dec_c]
    )

    return encoder_model, decoder_model


def translate(signs_text, encoder_model, decoder_model, sign_tok, sent_tok,
              max_sign_len, max_sent_len):
    """Translate a sign sequence to a French sentence."""
    # Encode input
    sign_seq = sign_tok.texts_to_sequences([signs_text.lower()])
    sign_padded = pad_sequences(sign_seq, maxlen=max_sign_len, padding='post')

    enc_outputs, state_h, state_c = encoder_model.predict(sign_padded, verbose=0)

    # Decode
    target_seq = np.array([[sent_tok.word_index.get('<start>', 1)]])
    result = []

    for _ in range(max_sent_len):
        output, state_h, state_c = decoder_model.predict(
            [target_seq, state_h, state_c, enc_outputs], verbose=0
        )

        token_id = np.argmax(output[0, 0, :])
        word = sent_tok.index_word.get(token_id, '<unk>')

        if word == '<end>' or token_id == 0:
            break

        result.append(word)
        target_seq = np.array([[token_id]])

    return ' '.join(result)


# ============================================================
# TRAINING
# ============================================================
def train(csv_path):
    """Train the seq2seq model."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    pairs = load_data(csv_path)
    np.random.shuffle(pairs)

    # Create tokenizers
    sign_tok, sent_tok = create_tokenizers(pairs)

    sign_vocab_size = len(sign_tok.word_index) + 1
    sent_vocab_size = len(sent_tok.word_index) + 1

    # Calculate max lengths
    max_sign_len = max(len(p[0].split()) for p in pairs)
    max_sent_len = max(len(p[1].split()) for p in pairs) - 1  # -1 for shifted target

    # Cap to reasonable lengths
    max_sign_len = min(max_sign_len, 15)
    max_sent_len = min(max_sent_len, 30)

    print(f"Sign vocab: {sign_vocab_size}")
    print(f"Sentence vocab: {sent_vocab_size}")
    print(f"Max sign length: {max_sign_len}")
    print(f"Max sentence length: {max_sent_len}")

    # Prepare sequences
    encoder_input, decoder_input, decoder_target = prepare_sequences(
        pairs, sign_tok, sent_tok, max_sign_len, max_sent_len
    )
    decoder_target = np.expand_dims(decoder_target, -1)

    print(f"Encoder input shape: {encoder_input.shape}")
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Decoder target shape: {decoder_target.shape}")

    # Split
    n = len(encoder_input)
    split = int(n * 0.9)
    train_enc, val_enc = encoder_input[:split], encoder_input[split:]
    train_dec_in, val_dec_in = decoder_input[:split], decoder_input[split:]
    train_dec_out, val_dec_out = decoder_target[:split], decoder_target[split:]

    print(f"Train: {split}, Val: {n - split}")

    # Build model
    model = build_model(sign_vocab_size, sent_vocab_size, max_sign_len, max_sent_len)
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.keras'),
            monitor='val_loss', save_best_only=True, verbose=1
        ),
    ]

    # Train
    print("\nTraining...")
    history = model.fit(
        [train_enc, train_dec_in], train_dec_out,
        validation_data=([val_enc, val_dec_in], val_dec_out),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    # Save model + tokenizers + config
    model.save(os.path.join(MODEL_DIR, 'model.keras'))

    config = {
        'sign_vocab_size': sign_vocab_size,
        'sent_vocab_size': sent_vocab_size,
        'max_sign_len': max_sign_len,
        'max_sent_len': max_sent_len,
        'embedding_dim': 128,
        'hidden_dim': 256,
    }
    with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(MODEL_DIR, 'sign_tokenizer.pkl'), 'wb') as f:
        pickle.dump(sign_tok, f)

    with open(os.path.join(MODEL_DIR, 'sentence_tokenizer.pkl'), 'wb') as f:
        pickle.dump(sent_tok, f)

    print(f"\nModel saved to {MODEL_DIR}")

    # Test
    print("\n=== Test translations ===")
    test_inputs = [
        "moi manger pomme",
        "hier moi aller ecole",
        "toi vouloir aide",
        "demain nous travailler",
        "moi pas comprendre",
        "ou toi habiter",
        "maman cuisiner bon",
        "moi aimer famille",
    ]

    # Build inference models
    encoder_model, decoder_model = build_inference_models(model)

    for signs in test_inputs:
        result = translate(signs, encoder_model, decoder_model,
                          sign_tok, sent_tok, max_sign_len, max_sent_len)
        print(f"  {signs:30s} → {result}")


def test(signs_text):
    """Test translation with a trained model."""
    # Load config
    with open(os.path.join(MODEL_DIR, 'config.json'), 'r') as f:
        config = json.load(f)

    with open(os.path.join(MODEL_DIR, 'sign_tokenizer.pkl'), 'rb') as f:
        sign_tok = pickle.load(f)

    with open(os.path.join(MODEL_DIR, 'sentence_tokenizer.pkl'), 'rb') as f:
        sent_tok = pickle.load(f)

    # Load model
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'model.keras'))
    encoder_model, decoder_model = build_inference_models(model, config['hidden_dim'])

    result = translate(signs_text, encoder_model, decoder_model,
                      sign_tok, sent_tok, config['max_sign_len'], config['max_sent_len'])
    print(f"Signs: {signs_text}")
    print(f"French: {result}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', type=str, help='Sign sequence to translate')
    parser.add_argument('--data', default=os.path.join(BASE_DIR, 'data', 'training_pairs.csv'))
    args = parser.parse_args()

    if args.train:
        train(args.data)
    elif args.test:
        test(args.test)
    else:
        parser.print_help()
