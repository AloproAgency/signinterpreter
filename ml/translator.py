"""
Sentence translator: sign words → grammatically correct French.
Loads the trained seq2seq model and provides fast translation.
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sentence_model')


class Translator:
    def __init__(self):
        self.loaded = False
        self.encoder_model = None
        self.decoder_model = None
        self.sign_tok = None
        self.sent_tok = None
        self.config = None

    def load(self):
        """Load the trained seq2seq model."""
        config_path = os.path.join(MODEL_DIR, 'config.json')
        model_path = os.path.join(MODEL_DIR, 'model.keras')

        if not os.path.exists(config_path) or not os.path.exists(model_path):
            print("WARNING: Sentence model not found. Translation disabled.")
            return

        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)

            with open(os.path.join(MODEL_DIR, 'sign_tokenizer.pkl'), 'rb') as f:
                self.sign_tok = pickle.load(f)

            with open(os.path.join(MODEL_DIR, 'sentence_tokenizer.pkl'), 'rb') as f:
                self.sent_tok = pickle.load(f)

            model = tf.keras.models.load_model(model_path)
            self.encoder_model, self.decoder_model = self._build_inference_models(
                model, self.config['hidden_dim']
            )

            self.loaded = True
            print("Translator loaded.")
        except Exception as e:
            print(f"ERROR loading translator: {e}")
            self.loaded = False

    def _build_inference_models(self, model, hidden_dim):
        """Build encoder and decoder for step-by-step inference."""
        encoder_inputs = model.input[0]
        encoder_emb_layer = model.layers[2]
        encoder_lstm = model.get_layer('encoder_lstm')
        encoder_emb_out = encoder_emb_layer(encoder_inputs)
        enc_outputs, enc_h, enc_c = encoder_lstm(encoder_emb_out)
        encoder_model = Model(encoder_inputs, [enc_outputs, enc_h, enc_c])

        decoder_inputs = Input(shape=(1,), name='decoder_inf_input')
        decoder_state_h = Input(shape=(hidden_dim,), name='decoder_inf_state_h')
        decoder_state_c = Input(shape=(hidden_dim,), name='decoder_inf_state_c')
        encoder_outputs_inf = Input(shape=(None, hidden_dim), name='encoder_outputs_inf')

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

    def score_signs(self, signs_list, max_steps=None):
        """
        Implicit language-model score for a sign sequence.

        Runs the seq2seq greedy decoding and sums log-probabilities of the
        produced French tokens. Higher (less negative) means the model
        'understands' the sequence better — i.e. the sign sequence is more
        plausible under the training distribution.

        The returned score is length-normalized by number of input signs so
        that rerank comparisons across candidates are fair.
        """
        if not self.loaded or not signs_list:
            return 0.0

        signs_text = ' '.join(w.lower() for w in signs_list)

        try:
            sign_seq = self.sign_tok.texts_to_sequences([signs_text])
            sign_padded = pad_sequences(sign_seq, maxlen=self.config['max_sign_len'], padding='post')

            enc_outputs, state_h, state_c = self.encoder_model.predict(sign_padded, verbose=0)

            target_seq = np.array([[self.sent_tok.word_index.get('<start>', 1)]])
            log_prob = 0.0
            max_steps = max_steps or self.config['max_sent_len']

            for _ in range(max_steps):
                output, state_h, state_c = self.decoder_model.predict(
                    [target_seq, state_h, state_c, enc_outputs], verbose=0
                )
                probs = output[0, 0, :]
                token_id = int(np.argmax(probs))
                p = float(probs[token_id])
                log_prob += float(np.log(max(p, 1e-9)))

                if token_id == 0 or self.sent_tok.index_word.get(token_id, '') == '<end>':
                    break
                target_seq = np.array([[token_id]])

            return log_prob / max(1, len(signs_list))

        except Exception:
            return 0.0

    def prune_intruders(self, signs_list, min_gain=0.15, min_keep=1):
        """
        Greedily drop signs whose removal clearly improves the translation
        log-score. Returns (cleaned_signs, removed_indices).

        `min_gain` is the minimum per-sign log-prob improvement required to
        justify dropping a sign (higher = more conservative). `min_keep` is
        the minimum length to preserve (never go below it).

        Uses O(N²) score_signs calls in the worst case — only call at phrase
        finalization, not on every new sign.
        """
        if not self.loaded or len(signs_list) <= min_keep:
            return list(signs_list), []

        signs = list(signs_list)
        removed = []
        # Greedy: keep dropping as long as any single removal improves the score
        while len(signs) > min_keep:
            baseline = self.score_signs(signs)
            best_idx = -1
            best_score = baseline
            for i in range(len(signs)):
                candidate = signs[:i] + signs[i + 1:]
                if len(candidate) < min_keep:
                    continue
                s = self.score_signs(candidate)
                if s - baseline > min_gain and s > best_score:
                    best_score = s
                    best_idx = i
            if best_idx < 0:
                break
            removed.append(signs[best_idx])
            signs = signs[:best_idx] + signs[best_idx + 1:]
        return signs, removed

    def reorder_signs(self, signs_list, min_gain=0.1, max_iter=40):
        """
        Find the sign ordering that maximises the translator log-score.

        - For short sequences (len ≤ 6, ≤ 720 permutations) we enumerate all of
          them (still only a few hundred forward passes of a small LSTM).
        - Otherwise we fall back to repeated greedy *adjacent-swap* moves
          (bubble-sort with the score as comparator) for O(N²) complexity.

        Only accepts the new order if it beats the baseline by `min_gain`.
        Returns (best_sequence, swapped: bool).
        """
        if not self.loaded or len(signs_list) < 2:
            return list(signs_list), False

        baseline_score = self.score_signs(signs_list)

        if len(signs_list) <= 6:
            from itertools import permutations
            best = tuple(signs_list)
            best_score = baseline_score
            for perm in permutations(signs_list):
                if perm == tuple(signs_list):
                    continue
                s = self.score_signs(list(perm))
                if s > best_score:
                    best_score = s
                    best = perm
            if best_score - baseline_score > min_gain and tuple(best) != tuple(signs_list):
                return list(best), True
            return list(signs_list), False

        # Greedy adjacent swaps for longer sequences
        seq = list(signs_list)
        current = baseline_score
        improved = True
        iters = 0
        any_swap = False
        while improved and iters < max_iter:
            improved = False
            for i in range(len(seq) - 1):
                candidate = seq[:i] + [seq[i + 1], seq[i]] + seq[i + 2:]
                s = self.score_signs(candidate)
                if s - current > min_gain / 5:  # smaller per-step gain OK
                    seq = candidate
                    current = s
                    improved = True
                    any_swap = True
            iters += 1
        if any_swap and current - baseline_score > min_gain:
            return seq, True
        return list(signs_list), False

    def translate(self, signs_list):
        """
        Translate a list of sign words to a French sentence.
        Args: signs_list: list of strings, e.g. ["moi", "manger", "pomme"]
        Returns: string, e.g. "Je mange une pomme"
        """
        if not self.loaded or not signs_list:
            return ' '.join(signs_list)

        signs_text = ' '.join(w.lower() for w in signs_list)

        try:
            sign_seq = self.sign_tok.texts_to_sequences([signs_text])
            sign_padded = pad_sequences(sign_seq, maxlen=self.config['max_sign_len'], padding='post')

            enc_outputs, state_h, state_c = self.encoder_model.predict(sign_padded, verbose=0)

            target_seq = np.array([[self.sent_tok.word_index.get('<start>', 1)]])
            result = []

            for _ in range(self.config['max_sent_len']):
                output, state_h, state_c = self.decoder_model.predict(
                    [target_seq, state_h, state_c, enc_outputs], verbose=0
                )
                token_id = np.argmax(output[0, 0, :])
                word = self.sent_tok.index_word.get(token_id, '')

                if word == '<end>' or token_id == 0:
                    break

                result.append(word)
                target_seq = np.array([[token_id]])

            translated = ' '.join(result)
            # Capitalize first letter
            if translated:
                translated = translated[0].upper() + translated[1:]
            return translated

        except Exception as e:
            # Fallback: return raw signs
            return ' '.join(signs_list)


# Singleton
_translator = None

def get_translator():
    global _translator
    if _translator is None:
        _translator = Translator()
        _translator.load()
    return _translator
