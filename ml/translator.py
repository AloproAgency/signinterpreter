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
