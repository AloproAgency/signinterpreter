"""
LSTM seq2seq with Bahdanau attention for LSF sign sequence → French translation.

Architecture:
  Encoder : Embedding → 2-layer BiLSTM → hidden projection
  Decoder : Embedding → 2-layer LSTM + Bahdanau attention → Linear
"""
import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'seq2seq',
)

PAD, SOS, EOS = 0, 1, 2


class _Attention(nn.Module):
    def __init__(self, enc_hid: int, dec_hid: int):
        super().__init__()
        self.attn = nn.Linear(enc_hid * 2 + dec_hid, dec_hid)
        self.v    = nn.Linear(dec_hid, 1, bias=False)

    def forward(self, dec_hidden, enc_out):
        # dec_hidden : (B, dec_hid)
        # enc_out    : (B, T, enc_hid*2)
        T = enc_out.size(1)
        h = dec_hidden.unsqueeze(1).repeat(1, T, 1)       # (B, T, dec_hid)
        energy = torch.tanh(self.attn(torch.cat([h, enc_out], dim=2)))
        return F.softmax(self.v(energy).squeeze(2), dim=1) # (B, T)


class Encoder(nn.Module):
    def __init__(self, src_vocab: int, emb_dim: int, hid: int, n_layers: int, dropout: float):
        super().__init__()
        self.emb  = nn.Embedding(src_vocab, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hid, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.)
        self.fc   = nn.Linear(hid * 2, hid)
        self.drop = nn.Dropout(dropout)

    def forward(self, src, src_len):
        emb    = self.drop(self.emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_len.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        out_packed, (h, c) = self.lstm(packed)
        enc_out, _         = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        # Merge bidirectional top layer → init decoder hidden
        h_top = torch.tanh(self.fc(torch.cat([h[-2], h[-1]], dim=1)))  # (B, hid)
        return enc_out, h_top


class Decoder(nn.Module):
    def __init__(self, tgt_vocab: int, emb_dim: int, hid: int,
                 enc_hid: int, n_layers: int, dropout: float):
        super().__init__()
        self.emb     = nn.Embedding(tgt_vocab, emb_dim, padding_idx=PAD)
        self.attn    = _Attention(enc_hid, hid)
        self.lstm    = nn.LSTM(emb_dim + enc_hid * 2, hid, n_layers,
                               batch_first=True, dropout=dropout if n_layers > 1 else 0.)
        self.fc_out  = nn.Linear(hid + enc_hid * 2 + emb_dim, tgt_vocab)
        self.drop    = nn.Dropout(dropout)

    def forward(self, tgt_tok, dec_hidden, dec_cell, enc_out):
        # tgt_tok : (B,)
        emb     = self.drop(self.emb(tgt_tok.unsqueeze(1)))         # (B,1,E)
        a       = self.attn(dec_hidden[-1], enc_out)                 # (B,T)
        context = torch.bmm(a.unsqueeze(1), enc_out)                 # (B,1,enc*2)
        lstm_in = torch.cat([emb, context], dim=2)                   # (B,1,E+enc*2)
        out, (h, c) = self.lstm(lstm_in, (dec_hidden, dec_cell))
        pred    = self.fc_out(torch.cat([out, context, emb], dim=2)) # (B,1,V)
        return pred.squeeze(1), h, c                                  # (B,V), (L,B,H), (L,B,H)


class Seq2SeqModel(nn.Module):
    def __init__(self, src_vocab, tgt_vocab,
                 emb_dim=64, hid=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(src_vocab, emb_dim, hid, n_layers, dropout)
        self.decoder = Decoder(tgt_vocab, emb_dim, hid, hid, n_layers, dropout)
        self.tgt_vocab = tgt_vocab

    def forward(self, src, src_len, tgt, teacher_forcing=0.5):
        B, T_tgt = tgt.shape
        outputs  = torch.zeros(B, T_tgt, self.tgt_vocab, device=src.device)
        enc_out, h = self.encoder(src, src_len)
        # Init cell state to zeros, repeat for n_layers
        c   = torch.zeros_like(h).unsqueeze(0).repeat(self.decoder.lstm.num_layers, 1, 1)
        h   = h.unsqueeze(0).repeat(self.decoder.lstm.num_layers, 1, 1)
        tok = tgt[:, 0]          # SOS
        for t in range(1, T_tgt):
            out, h, c    = self.decoder(tok, h, c, enc_out)
            outputs[:, t] = out
            tok = tgt[:, t] if torch.rand(1).item() < teacher_forcing else out.argmax(1)
        return outputs


class Seq2SeqEngine:
    """Inference engine for the trained seq2seq translation model."""

    MAX_LEN = 20

    def __init__(self):
        self._model    = None
        self._src_w2i  = {}
        self._tgt_i2w  = {}
        self._device   = None
        self.loaded    = False

    def load(self) -> bool:
        cfg_path   = os.path.join(_MODEL_DIR, 'config.json')
        model_path = os.path.join(_MODEL_DIR, 'model.pt')
        vocab_path = os.path.join(_MODEL_DIR, 'vocab.json')
        if not all(os.path.exists(p) for p in (cfg_path, model_path, vocab_path)):
            return False
        try:
            with open(cfg_path)  as f: cfg   = json.load(f)
            with open(vocab_path) as f: vocab = json.load(f)
            device = (
                torch.device('mps')  if torch.backends.mps.is_available()
                else torch.device('cpu')
            )
            model = Seq2SeqModel(
                src_vocab = cfg['src_vocab'],
                tgt_vocab = cfg['tgt_vocab'],
                emb_dim   = cfg.get('emb_dim', 64),
                hid       = cfg.get('hid', 128),
                n_layers  = cfg.get('n_layers', 2),
                dropout   = 0.0,
            )
            model.load_state_dict(torch.load(model_path, map_location=device,
                                             weights_only=True))
            model.eval()
            self._model   = model.to(device)
            self._device  = device
            self._src_w2i = vocab['src_w2i']
            self._tgt_i2w = {int(k): v for k, v in vocab['tgt_i2w'].items()}
            self.loaded   = True
            return True
        except Exception as e:
            print(f'[Seq2Seq] load error: {e}')
            return False

    @torch.no_grad()
    def translate(self, signs: list[str]) -> str:
        if not self.loaded:
            return ' '.join(signs)
        src = [self._src_w2i.get(s, self._src_w2i.get('<UNK>', 0)) for s in signs]
        src_t  = torch.tensor([src], dtype=torch.long, device=self._device)
        src_len = torch.tensor([len(src)], dtype=torch.long)
        enc_out, h = self._model.encoder(src_t, src_len)
        c   = torch.zeros_like(h).unsqueeze(0).repeat(self._model.decoder.lstm.num_layers, 1, 1)
        h   = h.unsqueeze(0).repeat(self._model.decoder.lstm.num_layers, 1, 1)
        tok = torch.tensor([SOS], device=self._device)
        words = []
        for _ in range(self.MAX_LEN):
            out, h, c = self._model.decoder(tok, h, c, enc_out)
            tok = out.argmax(1)
            word = self._tgt_i2w.get(tok.item(), '')
            if word == '<EOS>':
                break
            if word and word != '<PAD>':
                words.append(word)
        sentence = ' '.join(words)
        return sentence[0].upper() + sentence[1:] if sentence else ''
