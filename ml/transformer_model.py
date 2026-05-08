"""
LSF→French Transformer encoder-decoder.

Model definition shared by the training script and the inference wrapper.
Architecture: Pre-LN Transformer, d_model=128, 4 heads, 3+3 layers, ffn=256.
~1.1M parameters — fast on M1 MPS.
"""
import json, math, os
import torch
import torch.nn as nn

PAD, UNK, BOS, EOS = 0, 1, 2, 3

_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'transformer_model',
)


def tokenize_lsf(s: str) -> list:
    """Split LSF sign string, keeping multi-word signs ('au revoir', 'aujourd hui') intact."""
    s = s.strip().lower()
    for mw in ('au revoir', 'aujourd hui'):
        s = s.replace(mw, mw.replace(' ', '\x00'))
    return [t.replace('\x00', ' ') for t in s.split()]


class _PosEnc(nn.Module):
    def __init__(self, d: int, maxlen: int = 256, drop: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe  = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, dtype=torch.float) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):   # x: (B, T, d)
        return self.drop(x + self.pe[:, :x.size(1)])


class LSFTransformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d         = cfg['d_model']
        self.d    = d
        drop      = cfg.get('dropout', 0.1)
        self.src_emb = nn.Embedding(cfg['src_vocab_size'], d, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(cfg['tgt_vocab_size'], d, padding_idx=PAD)
        self.pos     = _PosEnc(d, drop=drop)
        self.tf      = nn.Transformer(
            d_model            = d,
            nhead              = cfg['nhead'],
            num_encoder_layers = cfg['enc_layers'],
            num_decoder_layers = cfg['dec_layers'],
            dim_feedforward    = cfg['ffn_dim'],
            dropout            = drop,
            batch_first        = True,
            norm_first         = True,      # Pre-LN: more stable training
        )
        self.proj = nn.Linear(d, cfg['tgt_vocab_size'])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.src_emb.weight, std=self.d ** -0.5)
        nn.init.normal_(self.tgt_emb.weight, std=self.d ** -0.5)

    def _embed_src(self, x):
        return self.pos(self.src_emb(x) * self.d ** 0.5)

    def _embed_tgt(self, x):
        return self.pos(self.tgt_emb(x) * self.d ** 0.5)

    def encode(self, src, src_pad):
        return self.tf.encoder(self._embed_src(src), src_key_padding_mask=src_pad)

    def decode(self, tgt, mem, tgt_mask, tgt_pad, src_pad):
        return self.tf.decoder(
            self._embed_tgt(tgt), mem,
            tgt_mask                = tgt_mask,
            tgt_key_padding_mask    = tgt_pad,
            memory_key_padding_mask = src_pad,
        )

    def forward(self, src, tgt, src_pad, tgt_pad, tgt_mask):
        mem = self.encode(src, src_pad)
        out = self.decode(tgt, mem, tgt_mask, tgt_pad, src_pad)
        return self.proj(out)


class TransformerTranslator:
    """Loads the trained LSFTransformer from disk and translates sign lists to French."""

    def __init__(self):
        self._model     = None
        self._src_vocab = None
        self._tgt_itos  = None
        self._device    = None
        self._max_len   = 45

    def load(self) -> bool:
        paths = [os.path.join(_MODEL_DIR, f) for f in ('config.json', 'vocab.json', 'model.pt')]
        if not all(os.path.exists(p) for p in paths):
            return False
        try:
            with open(paths[0]) as f: cfg   = json.load(f)
            with open(paths[1]) as f: vocab = json.load(f)
            device = (
                torch.device('mps')  if torch.backends.mps.is_available()
                else torch.device('cpu')
            )
            self._device    = device
            self._src_vocab = vocab['src']
            self._tgt_itos  = {int(i): t for t, i in vocab['tgt'].items()}
            self._max_len   = cfg.get('max_tgt', 40) + 10
            model = LSFTransformer(cfg)
            model.load_state_dict(torch.load(paths[2], map_location=device))
            model.eval()
            self._model = model.to(device)
            return True
        except Exception as e:
            print(f'[TransformerTranslator] load error: {e}')
            return False

    @torch.no_grad()
    def translate(self, signs_list: list) -> str:
        toks = tokenize_lsf(' '.join(s.lower() for s in signs_list))
        src  = torch.tensor(
            [[self._src_vocab.get(t, UNK) for t in toks]],
            device=self._device,
        )
        src_pad = (src == PAD)
        mem  = self._model.encode(src, src_pad)
        out  = torch.tensor([[BOS]], device=self._device)

        for _ in range(self._max_len):
            T        = out.size(1)
            tgt_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=self._device), diagonal=1
            )
            tgt_pad  = torch.zeros(1, T, dtype=torch.bool, device=self._device)
            dec      = self._model.decode(out, mem, tgt_mask, tgt_pad, src_pad)
            nxt      = self._model.proj(dec[:, -1]).argmax(-1).item()
            if nxt == EOS:
                break
            out = torch.cat([out, torch.tensor([[nxt]], device=self._device)], dim=1)

        words = [self._tgt_itos.get(i, '') for i in out[0, 1:].tolist()]
        raw   = ' '.join(w for w in words if w)
        return raw[0].upper() + raw[1:] if raw else ''
