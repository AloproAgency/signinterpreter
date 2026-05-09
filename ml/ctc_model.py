"""
BiLSTM CTC model for continuous sign recognition.

Architecture: 2-layer bidirectional LSTM → linear projection → CTC.
Blank token index = n_classes (appended after real sign classes).
"""
import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'ctc_model',
)


class CTCSignModel(nn.Module):

    def __init__(self, input_dim: int, n_classes: int,
                 hidden: int = 256, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.blank    = n_classes
        self.n_classes = n_classes
        self.lstm = nn.LSTM(
            input_dim, hidden,
            num_layers    = n_layers,
            bidirectional = True,
            dropout       = dropout if n_layers > 1 else 0.0,
            batch_first   = True,
        )
        self.proj = nn.Linear(hidden * 2, n_classes + 1)

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        x       : (B, T, D)
        lengths : (B,) actual frame counts (optional — enables pack/unpack)
        returns : (T, B, C) log-softmax probabilities
        """
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                         enforce_sorted=True)
            out_packed, _ = self.lstm(packed)
            out, _        = pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, _ = self.lstm(x)

        logits = self.proj(out)                              # (B, T, C+1)
        return F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, B, C+1)


def ctc_greedy_decode(log_probs: np.ndarray, blank: int) -> list[int]:
    """
    Greedy CTC decoder.
    log_probs : (T, C) numpy array
    Returns decoded label indices (blanks and consecutive duplicates removed).
    """
    if len(log_probs) == 0:
        return []
    best = np.argmax(log_probs, axis=-1)          # (T,)
    collapsed = [int(best[0])]
    for t in range(1, len(best)):
        v = int(best[t])
        if v != collapsed[-1]:
            collapsed.append(v)
    return [x for x in collapsed if x != blank]


class CTCEngine:
    """Loads and runs the trained CTC model for batch inference."""

    def __init__(self):
        self._model   = None
        self._classes = None
        self._device  = None
        self.loaded   = False

    def load(self) -> bool:
        cfg_path   = os.path.join(_MODEL_DIR, 'config.json')
        model_path = os.path.join(_MODEL_DIR, 'model.pt')
        cls_path   = os.path.join(_MODEL_DIR, 'classes.json')
        if not all(os.path.exists(p) for p in (cfg_path, model_path, cls_path)):
            return False
        try:
            with open(cfg_path)  as f: cfg     = json.load(f)
            with open(cls_path)  as f: classes = json.load(f)
            device = (
                torch.device('mps')  if torch.backends.mps.is_available()
                else torch.device('cpu')
            )
            model = CTCSignModel(
                input_dim = cfg['input_dim'],
                n_classes = cfg['n_classes'],
                hidden    = cfg['hidden'],
                n_layers  = cfg['n_layers'],
                dropout   = 0.0,
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            self._model   = model.to(device)
            self._device  = device
            self._classes = classes
            self.loaded   = True
            return True
        except Exception as e:
            print(f'[CTCEngine] load error: {e}')
            return False

    @torch.no_grad()
    def predict_sequence(self, frames: np.ndarray) -> list[str]:
        """frames: (T, D) numpy float32 → list of sign words in order."""
        if not self.loaded:
            return []
        x         = torch.tensor(frames[None], dtype=torch.float32, device=self._device)
        log_probs = self._model(x)[:, 0, :].cpu().numpy()  # (T, C+1)
        indices   = ctc_greedy_decode(log_probs, blank=len(self._classes))
        return [self._classes[i] for i in indices]
