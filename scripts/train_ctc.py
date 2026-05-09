#!/usr/bin/env python3
"""
Train BiLSTM CTC model on 30,000 synthetic sign sequences.
PyTorch + MPS (Apple Silicon).

Metric: Sign Accuracy = 1 - Sign Error Rate (Levenshtein on decoded sequences).
"""
import json, os, pickle, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_PATH = os.path.join(ROOT, 'data', 'ctc_sequences.pkl')
OUT_DIR  = os.path.join(ROOT, 'data', 'ctc_model')
os.makedirs(OUT_DIR, exist_ok=True)

import sys
sys.path.insert(0, ROOT)
from ml.ctc_model import CTCSignModel, ctc_greedy_decode

DEVICE = (
    torch.device('mps')  if torch.backends.mps.is_available()
    else torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu')
)
print(f"Device: {DEVICE}")

# CTCLoss not supported on MPS — must run on CPU there.
# On CUDA/CPU it runs natively on device (faster).
_CTC_ON_CPU = (DEVICE.type == 'mps')

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading ctc_sequences.pkl …")
with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)

classes   = data['classes']
seqs      = data['seqs']     # list of (T, D) float16 arrays
labels    = data['labels']   # list of list[int]
N_CLASSES = len(classes)
INPUT_DIM = int(seqs[0].shape[1])
print(f"  {len(seqs)} sequences  |  {N_CLASSES} classes  |  dim={INPUT_DIM}")

# ── Train / val split ──────────────────────────────────────────────────────────
random.seed(42)
idx   = list(range(len(seqs)))
random.shuffle(idx)
n_val     = max(300, int(0.10 * len(seqs)))
val_idx   = idx[:n_val]
train_idx = idx[n_val:]
print(f"  Train: {len(train_idx)}  |  Val: {len(val_idx)}")

# ── Dataset & collate ──────────────────────────────────────────────────────────
class CTCDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        j = self.indices[i]
        return seqs[j].astype('float32'), labels[j]


def collate_ctc(batch):
    # Sort descending by frame length (required for pack_padded_sequence)
    batch   = sorted(batch, key=lambda x: -len(x[0]))
    xs, ys  = zip(*batch)
    T_max   = max(len(x) for x in xs)
    D       = xs[0].shape[1]

    padded   = torch.zeros(len(xs), T_max, D)
    in_lens  = torch.tensor([len(x) for x in xs], dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, :len(x)] = torch.tensor(x)

    tgt      = torch.tensor([l for ls in ys for l in ls], dtype=torch.long)
    tgt_lens = torch.tensor([len(ls) for ls in ys],       dtype=torch.long)
    return padded, in_lens, tgt, tgt_lens


BATCH    = 32
train_dl = DataLoader(CTCDataset(train_idx), batch_size=BATCH, shuffle=True,
                      collate_fn=collate_ctc, num_workers=0)
val_dl   = DataLoader(CTCDataset(val_idx),   batch_size=BATCH, shuffle=False,
                      collate_fn=collate_ctc, num_workers=0)

# ── Model ──────────────────────────────────────────────────────────────────────
CFG = {
    'input_dim': INPUT_DIM,
    'n_classes': N_CLASSES,
    'hidden':    256,
    'n_layers':  2,
    'dropout':   0.3,
}
model    = CTCSignModel(**CFG).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {n_params:,} parameters  (blank={N_CLASSES})")

criterion = nn.CTCLoss(blank=N_CLASSES, reduction='mean', zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

EPOCHS   = 80
PATIENCE = 15
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5
)

# ── Sign Error Rate ────────────────────────────────────────────────────────────
def levenshtein(a: list, b: list) -> int:
    dp = list(range(len(b) + 1))
    for ai in a:
        ndp = [dp[0] + 1]
        for j, bi in enumerate(b):
            ndp.append(min(ndp[-1] + 1, dp[j + 1] + 1, dp[j] + (0 if ai == bi else 1)))
        dp = ndp
    return dp[-1]


@torch.no_grad()
def evaluate(loader):
    model.eval()
    tot_loss = tot_dist = tot_ref = 0
    for padded, in_lens, tgt, tgt_lens in loader:
        padded = padded.to(DEVICE)
        log_p  = model(padded, in_lens)
        # CTCLoss is not supported on MPS — always compute on CPU;
        # gradients flow back correctly through the device transfer.
        loss   = criterion(log_p.cpu() if _CTC_ON_CPU else log_p, tgt, in_lens, tgt_lens)
        tot_loss += loss.item()

        log_p_np = log_p.detach().cpu().numpy()
        offset   = 0
        for b in range(padded.size(0)):
            T   = in_lens[b].item()
            ref = tgt[offset: offset + tgt_lens[b].item()].tolist()
            offset += tgt_lens[b].item()
            hyp = ctc_greedy_decode(log_p_np[:T, b, :], blank=N_CLASSES)
            tot_dist += levenshtein(hyp, ref)
            tot_ref  += len(ref)

    avg_loss = tot_loss / len(loader)
    sign_acc = 1.0 - tot_dist / max(1, tot_ref)
    return avg_loss, sign_acc


# ── Training loop ──────────────────────────────────────────────────────────────
best_acc  = 0.0
wait      = 0
best_path = os.path.join(OUT_DIR, 'best_model.pt')

print(f"\nTraining up to {EPOCHS} epochs  (early-stop patience={PATIENCE}) …\n")
for ep in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    for padded, in_lens, tgt, tgt_lens in train_dl:
        padded = padded.to(DEVICE)
        log_p  = model(padded, in_lens)
        loss   = criterion(log_p.cpu() if _CTC_ON_CPU else log_p, tgt, in_lens, tgt_lens)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        tr_loss += loss.item()
    scheduler.step()

    vl_loss, vl_acc = evaluate(val_dl)
    lr_now = optimizer.param_groups[0]['lr']
    print(f"  {ep:3d}  train={tr_loss/len(train_dl):.4f}  "
          f"val={vl_loss:.4f}  sign_acc={vl_acc:.4f}  lr={lr_now:.2e}")

    if vl_acc > best_acc:
        best_acc = vl_acc
        torch.save(model.state_dict(), best_path)
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"\n  Early stop at epoch {ep}  (best sign_acc={best_acc:.4f})")
            break

# ── Save ───────────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
torch.save(model.state_dict(), os.path.join(OUT_DIR, 'model.pt'))

CFG['best_sign_acc'] = best_acc
with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump(CFG, f, indent=2)
with open(os.path.join(OUT_DIR, 'classes.json'), 'w') as f:
    json.dump(classes, f, ensure_ascii=False, indent=2)

print(f"\nSaved → {OUT_DIR}")
print(f"Best sign accuracy: {best_acc:.4f}")
print("\nDone. Restart the server to activate the CTC model.")
