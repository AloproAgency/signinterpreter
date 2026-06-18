#!/usr/bin/env python3
"""
Train LSTM seq2seq model on training_pairs.csv.
Input  : LSF sign sequence  (e.g. ["moi", "manger"])
Output : French sentence     (e.g. "Je mange.")
"""
import csv, json, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, 'data', 'training_pairs.csv')
OUT_DIR  = os.path.join(ROOT, 'data', 'seq2seq')
os.makedirs(OUT_DIR, exist_ok=True)

import sys
sys.path.insert(0, ROOT)
from ml.seq2seq_model import Seq2SeqModel, PAD, SOS, EOS

DEVICE = (
    torch.device('cuda') if torch.cuda.is_available()
    else torch.device('mps') if torch.backends.mps.is_available()
    else torch.device('cpu')
)
print(f"Device: {DEVICE}")

# ── Hyperparameters ────────────────────────────────────────────────────────────
EMB_DIM  = 64
HID      = 128
N_LAYERS = 2
DROPOUT  = 0.3
BATCH    = 64
EPOCHS   = 100
PATIENCE = 15
LR       = 1e-3

# ── Load & parse data ──────────────────────────────────────────────────────────
def parse_signs(raw: str) -> list[str]:
    toks = raw.strip().lower().split()
    out, i = [], 0
    while i < len(toks):
        if i+1 < len(toks) and toks[i] == 'au' and toks[i+1] == 'revoir':
            out.append('au revoir'); i += 2
        elif i+1 < len(toks) and toks[i] == "aujourd" and toks[i+1] == 'hui':
            out.append('aujourd hui'); i += 2
        else:
            out.append(toks[i]); i += 1
    return out

pairs = []
with open(CSV_PATH, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        signs   = parse_signs(row['signs'])
        words   = row['sentence'].strip().split()
        if signs and words:
            pairs.append((signs, words))

print(f"Pairs loaded: {len(pairs)}")

# ── Build vocabularies ─────────────────────────────────────────────────────────
src_words = sorted({s for signs, _ in pairs for s in signs})
tgt_words = sorted({w for _, words in pairs for w in words})

src_w2i = {'<PAD>': PAD, '<SOS>': SOS, '<EOS>': EOS, '<UNK>': 3}
for w in src_words:
    if w not in src_w2i:
        src_w2i[w] = len(src_w2i)

tgt_w2i = {'<PAD>': PAD, '<SOS>': SOS, '<EOS>': EOS}
for w in tgt_words:
    if w not in tgt_w2i:
        tgt_w2i[w] = len(tgt_w2i)
tgt_i2w = {i: w for w, i in tgt_w2i.items()}

SRC_VOCAB = len(src_w2i)
TGT_VOCAB = len(tgt_w2i)
print(f"Src vocab: {SRC_VOCAB}  |  Tgt vocab: {TGT_VOCAB}")

# ── Dataset ────────────────────────────────────────────────────────────────────
def encode_src(signs):
    return [SOS] + [src_w2i.get(s, src_w2i['<UNK>']) for s in signs] + [EOS]

def encode_tgt(words):
    return [SOS] + [tgt_w2i.get(w, tgt_w2i.get('<UNK>', PAD)) for w in words] + [EOS]

encoded = [(encode_src(s), encode_tgt(w)) for s, w in pairs]

random.seed(42)
random.shuffle(encoded)
n_val    = max(200, int(0.10 * len(encoded)))
val_data  = encoded[:n_val]
train_data = encoded[n_val:]
print(f"Train: {len(train_data)}  |  Val: {len(val_data)}")


class PairDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def collate(batch):
    srcs, tgts = zip(*batch)
    src_lens = torch.tensor([len(s) for s in srcs], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgts], dtype=torch.long)
    src_pad  = torch.zeros(len(srcs), max(len(s) for s in srcs), dtype=torch.long)
    tgt_pad  = torch.zeros(len(tgts), max(len(t) for t in tgts), dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, :len(s)] = torch.tensor(s)
        tgt_pad[i, :len(t)] = torch.tensor(t)
    return src_pad, src_lens, tgt_pad, tgt_lens


train_dl = DataLoader(PairDataset(train_data), batch_size=BATCH, shuffle=True,  collate_fn=collate)
val_dl   = DataLoader(PairDataset(val_data),   batch_size=BATCH, shuffle=False, collate_fn=collate)

# ── Model ──────────────────────────────────────────────────────────────────────
model = Seq2SeqModel(SRC_VOCAB, TGT_VOCAB, EMB_DIM, HID, N_LAYERS, DROPOUT).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"Seq2Seq: {n_params:,} parameters")

criterion = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# ── Eval ───────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0
    for src, src_len, tgt, _ in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        out = model(src, src_len, tgt, teacher_forcing=0.0)  # (B, T, V)
        loss = criterion(out[:, 1:].reshape(-1, TGT_VOCAB),
                         tgt[:, 1:].reshape(-1))
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def greedy_decode(signs: list[str]) -> str:
    model.eval()
    src = [SOS] + [src_w2i.get(s, src_w2i['<UNK>']) for s in signs] + [EOS]
    src_t   = torch.tensor([src], dtype=torch.long, device=DEVICE)
    src_len = torch.tensor([len(src)], dtype=torch.long)
    enc_out, h = model.encoder(src_t, src_len)
    c   = torch.zeros_like(h).unsqueeze(0).repeat(N_LAYERS, 1, 1)
    h   = h.unsqueeze(0).repeat(N_LAYERS, 1, 1)
    tok = torch.tensor([SOS], device=DEVICE)
    words = []
    for _ in range(20):
        out, h, c = model.decoder(tok, h, c, enc_out)
        tok  = out.argmax(1)
        word = tgt_i2w.get(tok.item(), '')
        if word == '<EOS>': break
        if word not in ('<PAD>', '<SOS>'): words.append(word)
    return ' '.join(words)

# ── Training ───────────────────────────────────────────────────────────────────
best_loss = float('inf')
wait      = 0
best_path = os.path.join(OUT_DIR, 'best_model.pt')

print(f"\nTraining {EPOCHS} epochs (patience={PATIENCE}) …\n")
for ep in range(1, EPOCHS + 1):
    model.train()
    tr_loss = 0.0
    tf = max(0.3, 1.0 - ep * 0.008)   # teacher forcing annealing
    for src, src_len, tgt, _ in train_dl:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        out  = model(src, src_len, tgt, teacher_forcing=tf)
        loss = criterion(out[:, 1:].reshape(-1, TGT_VOCAB),
                         tgt[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item()

    vl_loss = evaluate(val_dl)
    scheduler.step(vl_loss)
    lr_now = optimizer.param_groups[0]['lr']
    print(f"  {ep:3d}  train={tr_loss/len(train_dl):.4f}  val={vl_loss:.4f}  tf={tf:.2f}  lr={lr_now:.2e}")

    if vl_loss < best_loss:
        best_loss = vl_loss
        torch.save(model.state_dict(), best_path)
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"\n  Early stop at epoch {ep}  (best val={best_loss:.4f})")
            break

# ── Sample translations ────────────────────────────────────────────────────────
model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
print("\nSample translations:")
for signs, _ in random.sample(pairs, 8):
    pred = greedy_decode(signs)
    print(f"  {signs} → {pred}")

# ── Save ───────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(OUT_DIR, 'model.pt'))

cfg = {'src_vocab': SRC_VOCAB, 'tgt_vocab': TGT_VOCAB,
       'emb_dim': EMB_DIM, 'hid': HID, 'n_layers': N_LAYERS,
       'best_val_loss': round(best_loss, 6)}
with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump(cfg, f, indent=2)

vocab = {'src_w2i': src_w2i, 'tgt_i2w': {str(k): v for k, v in tgt_i2w.items()}}
with open(os.path.join(OUT_DIR, 'vocab.json'), 'w') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"\nSaved → {OUT_DIR}")
print(f"Best val loss: {best_loss:.4f}")
