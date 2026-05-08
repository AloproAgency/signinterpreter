#!/usr/bin/env python3
"""
Train a small Pre-LN Transformer for LSF→French translation.
PyTorch + MPS (Apple Silicon) — ~1.1M parameters.

Architecture (config below):
  Encoder: 3 × Pre-LN TransformerEncoderLayer (d=128, heads=4, ffn=256)
  Decoder: 3 × Pre-LN TransformerDecoderLayer
  ~1.1M parameters — trains in ~10 min on M1 MPS.

Usage:
    python scripts/train_transformer.py
"""
import csv, json, math, os, random, sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(ROOT, 'data', 'training_pairs.csv')
OUT_DIR  = os.path.join(ROOT, 'data', 'transformer_model')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from ml.transformer_model import LSFTransformer, tokenize_lsf, PAD, UNK, BOS, EOS

DEVICE = (
    torch.device('mps')  if torch.backends.mps.is_available()
    else torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu')
)
print(f"Device: {DEVICE}")

# ── Tokenize French (whitespace split + lowercase) ─────────────────────────────
def tokenize_fr(s: str) -> list:
    return s.strip().lower().split()

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading training_pairs.csv …")
src_seqs, tgt_seqs = [], []
with open(DATA_CSV, encoding='utf-8') as f:
    for row in csv.DictReader(f):
        src_seqs.append(tokenize_lsf(row['signs']))
        tgt_seqs.append(tokenize_fr(row['sentence']))

N = len(src_seqs)
print(f"  {N} pairs loaded")

# ── Build vocabularies ────────────────────────────────────────────────────────
_SPECIAL = ['<pad>', '<unk>', '<bos>', '<eos>']

def build_vocab(token_lists):
    vocab = {t: i for i, t in enumerate(_SPECIAL)}
    for lst in token_lists:
        for t in lst:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab

src_vocab = build_vocab(src_seqs)
tgt_vocab = build_vocab(tgt_seqs)
src_itos  = {i: t for t, i in src_vocab.items()}
tgt_itos  = {i: t for t, i in tgt_vocab.items()}
SRC_VOCAB = len(src_vocab)
TGT_VOCAB = len(tgt_vocab)
print(f"  Src vocab: {SRC_VOCAB}  |  Tgt vocab: {TGT_VOCAB}")

# ── Encode ────────────────────────────────────────────────────────────────────
def enc_src(toks): return [src_vocab.get(t, UNK) for t in toks]
def enc_tgt(toks): return [BOS] + [tgt_vocab.get(t, UNK) for t in toks] + [EOS]

src_enc = [enc_src(s) for s in src_seqs]
tgt_enc = [enc_tgt(s) for s in tgt_seqs]

MAX_SRC = max(len(s) for s in src_enc)
MAX_TGT = max(len(s) for s in tgt_enc)
print(f"  Max src len: {MAX_SRC}  |  Max tgt len: {MAX_TGT}")

# ── Train / val split ─────────────────────────────────────────────────────────
random.seed(42)
idx = list(range(N))
random.shuffle(idx)
n_val     = max(50, int(0.05 * N))
val_set   = set(idx[:n_val])
train_idx = [i for i in idx if i not in val_set]
val_idx   = [i for i in idx if i in val_set]
print(f"  Train: {len(train_idx)}  |  Val: {len(val_idx)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class Seq2SeqDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        j = self.indices[i]
        return src_enc[j], tgt_enc[j]

def collate(batch):
    srcs, tgts = zip(*batch)
    S = max(len(s) for s in srcs)
    T = max(len(t) for t in tgts)
    src_t = torch.zeros(len(batch), S, dtype=torch.long)
    tgt_t = torch.zeros(len(batch), T, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_t[i, :len(s)] = torch.tensor(s)
        tgt_t[i, :len(t)] = torch.tensor(t)
    return src_t, tgt_t, (src_t == PAD), (tgt_t == PAD)

BATCH = 64
train_dl = DataLoader(Seq2SeqDataset(train_idx), batch_size=BATCH, shuffle=True,  collate_fn=collate)
val_dl   = DataLoader(Seq2SeqDataset(val_idx),   batch_size=BATCH, shuffle=False, collate_fn=collate)

# ── Model ─────────────────────────────────────────────────────────────────────
CFG = {
    'd_model':       128,
    'nhead':           4,
    'ffn_dim':       256,
    'enc_layers':      3,
    'dec_layers':      3,
    'dropout':       0.1,
    'src_vocab_size': SRC_VOCAB,
    'tgt_vocab_size': TGT_VOCAB,
    'max_src':        MAX_SRC,
    'max_tgt':        MAX_TGT,
}

model    = LSFTransformer(CFG).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {n_params:,} parameters")

# ── Optimizer: Adam + warmup-cosine schedule ──────────────────────────────────
EPOCHS   = 150
LR_PEAK  = 5e-4
WARMUP   = 800          # gradient steps
PATIENCE = 25
SMOOTH   = 0.1

optimizer = torch.optim.Adam(
    model.parameters(), lr=LR_PEAK, betas=(0.9, 0.98), eps=1e-9
)
total_steps = EPOCHS * len(train_dl)

def lr_lambda(step):
    step = max(step, 1)
    if step < WARMUP:
        return step / WARMUP
    progress = (step - WARMUP) / max(1, total_steps - WARMUP)
    return max(0.02, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=SMOOTH)

def causal_mask(sz):
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=DEVICE), diagonal=1)

# ── Epoch pass ────────────────────────────────────────────────────────────────
def run_epoch(dl, train: bool):
    model.train(train)
    tot_loss = tot_ok = tot_n = 0
    with torch.set_grad_enabled(train):
        for src, tgt, src_pad, tgt_pad in dl:
            src, tgt         = src.to(DEVICE), tgt.to(DEVICE)
            src_pad, tgt_pad = src_pad.to(DEVICE), tgt_pad.to(DEVICE)
            dec_in   = tgt[:, :-1]
            dec_out  = tgt[:, 1:]
            dec_pad  = tgt_pad[:, :-1]
            T        = dec_in.size(1)
            logits   = model(src, dec_in, src_pad, dec_pad, causal_mask(T))
            loss     = criterion(logits.reshape(-1, TGT_VOCAB), dec_out.reshape(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            mask      = dec_out != PAD
            tot_ok   += (logits.argmax(-1)[mask] == dec_out[mask]).sum().item()
            tot_n    += mask.sum().item()
            tot_loss += loss.item() * mask.sum().item()
    return tot_loss / max(1, tot_n), tot_ok / max(1, tot_n)

# ── Training loop ─────────────────────────────────────────────────────────────
best_val  = 0.0
wait      = 0
best_path = os.path.join(OUT_DIR, 'best_model.pt')

print(f"\nTraining up to {EPOCHS} epochs  (early-stop patience={PATIENCE}) …\n")
for ep in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_dl, train=True)
    vl_loss, vl_acc = run_epoch(val_dl,   train=False)
    lr_now = optimizer.param_groups[0]['lr']
    print(f"  {ep:3d}  train={tr_acc:.4f}({tr_loss:.4f})  val={vl_acc:.4f}({vl_loss:.4f})  lr={lr_now:.2e}")

    if vl_acc > best_val:
        best_val = vl_acc
        torch.save(model.state_dict(), best_path)
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"\n  Early stop at epoch {ep}  (best val token acc = {best_val:.4f})")
            break

# ── Save ──────────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
torch.save(model.state_dict(), os.path.join(OUT_DIR, 'model.pt'))

CFG['best_val_token_acc'] = best_val
with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump(CFG, f, indent=2, ensure_ascii=False)
with open(os.path.join(OUT_DIR, 'vocab.json'), 'w') as f:
    json.dump({'src': src_vocab, 'tgt': tgt_vocab}, f, indent=2, ensure_ascii=False)

print(f"\nSaved → {OUT_DIR}")
print(f"Best val token accuracy: {best_val:.4f}")

# ── Greedy decode (used in the test below) ────────────────────────────────────
@torch.no_grad()
def greedy_decode(signs_str: str) -> str:
    model.eval()
    toks = tokenize_lsf(signs_str)
    src  = torch.tensor([[src_vocab.get(t, UNK) for t in toks]], device=DEVICE)
    src_pad = (src == PAD)
    mem  = model.encode(src, src_pad)
    out  = torch.tensor([[BOS]], device=DEVICE)
    for _ in range(MAX_TGT + 10):
        T        = out.size(1)
        tgt_mask = causal_mask(T)
        tgt_pad  = torch.zeros(1, T, dtype=torch.bool, device=DEVICE)
        dec      = model.decode(out, mem, tgt_mask, tgt_pad, src_pad)
        nxt      = model.proj(dec[:, -1]).argmax(-1).item()
        if nxt == EOS:
            break
        out = torch.cat([out, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    words = [tgt_itos.get(i, '') for i in out[0, 1:].tolist()]
    raw   = ' '.join(w for w in words if w)
    return raw[0].upper() + raw[1:] if raw else ''

# ── End-to-end test ───────────────────────────────────────────────────────────
from ml.translator import rule_translate

TESTS = [
    "moi manger",
    "tu boire",
    "nous apprendre",
    "moi non comprendre",
    "tu non manger",
    "moi vouloir apprendre",
    "moi aimer chat",
    "moi donner aide ami",
    "ami attendre moi",
    "moi manger avec ami",
    "nous apprendre aujourd hui",
    "moi non vouloir aide",
    "moi non manger aide bientot avec ami",
    "bonjour",
    "au revoir ami",
    "bebe manger",
    "chat aimer bebe",
    "aujourd hui moi aller avec ami",
    "bonjour moi manger aide bientot avec ami",
    "comment moi comprendre",
]

print("\n── End-to-end inference test ──────────────────────────────────────────")
ok = 0
for s in TESTS:
    toks   = tokenize_lsf(s)
    rule   = rule_translate(toks)
    neural = greedy_decode(s)
    match  = "✓" if neural == rule else "✗"
    print(f"  {match}  {s:42s} → {neural}")
    if neural == rule:
        ok += 1
    else:
        print(f"      (rule: {rule})")

print(f"\n{ok}/{len(TESTS)} exact matches with rule-based output")
print("\nDone. Restart the server to activate the new model.")
