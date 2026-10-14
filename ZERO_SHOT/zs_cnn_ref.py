"""
zs_cnn_ref.py  —  CCA-Net released CNN (x-branch only) on our LOSO folds.

Reproduces the official training recipe:
  - All 34 source subjects, no validation set, no early stopping
  - 100 epochs, Adam lr=2e-4, batch=128, last-epoch model, seed=41

Usage (from ZERO_SHOT/):
  python3 zs_cnn_ref.py --sl 0.5 --subjects 1          # single subject test
  python3 zs_cnn_ref.py --sl 0.5 1.0 --resume          # full run, resumable
  python3 zs_cnn_ref.py --smoke                         # shape + gradient check only

Results saved to: results/cnnref_sXX_0pY.json
Same schema as GNN results for direct comparison.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import zs_config as C
import zs_data as D
from zs_gnn_train import set_seed, predict_all, compute_itr, DEVICE

# ---------------------------------------------------------------------------
# constants  [REPO] unless tagged
# ---------------------------------------------------------------------------

METHOD_TAG = "cnn_ref"
EPOCHS     = 100        # [REPO] config.n_epochs
BATCH_SIZE = 128        # [REPO]
LR         = 2e-4       # [REPO] Adam lr in model.py


# ---------------------------------------------------------------------------
# CNN architecture  [REPO] Siamese x-branch (only branch returned by forward)
#
# Input : (B, 3, 9, T)   bank-A fb_stand features
# Output: (B, 40)        unnormalised logits
#
# c1: 1x1 conv, 3 bands -> 1        (B,3,9,T) -> (B,1,9,T)
# c2: spatial conv (9,1), 1->120    (B,1,9,T) -> (B,120,1,T)
# d1: Dropout2d(0.1)
# c3: temporal stride-2 conv (1,2)  (B,120,1,T) -> (B,120,1,T//2)
# d2: Dropout2d(0.1)
# c4: temporal conv (1,10)          (B,120,1,T//2) -> (B,120,1,T//2)
# d3: Dropout2d(0.95)               [REPO] config.dropout=0.95
# fc: flatten -> Linear(120*T//2, 40)
# ---------------------------------------------------------------------------

class CNNRef(nn.Module):
    def __init__(self, T: int, n_bands: int = 3, n_ch: int = 9, n_cls: int = 40):
        super().__init__()
        self.T = T
        self.c1 = nn.Conv2d(n_bands, 1,   kernel_size=(1, 1))
        self.c2 = nn.Conv2d(1,      120,  kernel_size=(n_ch, 1))
        self.d1 = nn.Dropout2d(p=0.1)
        self.c3 = nn.Conv2d(120,    120,  kernel_size=(1, 2), stride=(1, 2))
        self.d2 = nn.Dropout2d(p=0.1)
        self.c4 = nn.Conv2d(120,    120,  kernel_size=(1, 10), padding=(0, 4))
        self.d3 = nn.Dropout2d(p=0.95)   # [REPO] config.dropout
        # infer flat size with a dummy pass to avoid hardcoding T-dependent math
        with torch.no_grad():
            dummy = torch.zeros(1, n_bands, n_ch, T)
            flat  = self._features(dummy).shape[1]
        self.fc = nn.Linear(flat, n_cls)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.d1(F.relu(self.c2(self.c1(x))))
        x = self.d2(F.relu(self.c3(x)))
        x = self.d3(F.relu(self.c4(x)))
        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self._features(x))


# ---------------------------------------------------------------------------
# single-fold runner
# ---------------------------------------------------------------------------

def run_fold_cnn(sid: int, sl: float, device: str = DEVICE) -> dict:
    set_seed(C.SEED)
    T    = int(sl * C.FS)
    fold = D.make_fold(sid)

    assert fold["test"] == sid
    assert sid not in fold["train"] + fold["val"], \
        "ZERO-SHOT VIOLATION: target in source pool"

    # all 34 sources, no val split  [REPO]
    src_ids = fold["train"] + fold["val"]
    print(f"  S{sid:02d} sl={sl} CNN_ref  {len(src_ids)} source subjects  device={device}",
          flush=True)

    parts = [D.prep(s, sl) for s in src_ids]
    X_src = np.concatenate([p["A"] for p in parts]).astype(np.float32)
    y_src = np.concatenate([p["y"] for p in parts])

    te    = D.prep(sid, sl)
    X_te  = te["A"].astype(np.float32)
    y_te  = te["y"]

    print(f"  S{sid:02d} sl={sl} CNN_ref  train={len(X_src)}  test={len(X_te)}", flush=True)

    model = CNNRef(T).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    ds    = TensorDataset(torch.tensor(X_src), torch.tensor(y_src, dtype=torch.long))
    dl    = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, pin_memory=(device == "cuda"))

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(yb); n += len(yb)
        if ep % 25 == 0 or ep == 1:
            print(f"    ep {ep:3d}  loss={tot/n:.4f}", flush=True)

    elapsed = round(time.time() - t0, 1)

    preds = predict_all(model, X_te, device)
    acc   = float((preds == y_te).mean())
    model.eval()
    with torch.no_grad(): lg = np.concatenate([model(torch.as_tensor(np.asarray(X_te[i:i+64]), dtype=torch.float32).to(device)).cpu().numpy() for i in range(0, len(X_te), 64)])
    print(f'  logits check: argmax agrees with preds on {(lg.argmax(1) == preds).mean():.3f}', flush=True)
    np.savez(C.RESULTS_DIR / f"logits_cnnref_s{sid:02d}_{str(sl).replace('.', 'p')}.npz", logits=lg, y=y_te)
    itr   = compute_itr(acc, T)

    print(f"  S{sid:02d} sl={sl} CNN_ref  acc={acc:.4f}  ITR={itr:.2f}  ({elapsed}s)",
          flush=True)

    return dict(subject=sid, method=METHOD_TAG, sl=sl,
                acc=acc, itr=itr, elapsed=elapsed, seed=C.SEED,
                preds=preds.tolist(), labels=y_te.tolist())


# ---------------------------------------------------------------------------
# checkpoint helpers
# ---------------------------------------------------------------------------

def ckpt_path(sid: int, sl: float) -> Path:
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    return C.RESULTS_DIR / f"{METHOD_TAG}_s{sid:02d}_{str(sl).replace('.','p')}.json"

def already_done(sid, sl): return ckpt_path(sid, sl).exists()


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def summarise(results, sl):
    rs = [r for r in results if r["sl"] == sl]
    if not rs: return
    ac = np.array([r["acc"] for r in rs]); it = np.array([r["itr"] for r in rs]); n = len(ac)
    print(f"\nCNN_ref  sl={sl}  n={n} subjects")
    print(f"  acc  mean={ac.mean():.4f}  std={ac.std():.4f}  se={ac.std()/np.sqrt(n):.4f}")
    print(f"  ITR  mean={it.mean():.2f}   std={it.std():.2f}   se={it.std()/np.sqrt(n):.2f}")
    print(f"  per-subject acc: {[round(a,3) for a in ac.tolist()]}")


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------

def smoke_test():
    print("\n=== CNN_ref smoke test ===\n")
    device = DEVICE
    for sl, T in [(0.5, 125), (1.0, 250)]:
        m = CNNRef(T).to(device)
        x = torch.randn(4, 3, 9, T, device=device)
        out = m(x)
        assert out.shape == (4, 40), f"bad shape {out.shape}"
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  sl={sl}  T={T}  output shape {tuple(out.shape)}  params={n:,}  OK")
    # gradient check
    m = CNNRef(125).to(device); m.train()
    x = torch.randn(8, 3, 9, 125, device=device)
    y = torch.randint(0, 40, (8,), device=device)
    F.cross_entropy(m(x), y).backward()
    bad = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, f"bad grads: {bad}"
    print("  gradient check: OK\nSmoke test PASSED.\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl",       nargs="+", type=float, default=[0.5])
    ap.add_argument("--subjects", nargs="+", type=int,   default=C.SUBJECTS)
    ap.add_argument("--resume",   action="store_true")
    ap.add_argument("--smoke",    action="store_true")
    ap.add_argument("--device",   default=DEVICE)
    a = ap.parse_args()

    if a.smoke:
        smoke_test(); import sys; sys.exit(0)

    print(f"\nDevice: {a.device}")
    print(f"Subjects: {a.subjects}")
    print(f"Windows: {a.sl}")
    print(f"Epochs: {EPOCHS}  Batch: {BATCH_SIZE}  LR: {LR}  Seed: {C.SEED}\n")

    all_results = []
    for sl in a.sl:
        print(f"\n=== sl={sl} ===")
        for sid in a.subjects:
            p = ckpt_path(sid, sl)
            if a.resume and already_done(sid, sl):
                print(f"  S{sid:02d} sl={sl} CNN_ref  [skipping — checkpoint exists]")
                all_results.append(json.loads(p.read_text())); continue
            r = run_fold_cnn(sid, sl, a.device)
            p.write_text(json.dumps(r, indent=1))
            all_results.append(r)

    print("\n=== SUMMARY ===")
    for sl in a.sl:
        summarise([r for r in all_results if r["sl"] == sl], sl)
