"""
zs_gnn_train.py  —  Zero-shot LOSO training runner for SSVEPGraphNet.

Usage (from ZERO_SHOT/):
  # shape smoke-test only (no data needed):
  python3 zs_gnn_train.py --smoke

  # single subject (S01) at sl=0.5:
  python3 zs_gnn_train.py --sl 0.5 --subjects 1

  # three subjects to check fold reproducibility:
  python3 zs_gnn_train.py --sl 0.5 --subjects 1 2 3

  # full run all 35 subjects (only after single-subject test passes):
  python3 zs_gnn_train.py --sl 0.5 1.0

Protocol (strict zero-shot LOSO):
  - Target subject is COMPLETELY held out (no training, no val)
  - 30 source subjects: training data
  -  4 source subjects: validation data (early stopping)
  - Model trained entirely on source subjects
  - Evaluated on all 240 target-subject trials
  - No target labels used anywhere

Checkpoint strategy:
  - One JSON per subject per sl, saved after every fold
  - Skips already-completed folds (safe to resume interrupted runs)
  - Example: results/gnn_s01_0p5.json

Metrics recorded per subject:
  subject, method, sl, acc, itr, best_val_acc, best_epoch,
  elapsed, preds, labels
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
import zs_baselines as B  # for itr()
from zs_gnn_model import SSVEPGraphNet


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ---------------------------------------------------------------------------
# constants  [MY CHOICE unless tagged]
# ---------------------------------------------------------------------------

METHOD_TAG   = "gnn"
EPOCHS       = 150       # max training epochs
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 20        # early stopping patience (val acc)
D_T          = 16        # temporal features per (band, node)
HIDDEN       = 64        # GCN hidden dim
KERNEL       = 16        # temporal conv kernel
DROPOUT      = 0.3
DEVICE       = (
    "cuda" if torch.cuda.is_available()
    else "mps"  if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------------
# ITR  (same formula as zs_runner.py)
# ---------------------------------------------------------------------------

def compute_itr(acc, T_samples, n=40):
    t = T_samples / C.FS + 0.5
    return B.itr(acc, n=n, t=t)


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------

def _to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype)


def make_loaders(fold: dict, sl: float, batch_size: int = BATCH_SIZE):
    """Build train / val DataLoaders and return raw test arrays.

    Returns
    -------
    train_loader : DataLoader   (bank-A fb_stand features, labels)
    val_loader   : DataLoader
    X_test_np    : (240, 3, 9, T) float32 numpy  — for eval loop
    y_test_np    : (240,)         int64   numpy
    """
    T = int(sl * C.FS)

    def load_split(sids):
        parts = [D.prep(s, sl) for s in sids]
        A  = np.concatenate([p["A"] for p in parts]).astype(np.float32)   # (N, 3, 9, T)
        y  = np.concatenate([p["y"] for p in parts])                       # (N,)
        return A, y

    X_tr, y_tr = load_split(fold["train"])
    X_va, y_va = load_split(fold["val"])
    X_te, y_te = load_split([fold["test"]])    # target subject

    # DataLoaders for train / val
    def make_dl(X, y, shuffle):
        ds = TensorDataset(_to_tensor(X), _to_tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(DEVICE == "cuda"))

    return (
        make_dl(X_tr, y_tr, shuffle=True),
        make_dl(X_va, y_va, shuffle=False),
        X_te.astype(np.float32),
        y_te,
    )


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        n += len(yb)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        n += len(yb)
    return correct / n


@torch.no_grad()
def predict_all(model, X_np, device, batch_size=BATCH_SIZE):
    """Run inference on a numpy array, return preds (N,) as numpy int."""
    model.eval()
    all_preds = []
    N = len(X_np)
    for i in range(0, N, batch_size):
        xb = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32).to(device)
        all_preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(all_preds)


# ---------------------------------------------------------------------------
# single-fold runner
# ---------------------------------------------------------------------------

def run_fold_gnn(sid: int, sl: float, device: str = DEVICE) -> dict:
    """Train and evaluate one LOSO fold for target subject `sid` at window `sl`.

    Returns the result dict (same schema as zs_runner.py baselines).
    """
    set_seed(C.SEED)
    T = int(sl * C.FS)
    fold = D.make_fold(sid)

    assert fold["test"] == sid
    assert sid not in fold["train"] and sid not in fold["val"], \
        "ZERO-SHOT VIOLATION: target subject in training set"

    print(
        f"  S{sid:02d} sl={sl} GNN  "
        f"train={len(fold['train'])} val={len(fold['val'])} subjects  "
        f"device={device}",
        flush=True,
    )

    # --- data ---
    t_data = time.time()
    train_loader, val_loader, X_test, y_test = make_loaders(fold, sl)
    print(
        f"  S{sid:02d} sl={sl} GNN  "
        f"data loaded in {time.time()-t_data:.1f}s  "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"test={len(X_test)}",
        flush=True,
    )

    # sanity: target not in source
    assert len(X_test) == 240, f"Expected 240 test trials, got {len(X_test)}"
    assert len(y_test) == 240

    # --- model ---
    model = SSVEPGraphNet(
        T=T, n_bands=3, n_nodes=9, n_classes=C.N_CLS,
        d_t=D_T, hidden=HIDDEN, kernel=KERNEL, dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    # --- training loop with early stopping ---
    t_train = time.time()
    best_val_acc  = 0.0
    best_epoch    = 0
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"    ep {epoch:3d}  loss={tr_loss:.4f}  "
                f"val_acc={val_acc:.4f}  best={best_val_acc:.4f}@{best_epoch}",
                flush=True,
            )

        if patience_cnt >= PATIENCE:
            print(f"    early stop at epoch {epoch} (patience={PATIENCE})", flush=True)
            break

    train_elapsed = round(time.time() - t_train, 2)

    # --- restore best model ---
    model.load_state_dict(best_state)

    # --- test inference ---
    preds = predict_all(model, X_test, device)

    acc     = float((preds == y_test).mean())
    itr_val = compute_itr(acc, T)

    print(
        f"  S{sid:02d} sl={sl} GNN  "
        f"acc={acc:.4f}  ITR={itr_val:.2f}  "
        f"best_val={best_val_acc:.4f}@ep{best_epoch}  "
        f"({train_elapsed:.1f}s)",
        flush=True,
    )

    return dict(
        subject      = sid,
        method       = METHOD_TAG,
        sl           = sl,
        acc          = acc,
        itr          = itr_val,
        best_val_acc = float(best_val_acc),
        best_epoch   = best_epoch,
        elapsed      = train_elapsed,
        preds        = preds.tolist(),
        labels       = y_test.tolist(),
    )


# ---------------------------------------------------------------------------
# checkpoint helpers
# ---------------------------------------------------------------------------

def ckpt_path(sid: int, sl: float) -> Path:
    tag = str(sl).replace(".", "p")
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    return C.RESULTS_DIR / f"{METHOD_TAG}_s{sid:02d}_{tag}.json"


def save_ckpt(r: dict):
    ckpt_path(r["subject"], r["sl"]).write_text(json.dumps(r, indent=1))


def already_done(sid: int, sl: float) -> bool:
    return ckpt_path(sid, sl).exists()


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def summarise(all_results, sl):
    rs = [r for r in all_results if r["sl"] == sl and r["method"] == METHOD_TAG]
    if not rs:
        return
    accs = np.array([r["acc"] for r in rs])
    itrs = np.array([r["itr"] for r in rs])
    n = len(accs)
    print(f"\nGNN  sl={sl}  n={n} subjects")
    print(f"  acc   mean={accs.mean():.4f}  std={accs.std():.4f}  se={accs.std()/np.sqrt(n):.4f}")
    print(f"  ITR   mean={itrs.mean():.2f}    std={itrs.std():.2f}    se={itrs.std()/np.sqrt(n):.2f}")
    print(f"  per-subject acc: {[round(a, 3) for a in accs.tolist()]}")


# ---------------------------------------------------------------------------
# smoke test (no real data required)
# ---------------------------------------------------------------------------

def smoke_test():
    """Tiny synthetic training loop to verify model + training code."""
    print("\n=== GNN smoke test (synthetic data, no real data required) ===\n")

    device = DEVICE
    sl     = 0.5
    T      = int(sl * C.FS)   # 125
    B_sz   = 8
    N_src  = 80    # tiny source set
    N_tgt  = 24    # tiny target set

    # synthetic (N, 3, 9, T) float32
    rng = np.random.RandomState(0)
    X_src = rng.randn(N_src, 3, 9, T).astype(np.float32)
    y_src = np.tile(np.arange(C.N_CLS), N_src // C.N_CLS + 1)[:N_src].astype(np.int64)
    X_tgt = rng.randn(N_tgt, 3, 9, T).astype(np.float32)
    y_tgt = np.tile(np.arange(C.N_CLS), N_tgt // C.N_CLS + 1)[:N_tgt].astype(np.int64)

    model = SSVEPGraphNet(T=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ds = TensorDataset(
        torch.tensor(X_src, dtype=torch.float32),
        torch.tensor(y_src, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=B_sz, shuffle=True)

    losses = []
    for ep in range(1, 6):
        loss = train_one_epoch(model, loader, optimizer, device)
        losses.append(loss)
        print(f"  ep {ep}  loss={loss:.4f}")

    assert all(np.isfinite(losses)), "NaN/Inf loss encountered"
    print(f"  losses finite: OK")

    # check loss generally decreasing (first vs last 3 epochs)
    # not guaranteed on random data but usually holds
    if losses[0] > losses[-1]:
        print(f"  loss decreased {losses[0]:.4f} -> {losses[-1]:.4f}: OK")
    else:
        print(f"  loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f}): note — expected on tiny random data")

    preds = predict_all(model, X_tgt, device)
    assert len(preds) == N_tgt, f"Expected {N_tgt} preds, got {len(preds)}"
    assert preds.min() >= 0 and preds.max() < C.N_CLS, "Prediction out of range"
    print(f"  predict_all: {len(preds)} preds, range [{preds.min()}, {preds.max()}]: OK")

    # gradient check
    model.train()
    x = torch.randn(4, 3, 9, T, device=device)
    y = torch.randint(0, 40, (4,), device=device)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    bad = [(n, p) for n, p in model.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all()]
    if bad:
        print(f"  FAIL — non-finite grads: {[n for n, _ in bad]}")
        import sys; sys.exit(1)
    print(f"  all gradients finite: OK")

    print("\nSmoke test PASSED.\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl",       nargs="+", type=float, default=[0.5])
    ap.add_argument("--subjects", nargs="+", type=int,   default=C.SUBJECTS)
    ap.add_argument("--smoke",    action="store_true",   help="Run synthetic smoke test only")
    ap.add_argument("--resume",   action="store_true",   help="Skip already-completed folds")
    ap.add_argument("--device",   type=str,              default=DEVICE)
    args = ap.parse_args()

    if args.smoke:
        smoke_test()
        import sys; sys.exit(0)

    print(f"\nDevice: {args.device}")
    print(f"Subjects: {args.subjects}")
    print(f"Windows: {args.sl}")
    print(f"Epochs: {EPOCHS}  Patience: {PATIENCE}  Batch: {BATCH_SIZE}\n")

    all_results = []

    for sl in args.sl:
        print(f"\n=== sl={sl} ===")
        for sid in args.subjects:
            if args.resume and already_done(sid, sl):
                print(f"  S{sid:02d} sl={sl} GNN  [skipping — checkpoint exists]")
                # load for summary
                r = json.loads(ckpt_path(sid, sl).read_text())
                all_results.append(r)
                continue

            r = run_fold_gnn(sid, sl, device=args.device)
            save_ckpt(r)
            all_results.append(r)

    print("\n=== SUMMARY ===")
    for sl in args.sl:
        summarise(all_results, sl)