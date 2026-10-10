"""
zs_gnn_v2c.py  —  Time-preserving GNN for zero-shot SSVEP (Benchmark dataset).

Identical to zs_gnn_v2c.py except:
  - H=64, D_T=16 restored (original larger model, 335,962 params for g2)
  - GNNv2_G3.conv_bn changed from BatchNorm1d to GroupNorm (consistent with G2)

All BatchNorm1d replaced with GroupNorm throughout for cross-platform
consistency between MPS (Mac) and CUDA (Linux/Windows).

Variants (--variant flag):
  g2  : 2-layer GCN with learned 9x9 adjacency (full GNN v2)
  g3  : no graph — temporal features flattened directly to FC (ablation)

Training recipe matches CNN_ref exactly for apples-to-apples comparison:
  - All 34 source subjects, no val set, no early stopping
  - 100 epochs, Adam lr=2e-4, batch=128, last-epoch model, seed=41

Results saved to: results/gnnv2c_{variant}_s{sid}_{sl}.json
  (separate from v2b results — old files are not touched)

Usage (from ZERO_SHOT/):
  python3 zs_gnn_v2c.py --smoke
  python3 zs_gnn_v2c.py --sl 0.5 --subjects 1 --variant g2 g3
  python3 zs_gnn_v2c.py --sl 0.5 1.0 --variant g2 g3 --resume
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
from zs_gnn_train import set_seed, compute_itr, DEVICE

# ---------------------------------------------------------------------------
# constants  [MY CHOICE unless tagged]
# ---------------------------------------------------------------------------

EPOCHS     = 100      # [REPO] matches CNN_ref / CCA-Net recipe
BATCH_SIZE = 128      # [REPO]
LR         = 2e-4     # [REPO]
D_T        = 16       # temporal features per (band, node) after conv
KERNEL     = 16       # temporal conv kernel width
HIDDEN     = 64       # GCN hidden dim (g2 only)
DROPOUT    = 0.3

VARIANTS   = ("g2", "g3")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sym_row_norm(W: torch.Tensor) -> torch.Tensor:
    """Symmetrise, softplus-positive, row-normalise. Same as GNN F1."""
    A = (W + W.T) / 2.0
    A = F.softplus(A)
    D_inv = 1.0 / A.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return A * D_inv


# ---------------------------------------------------------------------------
# G2: time-preserving GCN with GroupNorm (v2c change vs v2)
#
# Only change from zs_gnn_v2.py:
#   FastGCNLayer.bn  : BatchNorm1d(out_dim) -> GroupNorm(min(8,out_dim), out_dim)
#   FastGCNLayer.forward: 4-line BN reshape dance replaced by direct GroupNorm call
#     GroupNorm operates on (B, C, *) so (B, f_out, 9, T) works without reshape.
# ---------------------------------------------------------------------------

class FastGCNLayer(nn.Module):
    """GCN layer keeping T as a dimension — GroupNorm replaces BatchNorm1d.

    GroupNorm works directly on (B, f_out, 9, T) with no reshape needed,
    removing the (B*T, f, 9) permute/reshape overhead that slowed MPS.
    """

    def __init__(self, in_dim: int, out_dim: int, n_nodes: int = 9):
        super().__init__()
        self.A_raw = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1)
        self.W     = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias  = nn.Parameter(torch.zeros(out_dim))
        # GroupNorm: operates on (B, C, *) directly — no reshape needed
        self.bn    = nn.GroupNorm(num_groups=min(8, out_dim), num_channels=out_dim)
        self.act   = nn.ELU()
        nn.init.xavier_uniform_(self.W)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: (B, f_in, 9, T) -> (B, f_out, 9, T)"""
        A = _sym_row_norm(self.A_raw)                        # (9, 9)

        # graph mix: aggregate neighbour features across nodes
        H_agg = torch.einsum('bfnt,mn->bfmt', H, A)         # (B, f_in, 9, T)

        # linear: (B, f_in, 9, T) -> (B, f_out, 9, T)
        H_out = torch.einsum('bfnt,fo->bont', H_agg, self.W) + self.bias.view(1, -1, 1, 1)

        # GroupNorm works directly on (B, f_out, 9, T) — no reshape required
        H_out = self.bn(H_out)
        H_out = self.act(H_out)
        return H_out


class GNNv2_G2(nn.Module):
    """Time-preserving GNN with learned graph (variant g2).

    Input : (B, n_bands, n_nodes, T)
    Output: (B, n_classes)
    """

    def __init__(self, T: int, n_bands: int = 3, n_nodes: int = 9,
                 n_classes: int = 40, d_t: int = D_T,
                 hidden: int = HIDDEN, kernel: int = KERNEL,
                 dropout: float = DROPOUT):
        super().__init__()
        self.T       = T
        self.n_bands = n_bands
        self.n_nodes = n_nodes
        self.d_t     = d_t

        groups = n_bands * n_nodes
        self.conv = nn.Conv1d(
            in_channels=groups,
            out_channels=groups * d_t,
            kernel_size=kernel,
            padding=kernel // 2,
            groups=groups,
            bias=True,
        )
        self.conv_bn  = nn.GroupNorm(min(8, groups * d_t), groups * d_t)
        self.conv_act = nn.ELU()

        node_feat_dim = n_bands * d_t   # features per node (24 with d_t=8)

        self.gcn1 = FastGCNLayer(node_feat_dim, hidden, n_nodes)
        self.gcn2 = FastGCNLayer(hidden, hidden, n_nodes)
        self.drop = nn.Dropout(p=dropout)

        flat = hidden * T
        self.fc = nn.Linear(flat, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 9, T)"""
        B, n_bands, n_nodes, T = x.shape

        # --- temporal conv, keep time ---
        x2 = x.view(B, n_bands * n_nodes, T)              # (B, 27, T)
        h  = self.conv(x2)                                 # (B, 27*d_t, T')
        h  = h[..., :T]                                    # trim to T
        h  = self.conv_act(self.conv_bn(h))                # (B, 27*d_t, T)

        # reshape to (B, n_bands*d_t, n_nodes, T)
        h = h.view(B, n_bands, n_nodes, self.d_t, T)      # (B, 3, 9, d_t, T)
        h = h.permute(0, 1, 3, 2, 4)                      # (B, 3, d_t, 9, T)
        h = h.reshape(B, n_bands * self.d_t, n_nodes, T)  # (B, 24, 9, T)

        # --- fast GCN: T stays as a dim ---
        h = self.gcn1(h)                                   # (B, hidden, 9, T)
        h = self.gcn2(h)                                   # (B, hidden, 9, T)

        # --- pool over nodes, keep time, flatten + classify ---
        h = h.mean(dim=2)                                  # (B, hidden, T)
        h = self.drop(h.reshape(B, -1))                    # (B, hidden*T)
        return self.fc(h)                                  # (B, 40)


# ---------------------------------------------------------------------------
# G3: no graph — temporal conv flatten only (ablation)
# Unchanged from zs_gnn_v2.py
# ---------------------------------------------------------------------------

class GNNv2_G3(nn.Module):
    """Time-preserving, no-graph ablation (variant g3).

    Input : (B, n_bands, n_nodes, T)
    Output: (B, n_classes)
    """

    def __init__(self, T: int, n_bands: int = 3, n_nodes: int = 9,
                 n_classes: int = 40, d_t: int = D_T,
                 kernel: int = KERNEL, dropout: float = DROPOUT):
        super().__init__()
        self.T       = T
        self.n_bands = n_bands
        self.n_nodes = n_nodes
        self.d_t     = d_t

        groups = n_bands * n_nodes
        self.conv = nn.Conv1d(
            in_channels=groups,
            out_channels=groups * d_t,
            kernel_size=kernel,
            padding=kernel // 2,
            groups=groups,
            bias=True,
        )
        self.conv_bn  = nn.GroupNorm(min(8, groups * d_t), groups * d_t)
        self.conv_act = nn.ELU()
        self.drop     = nn.Dropout(p=dropout)

        flat = n_bands * d_t * T
        self.fc = nn.Linear(flat, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 9, T)"""
        B, n_bands, n_nodes, T = x.shape

        x2 = x.view(B, n_bands * n_nodes, T)
        h  = self.conv(x2)
        h  = h[..., :T]
        h  = self.conv_act(self.conv_bn(h))                # (B, 27*d_t, T)

        h = h.view(B, n_bands * self.d_t, n_nodes, T)     # (B, 3*d_t, 9, T)
        h = h.mean(dim=2)                                  # (B, 3*d_t, T)
        h = self.drop(h.reshape(B, -1))                    # (B, 3*d_t*T)
        return self.fc(h)                                  # (B, 40)


# ---------------------------------------------------------------------------
# model factory
# ---------------------------------------------------------------------------

def build_model(variant: str, T: int) -> nn.Module:
    if variant == "g2":
        return GNNv2_G2(T=T)
    elif variant == "g3":
        return GNNv2_G3(T=T)
    else:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {VARIANTS}")


# ---------------------------------------------------------------------------
# inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_all(model: nn.Module, X_np: np.ndarray,
                device: str, batch_size: int = BATCH_SIZE) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X_np), batch_size):
        xb = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32).to(device)
        preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# single-fold runner
# ---------------------------------------------------------------------------

def run_fold(sid: int, sl: float, variant: str, device: str = DEVICE) -> dict:
    set_seed(C.SEED)
    T    = int(sl * C.FS)
    fold = D.make_fold(sid)

    assert fold["test"] == sid
    assert sid not in fold["train"] + fold["val"], \
        "ZERO-SHOT VIOLATION: target in source pool"

    src_ids = fold["train"] + fold["val"]
    print(f"  S{sid:02d} sl={sl} GNNv2c-{variant.upper()}  "
          f"{len(src_ids)} source subjects  device={device}", flush=True)

    parts = [D.prep(s, sl) for s in src_ids]
    X_src = np.concatenate([p["A"] for p in parts]).astype(np.float32)
    y_src = np.concatenate([p["y"] for p in parts])

    te   = D.prep(sid, sl)
    X_te = te["A"].astype(np.float32)
    y_te = te["y"]

    print(f"  S{sid:02d} sl={sl} GNNv2c-{variant.upper()}  "
          f"train={len(X_src)}  test={len(X_te)}", flush=True)

    model = build_model(variant, T).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  S{sid:02d} sl={sl} GNNv2c-{variant.upper()}  params={n_params:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ds  = TensorDataset(
        torch.tensor(X_src, dtype=torch.float32),
        torch.tensor(y_src, dtype=torch.long),
    )
    dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
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
    itr   = compute_itr(acc, T)

    print(f"  S{sid:02d} sl={sl} GNNv2c-{variant.upper()}  "
          f"acc={acc:.4f}  ITR={itr:.2f}  ({elapsed}s)", flush=True)

    return dict(subject=sid, method=f"gnnv2c_{variant}", sl=sl,
                acc=acc, itr=itr, elapsed=elapsed, seed=C.SEED,
                n_params=n_params,
                preds=preds.tolist(), labels=y_te.tolist())


# ---------------------------------------------------------------------------
# checkpoint helpers
# ---------------------------------------------------------------------------

def ckpt_path(sid: int, sl: float, variant: str) -> Path:
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    tag = str(sl).replace(".", "p")
    return C.RESULTS_DIR / f"gnnv2c_{variant}_s{sid:02d}_{tag}.json"

def already_done(sid, sl, variant): return ckpt_path(sid, sl, variant).exists()


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def summarise(results: list, sl: float, variant: str):
    rs = [r for r in results if r["sl"] == sl and r["method"] == f"gnnv2c_{variant}"]
    if not rs: return
    ac = np.array([r["acc"] for r in rs])
    it = np.array([r["itr"] for r in rs])
    n  = len(ac)
    print(f"\nGNNv2c-{variant.upper()}  sl={sl}  n={n} subjects")
    print(f"  acc  mean={ac.mean():.4f}  std={ac.std():.4f}  se={ac.std()/np.sqrt(n):.4f}")
    print(f"  ITR  mean={it.mean():.2f}   std={it.std():.2f}   se={it.std()/np.sqrt(n):.2f}")
    print(f"  per-subject acc: {[round(a, 3) for a in ac.tolist()]}")


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------

def smoke_test():
    print("\n=== GNNv2c smoke test ===\n")
    device = DEVICE
    for variant in VARIANTS:
        for sl, T in [(0.5, 125), (1.0, 250)]:
            m   = build_model(variant, T).to(device)
            x   = torch.randn(4, 3, 9, T, device=device)
            out = m(x)
            assert out.shape == (4, 40), f"{variant} sl={sl} bad shape {out.shape}"
            n   = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"  {variant}  sl={sl}  T={T}  output={tuple(out.shape)}  params={n:,}  OK")
            m.train()
            y = torch.randint(0, 40, (4,), device=device)
            F.cross_entropy(m(x), y).backward()
            bad = [nm for nm, p in m.named_parameters()
                   if p.grad is None or not torch.isfinite(p.grad).all()]
            assert not bad, f"bad grads: {bad}"
            print(f"  {variant}  sl={sl}  gradients OK")
    print("\nSmoke test PASSED.\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl",       nargs="+", type=float,  default=[0.5])
    ap.add_argument("--subjects", nargs="+", type=int,    default=C.SUBJECTS)
    ap.add_argument("--variant",  nargs="+", choices=list(VARIANTS), default=["g2", "g3"])
    ap.add_argument("--resume",   action="store_true")
    ap.add_argument("--smoke",    action="store_true")
    ap.add_argument("--device",   default=DEVICE)
    a = ap.parse_args()

    if a.smoke:
        smoke_test(); import sys; sys.exit(0)

    print(f"\nDevice: {a.device}")
    print(f"Subjects: {a.subjects}")
    print(f"Windows: {a.sl}")
    print(f"Variants: {a.variant}")
    print(f"Epochs: {EPOCHS}  Batch: {BATCH_SIZE}  LR: {LR}  Seed: {C.SEED}\n")

    all_results = []

    for sl in a.sl:
        print(f"\n=== sl={sl} ===")
        for variant in a.variant:
            print(f"\n--- variant={variant} ---")
            for sid in a.subjects:
                p = ckpt_path(sid, sl, variant)
                if a.resume and already_done(sid, sl, variant):
                    print(f"  S{sid:02d} sl={sl} GNNv2c-{variant.upper()}  "
                          f"[skipping — checkpoint exists]")
                    all_results.append(json.loads(p.read_text()))
                    continue
                r = run_fold(sid, sl, variant, a.device)
                p.write_text(json.dumps(r, indent=1))
                all_results.append(r)

    print("\n=== SUMMARY ===")
    for sl in a.sl:
        for variant in a.variant:
            summarise(all_results, sl, variant)
