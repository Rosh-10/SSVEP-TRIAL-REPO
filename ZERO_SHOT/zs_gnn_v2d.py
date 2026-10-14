"""
zs_gnn_v2d.py — Graph ablations for GNNv2c-G2.
Identical to zs_gnn_v2c.py except FastGCNLayer accepts adj_mode in
{learned, identity, uniform, anatomical} and GNNv2_G2 passes it through.
New variants: g2_identity, g2_uniform, g2_anatomical.
Usage:
  python3 zs_gnn_v2d.py --smoke
  python3 zs_gnn_v2d.py --sl 1.0 --variant g2_identity g2_uniform g2_anatomical --resume --device cuda
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import zs_config as C
import zs_data as D
from zs_gnn_train import set_seed, compute_itr, DEVICE
from zs_gnn_v2c import (EPOCHS, BATCH_SIZE, LR, D_T, KERNEL, HIDDEN,
                         DROPOUT, _sym_row_norm, GNNv2_G3, predict_all)

VARIANTS = ("g2_learned", "g2_identity", "g2_uniform", "g2_anatomical")


# ---------------------------------------------------------------------------
# anatomical adjacency from 64-channels.loc
# ---------------------------------------------------------------------------

def _anatomical_adj(n_nodes: int = 9, ch_indices=None) -> torch.Tensor:
    """Row-normalised adjacency from inverse Euclidean distances (x,y coords).
    ch_indices: 0-based indices of the 9 channels in the .loc file order."""
    if ch_indices is None:
        ch_indices = C.CHANNELS  # [47,53,54,55,56,57,60,61,62]
    loc_file = Path(__file__).parent / "benchmark" / "64-channels.loc"
    coords = {}
    with open(loc_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                idx = int(parts[0]) - 1  # 1-indexed in file -> 0-indexed
                x, y = float(parts[1]), float(parts[2])
                coords[idx] = (x, y)
    xy = np.array([coords[i] for i in ch_indices])  # (9, 2)
    # pairwise Euclidean distance
    diff = xy[:, None, :] - xy[None, :, :]          # (9, 9, 2)
    dist = np.sqrt((diff ** 2).sum(-1))              # (9, 9)
    # inverse distance; self-connection = mean of off-diagonal
    np.fill_diagonal(dist, np.inf)
    W = 1.0 / dist
    np.fill_diagonal(W, W[~np.eye(9, dtype=bool)].mean())
    # row-normalise
    W = W / W.sum(1, keepdims=True)
    return torch.tensor(W, dtype=torch.float32)


# cache it once at import time
_ANAT_ADJ = None

def get_anatomical_adj(device):
    global _ANAT_ADJ
    if _ANAT_ADJ is None:
        _ANAT_ADJ = _anatomical_adj()
    return _ANAT_ADJ.to(device)


# ---------------------------------------------------------------------------
# FastGCNLayer with adj_mode
# ---------------------------------------------------------------------------

class FastGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_nodes: int = 9,
                 adj_mode: str = "learned"):
        super().__init__()
        self.adj_mode = adj_mode
        if adj_mode == "learned":
            self.A_raw = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1)
        self.W    = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.bn   = nn.GroupNorm(min(8, out_dim), out_dim)
        self.act  = nn.ELU()
        nn.init.xavier_uniform_(self.W)

    def _get_A(self, device, n_nodes=9):
        if self.adj_mode == "learned":
            return _sym_row_norm(self.A_raw)
        elif self.adj_mode == "identity":
            return (torch.eye(n_nodes, device=device) / n_nodes)
        elif self.adj_mode == "uniform":
            return torch.full((n_nodes, n_nodes), 1.0 / n_nodes, device=device)
        elif self.adj_mode == "anatomical":
            return get_anatomical_adj(device)
        else:
            raise ValueError(self.adj_mode)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        A = self._get_A(H.device, H.shape[2])
        H_agg = torch.einsum('bfnt,mn->bfmt', H, A)
        H_out = torch.einsum('bfnt,fo->bont', H_agg, self.W) + self.bias.view(1,-1,1,1)
        return self.act(self.bn(H_out))


# ---------------------------------------------------------------------------
# GNNv2_G2 with adj_mode threaded through
# ---------------------------------------------------------------------------

class GNNv2_G2(nn.Module):
    def __init__(self, T: int, n_bands: int = 3, n_nodes: int = 9,
                 n_classes: int = 40, d_t: int = D_T,
                 hidden: int = HIDDEN, kernel: int = KERNEL,
                 dropout: float = DROPOUT, adj_mode: str = "learned"):
        super().__init__()
        self.T = T; self.n_bands = n_bands
        self.n_nodes = n_nodes; self.d_t = d_t
        groups = n_bands * n_nodes
        self.conv = nn.Conv1d(groups, groups * d_t, kernel_size=kernel,
                              padding=kernel // 2, groups=groups, bias=True)
        self.conv_bn  = nn.GroupNorm(min(8, groups * d_t), groups * d_t)
        self.conv_act = nn.ELU()
        node_feat_dim = n_bands * d_t
        self.gcn1 = FastGCNLayer(node_feat_dim, hidden, n_nodes, adj_mode)
        self.gcn2 = FastGCNLayer(hidden, hidden, n_nodes, adj_mode)
        self.drop = nn.Dropout(p=dropout)
        self.fc   = nn.Linear(hidden * T, n_classes)
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
        B, n_bands, n_nodes, T = x.shape
        x2 = x.view(B, n_bands * n_nodes, T)
        h  = self.conv_act(self.conv_bn(self.conv(x2)[..., :T]))
        h  = h.view(B, n_bands, n_nodes, self.d_t, T)
        h  = h.permute(0, 1, 3, 2, 4).reshape(B, n_bands * self.d_t, n_nodes, T)
        h  = self.gcn2(self.gcn1(h))
        h  = self.drop(h.mean(dim=2).reshape(B, -1))
        return self.fc(h)


# ---------------------------------------------------------------------------
# factory, runner, checkpoint, main  (identical structure to v2c)
# ---------------------------------------------------------------------------

def build_model(variant: str, T: int) -> nn.Module:
    mode_map = {"g2_learned": "learned", "g2_identity": "identity",
                "g2_uniform": "uniform", "g2_anatomical": "anatomical"}
    if variant in mode_map:
        return GNNv2_G2(T=T, adj_mode=mode_map[variant])
    raise ValueError(variant)


def ckpt_path(sid, sl, variant):
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    return C.RESULTS_DIR / f"gnnv2d_{variant}_s{sid:02d}_{str(sl).replace('.','p')}.json"

def already_done(sid, sl, variant): return ckpt_path(sid, sl, variant).exists()


def run_fold(sid, sl, variant, device=DEVICE):
    set_seed(C.SEED)
    T = int(sl * C.FS); fold = D.make_fold(sid)
    src_ids = fold["train"] + fold["val"]
    print(f"  S{sid:02d} sl={sl} GNNv2d-{variant}  {len(src_ids)} source  device={device}", flush=True)
    parts  = [D.prep(s, sl) for s in src_ids]
    X_src  = np.concatenate([p["A"] for p in parts]).astype(np.float32)
    y_src  = np.concatenate([p["y"] for p in parts])
    te     = D.prep(sid, sl)
    X_te, y_te = te["A"].astype(np.float32), te["y"]
    model  = build_model(variant, T).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  S{sid:02d} sl={sl} GNNv2d-{variant}  params={n_params:,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl  = DataLoader(TensorDataset(torch.tensor(X_src), torch.tensor(y_src, dtype=torch.long)),
                     batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                     pin_memory=(device=="cuda"))
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tot, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(yb); n += len(yb)
        if ep % 25 == 0 or ep == 1:
            print(f"    ep {ep:3d}  loss={tot/n:.4f}", flush=True)
    elapsed = round(time.time() - t0, 1)
    preds = predict_all(model, X_te, device)
    acc   = float((preds == y_te).mean())
    model.eval()
    with torch.no_grad():
        lg = np.concatenate([model(torch.as_tensor(X_te[i:i+64]).to(device)).cpu().numpy()
                             for i in range(0, len(X_te), 64)])
    np.savez(C.RESULTS_DIR / f"logits_gnnv2d_{variant}_s{sid:02d}_{str(sl).replace('.','p')}.npz",
             logits=lg, y=y_te)
    itr = compute_itr(acc, T)
    print(f"  S{sid:02d} sl={sl} GNNv2d-{variant}  acc={acc:.4f}  ITR={itr:.2f}  ({elapsed}s)", flush=True)
    return dict(subject=sid, method=f"gnnv2d_{variant}", sl=sl, acc=acc, itr=itr,
                elapsed=elapsed, seed=C.SEED, n_params=n_params,
                preds=preds.tolist(), labels=y_te.tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl",       nargs="+", type=float, default=[1.0])
    ap.add_argument("--subjects", nargs="+", type=int,   default=C.SUBJECTS)
    ap.add_argument("--variant",  nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    ap.add_argument("--resume",   action="store_true")
    ap.add_argument("--smoke",    action="store_true")
    ap.add_argument("--device",   default=DEVICE)
    a = ap.parse_args()

    if a.smoke:
        print("\n=== smoke test ===")
        for v in VARIANTS:
            for sl, T in [(0.5, 125), (1.0, 250)]:
                m = build_model(v, T)
                x = torch.randn(4, 3, 9, T)
                o = m(x); assert o.shape == (4, 40)
                n = sum(p.numel() for p in m.parameters() if p.requires_grad)
                print(f"  {v}  sl={sl}  params={n:,}  OK")
        import sys; sys.exit(0)

    print(f"\nDevice: {a.device}  Windows: {a.sl}  Variants: {a.variant}")
    all_results = []
    for sl in a.sl:
        print(f"\n=== sl={sl} ===")
        for variant in a.variant:
            print(f"\n--- variant={variant} ---")
            for sid in a.subjects:
                if a.resume and already_done(sid, sl, variant):
                    print(f"  S{sid:02d} [skip]"); all_results.append(json.loads(ckpt_path(sid,sl,variant).read_text())); continue
                r = run_fold(sid, sl, variant, a.device)
                ckpt_path(sid, sl, variant).write_text(json.dumps(r, indent=1))
                all_results.append(r)

    for sl in a.sl:
        for v in a.variant:
            rs = [r for r in all_results if r["sl"]==sl and r["method"]==f"gnnv2d_{v}"]
            if not rs: continue
            ac = np.array([r["acc"] for r in rs])
            print(f"\nGNNv2d-{v}  sl={sl}  n={len(ac)}  acc={ac.mean():.4f}±{ac.std():.4f}")