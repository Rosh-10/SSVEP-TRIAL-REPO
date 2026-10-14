"""
zs_gnn_v2e.py — GNNv2c with per-band adjacency matrices.

Key difference from v2c:
  FastGCNLayer receives (B, n_bands, d_t, 9, T) and applies a separate
  learned 9x9 adjacency per band, then flattens bands*d_t before output.
  Everything else (temporal conv, GroupNorm, head, training recipe) is
  identical to v2c-G2.

Usage:
  python3 zs_gnn_v2e.py --smoke
  python3 zs_gnn_v2e.py --sl 1.0 --device cuda --resume
  python3 zs_gnn_v2e.py --sl 0.5 1.0 --device cuda --resume
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
                         DROPOUT, _sym_row_norm, predict_all)

TAG = "gnnv2e"


# ---------------------------------------------------------------------------
# Per-band GCN layer
# Input:  (B, n_bands, d_t, n_nodes, T)
# Output: (B, n_bands, out_dim, n_nodes, T)
# Each band b gets its own 9x9 adjacency A_raw[b].
# ---------------------------------------------------------------------------

class BandGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 n_bands: int = 3, n_nodes: int = 9):
        super().__init__()
        self.n_bands = n_bands
        # one adjacency per band
        self.A_raw = nn.Parameter(
            torch.randn(n_bands, n_nodes, n_nodes) * 0.1)
        self.W    = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        # GroupNorm on (B, n_bands*out_dim, n_nodes, T)
        out_total = n_bands * out_dim
        self.bn  = nn.GroupNorm(min(8, out_total), out_total)
        self.act = nn.ELU()
        nn.init.xavier_uniform_(self.W)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Fast path: 1x1 conv for W, broadcast matmul for A (no permutes).
        Mathematically equal to forward_ref (A acts on nodes, W on features)."""
        B, nb, f_in, n, T = H.shape
        A = torch.stack([_sym_row_norm(self.A_raw[b]) for b in range(nb)])
        f_out = self.W.shape[1]
        Hw = F.conv2d(H.reshape(B * nb, f_in, n, T), self.W.t()[:, :, None, None])
        Hw = Hw.view(B, nb, f_out, n, T)
        Ho = torch.matmul(A[None, :, None], Hw) + self.bias.view(1, 1, -1, 1, 1)
        Hf = self.act(self.bn(Ho.reshape(B, nb * f_out, n, T)))
        return Hf.reshape(B, nb, f_out, n, T)

    def forward_ref(self, H: torch.Tensor) -> torch.Tensor:
        """H: (B, n_bands, f_in, n_nodes, T)
           -> (B, n_bands, f_out, n_nodes, T)"""
        B, n_bands, f_in, n_nodes, T = H.shape

        # per-band row-normalised adjacency: (n_bands, 9, 9)
        A = torch.stack([_sym_row_norm(self.A_raw[b])
                         for b in range(n_bands)], dim=0)

        # graph mix per band: einsum over nodes
        # H: (B, n_bands, f_in, n_nodes, T)
        # A: (n_bands, n_nodes, n_nodes)  [m=dest, n=src]
        H_agg = torch.einsum('bkfnt,kmn->bkfmt', H, A)   # (B,bands,f_in,9,T)

        # linear transform (shared W across bands and nodes)
        H_out = torch.einsum('bkfnt,fo->bkont', H_agg, self.W) \
                + self.bias.view(1, 1, -1, 1, 1)          # (B,bands,f_out,9,T)

        # GroupNorm expects (B, C, *): merge bands into channel dim
        f_out = H_out.shape[2]
        H_flat = H_out.reshape(B, n_bands * f_out, n_nodes, T)
        H_flat = self.act(self.bn(H_flat))

        # restore band dim
        return H_flat.reshape(B, n_bands, f_out, n_nodes, T)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class GNNv2e(nn.Module):
    def __init__(self, T: int, n_bands: int = 3, n_nodes: int = 9,
                 n_classes: int = 40, d_t: int = D_T,
                 hidden: int = HIDDEN, kernel: int = KERNEL,
                 dropout: float = DROPOUT):
        super().__init__()
        self.T = T; self.n_bands = n_bands
        self.n_nodes = n_nodes; self.d_t = d_t

        # temporal depthwise conv (identical to v2c)
        groups = n_bands * n_nodes
        self.conv     = nn.Conv1d(groups, groups * d_t, kernel_size=kernel,
                                  padding=kernel // 2, groups=groups, bias=True)
        self.conv_bn  = nn.GroupNorm(min(8, groups * d_t), groups * d_t)
        self.conv_act = nn.ELU()

        # two per-band GCN layers
        self.gcn1 = BandGCNLayer(d_t,     hidden, n_bands, n_nodes)
        self.gcn2 = BandGCNLayer(hidden,  hidden, n_bands, n_nodes)

        self.drop = nn.Dropout(p=dropout)
        # after mean over nodes: (B, n_bands, hidden, T) -> flatten
        self.fc   = nn.Linear(hidden * T, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 9, T)"""
        B, n_bands, n_nodes, T = x.shape

        # temporal conv (same as v2c)
        x2 = x.view(B, n_bands * n_nodes, T)
        h  = self.conv_act(self.conv_bn(self.conv(x2)[..., :T]))  # (B,27*d_t,T)

        # reshape to keep bands separate: (B, n_bands, d_t, n_nodes, T)
        h = h.view(B, n_bands, n_nodes, self.d_t, T)   # (B,3,9,d_t,T)
        h = h.permute(0, 1, 3, 2, 4)                   # (B,3,d_t,9,T)

        # per-band GCN
        h = self.gcn1(h)                                # (B,3,hidden,9,T)
        h = self.gcn2(h)                                # (B,3,hidden,9,T)

        # pool over nodes then bands, flatten, classify
        h = h.mean(dim=3)                               # (B,3,hidden,T)
        h = h.mean(dim=1)                               # (B,hidden,T)
        h = self.drop(h.reshape(B, -1))                 # (B,hidden*T)
        return self.fc(h)                               # (B,40)


# ---------------------------------------------------------------------------
# Training helpers (identical structure to v2c)
# ---------------------------------------------------------------------------

def ckpt_path(sid, sl):
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    return C.RESULTS_DIR / f"{TAG}_s{sid:02d}_{str(sl).replace('.','p')}.json"

def already_done(sid, sl): return ckpt_path(sid, sl).exists()


def run_fold(sid, sl, device=DEVICE):
    set_seed(C.SEED)
    T = int(sl * C.FS); fold = D.make_fold(sid)
    src_ids = fold["train"] + fold["val"]
    print(f"  S{sid:02d} sl={sl} GNNv2e  {len(src_ids)} source  device={device}",
          flush=True)
    parts  = [D.prep(s, sl) for s in src_ids]
    X_src  = np.concatenate([p["A"] for p in parts]).astype(np.float32)
    y_src  = np.concatenate([p["y"] for p in parts])
    te     = D.prep(sid, sl)
    X_te, y_te = te["A"].astype(np.float32), te["y"]

    model  = GNNv2e(T=T).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  S{sid:02d} sl={sl} GNNv2e  params={n_params:,}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl  = DataLoader(
        TensorDataset(torch.tensor(X_src),
                      torch.tensor(y_src, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=(device == "cuda"))

    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tot, n = 0.0, 0
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

    # save logits for fusion
    model.eval()
    with torch.no_grad():
        lg = np.concatenate([
            model(torch.as_tensor(X_te[i:i+64]).to(device)).cpu().numpy()
            for i in range(0, len(X_te), 64)])
    agree = (lg.argmax(1) == preds).mean()
    print(f"  logits check: argmax agrees with preds on {agree:.3f}", flush=True)
    np.savez(C.RESULTS_DIR /
             f"logits_{TAG}_s{sid:02d}_{str(sl).replace('.','p')}.npz",
             logits=lg, y=y_te)

    itr = compute_itr(acc, T)
    print(f"  S{sid:02d} sl={sl} GNNv2e  acc={acc:.4f}  ITR={itr:.2f}"
          f"  ({elapsed}s)", flush=True)
    return dict(subject=sid, method=TAG, sl=sl, acc=acc, itr=itr,
                elapsed=elapsed, seed=C.SEED, n_params=n_params,
                preds=preds.tolist(), labels=y_te.tolist())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl",       nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--subjects", nargs="+", type=int,   default=C.SUBJECTS)
    ap.add_argument("--resume",   action="store_true")
    ap.add_argument("--smoke",    action="store_true")
    ap.add_argument("--device",   default=DEVICE)
    a = ap.parse_args()

    if a.smoke:
        print("\n=== smoke test ===")
        for sl, T in [(0.5, 125), (1.0, 250)]:
            m = GNNv2e(T=T)
            x = torch.randn(4, 3, 9, T)
            o = m(x); assert o.shape == (4, 40)
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"  sl={sl}  T={T}  params={n:,}  output={o.shape}  OK")
        import sys; sys.exit(0)

    print(f"\nDevice: {a.device}  Windows: {a.sl}")
    print(f"Epochs: {EPOCHS}  Batch: {BATCH_SIZE}  LR: {LR}  Seed: {C.SEED}\n")
    all_results = []
    for sl in a.sl:
        print(f"\n=== sl={sl} ===")
        for sid in a.subjects:
            if a.resume and already_done(sid, sl):
                print(f"  S{sid:02d} sl={sl} GNNv2e  [skip]")
                all_results.append(json.loads(ckpt_path(sid, sl).read_text()))
                continue
            r = run_fold(sid, sl, a.device)
            ckpt_path(sid, sl).write_text(json.dumps(r, indent=1))
            all_results.append(r)

    for sl in a.sl:
        rs = [r for r in all_results if r["sl"] == sl]
        if not rs: continue
        ac = np.array([r["acc"] for r in rs])
        it = np.array([r["itr"] for r in rs])
        print(f"\nGNNv2e  sl={sl}  n={len(ac)}"
              f"  acc={ac.mean():.4f}±{ac.std():.4f}"
              f"  ITR={it.mean():.2f}")