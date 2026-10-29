"""
zs_gnn_v3_common.py -- shared 35-fold LOSO runner for the v3 GNNs (no CCA anywhere).

Recipe is copied from zs_gnn_v2c.py so results stay comparable with CNN_ref / v2c:
  34 source subjects, no val set, no early stopping, 100 epochs, Adam lr 2e-4,
  batch 128, last-epoch model, seed 41 (C.SEED), input = bank-A standardized (3,9,T).
Results: results/{tag}_{variant}_s{sid}_{sl}.json  (one file per fold, --resume skips done folds).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import zs_config as C
import zs_data as D
from zs_gnn_train import set_seed, compute_itr, DEVICE

EPOCHS, BATCH_SIZE, LR = 100, 128, 2e-4   # [REPO] same as CNN_ref / v2c


@torch.no_grad()
def predict_all(model, X_np, device, batch_size=BATCH_SIZE):
    model.eval()
    out = []
    for i in range(0, len(X_np), batch_size):
        xb = torch.tensor(X_np[i:i + batch_size], dtype=torch.float32).to(device)
        out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def ckpt_path(tag, sid, sl, variant):
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    return C.RESULTS_DIR / f"{tag}_{variant}_s{sid:02d}_{str(sl).replace('.', 'p')}.json"


def model_path(tag, sid, sl, variant):
    d = C.RESULTS_DIR / "models"; d.mkdir(exist_ok=True, parents=True)
    return d / f"{tag}_{variant}_s{sid:02d}_{str(sl).replace('.', 'p')}.pt"


def load_trained(tag, build_model, sid, sl, variant, device="cpu"):
    """Reload a saved fold model: model = load_trained('gnnv3a', build_model, 1, 1.0, 'pb')."""
    ck = torch.load(model_path(tag, sid, sl, variant), map_location=device)
    m = build_model(variant, ck["T"]).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def run_fold(tag, build_model, sid, sl, variant, device):
    set_seed(C.SEED)
    T = int(sl * C.FS)
    fold = D.make_fold(sid)
    assert fold["test"] == sid and sid not in fold["train"] + fold["val"], "ZERO-SHOT VIOLATION"
    src_ids = fold["train"] + fold["val"]               # all 34 source subjects, as in v2c
    name = f"S{sid:02d} sl={sl} {tag}-{variant}"
    parts = [D.prep(s, sl) for s in src_ids]
    X_src = np.concatenate([p["A"] for p in parts]).astype(np.float32)
    y_src = np.concatenate([p["y"] for p in parts])
    te = D.prep(sid, sl)
    X_te, y_te = te["A"].astype(np.float32), te["y"]

    model = build_model(variant, T).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {name}  train={len(X_src)} test={len(X_te)} params={n_params:,} device={device}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = DataLoader(TensorDataset(torch.tensor(X_src), torch.tensor(y_src, dtype=torch.long)),
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = n = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(yb); n += len(yb)
        if ep == 1 or ep % 25 == 0:
            print(f"    ep {ep:3d}  loss={tot / n:.4f}", flush=True)
    elapsed = round(time.time() - t0, 1)

    torch.save(dict(state_dict=model.state_dict(), T=T, sid=sid, sl=sl, variant=variant, tag=tag,
                    seed=C.SEED, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, n_params=n_params),
               model_path(tag, sid, sl, variant))       # saved right after training, before evaluation

    preds = predict_all(model, X_te, device)
    acc = float((preds == y_te).mean()); itr = compute_itr(acc, T)
    print(f"  {name}  acc={acc:.4f} ITR={itr:.2f} ({elapsed}s)", flush=True)
    res = dict(subject=sid, method=f"{tag}_{variant}", sl=sl, acc=acc, itr=itr, elapsed=elapsed,
               seed=C.SEED, n_params=n_params, preds=preds.tolist(), labels=y_te.tolist())
    if hasattr(model, "export_graph"):
        res["graph"] = model.export_graph()             # learned adjacency, for plotting later
    return res


def summarise(results, tag, sl, variant):
    rs = [r for r in results if r["sl"] == sl and r["method"] == f"{tag}_{variant}"]
    if not rs: return
    ac = np.array([r["acc"] for r in rs]); it = np.array([r["itr"] for r in rs]); n = len(ac)
    print(f"\n{tag}-{variant}  sl={sl}  n={n}")
    print(f"  acc mean={ac.mean():.4f} std={ac.std():.4f} se={ac.std() / np.sqrt(n):.4f}")
    print(f"  ITR mean={np.nanmean(it):.2f} se={np.nanstd(it) / np.sqrt(n):.2f}")


def smoke(tag, variants, build_model, **kw):
    print(f"\n=== {tag} smoke test ===")
    for v in variants:
        for T in (125, 250):
            m = build_model(v, T, **kw).to(DEVICE)
            x = torch.randn(4, 3, 9, T, device=DEVICE)
            out = m(x); assert out.shape == (4, 40), out.shape
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            m.train(); F.cross_entropy(m(x), torch.randint(0, 40, (4,), device=DEVICE)).backward()
            bad = [k for k, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
            assert not bad, bad
            print(f"  {v} T={T} out={tuple(out.shape)} params={n:,} grads OK")
    print("PASSED\n")


def cli(tag, variants, build_model, default_variants=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl", nargs="+", type=float, default=[0.5])
    ap.add_argument("--subjects", nargs="+", type=int, default=C.SUBJECTS)
    ap.add_argument("--variant", nargs="+", choices=list(variants), default=default_variants or list(variants))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=DEVICE)
    a = ap.parse_args()
    if a.smoke:
        smoke(tag, variants, build_model); sys.exit(0)
    print(f"{tag} device={a.device} subjects={a.subjects} sl={a.sl} variants={a.variant} "
          f"epochs={EPOCHS} bs={BATCH_SIZE} lr={LR} seed={C.SEED}")
    allr = []
    for sl in a.sl:
        for v in a.variant:
            for sid in a.subjects:
                p = ckpt_path(tag, sid, sl, v)
                if a.resume and p.exists():
                    print(f"  S{sid:02d} sl={sl} {tag}-{v} [skip, checkpoint exists]")
                    allr.append(json.loads(p.read_text())); continue
                r = run_fold(tag, build_model, sid, sl, v, a.device)
                p.write_text(json.dumps(r, indent=1)); allr.append(r)
    print("\n=== SUMMARY ===")
    for sl in a.sl:
        for v in a.variant:
            summarise(allr, tag, sl, v)
