"""zs_export_logits.py -- regenerate per-fold logits of saved v3 models (no retraining) so zs_fuse.py can fuse them with CCA features.
Reads results/models/<tag>_<variant>_sXX_<sl>.pt (saved by zs_gnn_v3_common.run_fold) and the same test data the fold used.
Writes results/logits_<tag>_<variant>_sXX_<0p5|1p0>.npz (logits (240,40), y). Checks argmax(logits) against the preds in the fold JSON.
Usage (one family file per call; CPU is fine and does not disturb a running GPU job):
  python3 zs_export_logits.py zs_gnn_v3g_graph dense19 --sl 0.5 1.0
  python3 zs_export_logits.py zs_gnn_v3l_bank5 b5d19 b5d9 --sl 0.5 1.0
Then:  python3 zs_fuse.py --tag gnnv3g_dense19 --sl 0.5 1.0 --vs gnnv2c_g2
"""
import argparse, importlib, json, re, sys
import numpy as np, torch
import zs_config as C, zs_gnn_v3_common as K

ap = argparse.ArgumentParser()
ap.add_argument("module"); ap.add_argument("variant", nargs="+")
ap.add_argument("--sl", nargs="+", type=float, default=[0.5, 1.0]); ap.add_argument("--device", default="cpu")
a = ap.parse_args()
M = importlib.import_module(a.module); G = sys.modules.get("zs_gnn_v3g_graph"); L = sys.modules.get("zs_gnn_v3l_bank5")

def set_channels(v):                       # families built on v3g / v3l read the channel set from a module global CH
    if G is None: return
    ch = G.chans(int(re.search(r"\d+$", v).group()))
    for m in (G, L):
        if m is not None and hasattr(m, "CH"): m.CH = ch

for sl in a.sl:
    k = str(sl).replace(".", "p")
    for v in a.variant:
        set_channels(v); ag = []
        for sid in C.SUBJECTS:
            out = C.RESULTS_DIR / f"logits_{M.TAG}_{v}_s{sid:02d}_{k}.npz"
            if out.exists(): continue
            m = K.load_trained(M.TAG, M.build_model, sid, sl, v, a.device)
            te = K.D.prep(sid, sl); X, y = te["A"].astype(np.float32), np.asarray(te["y"])
            with torch.no_grad(): lg = np.concatenate([m(torch.as_tensor(X[i:i + 64]).to(a.device)).cpu().numpy() for i in range(0, len(X), 64)])
            j = json.loads(K.ckpt_path(M.TAG, sid, sl, v).read_text())
            assert (y == np.array(j["labels"])).all(), f"S{sid:02d}: label order differs from the fold JSON"
            ag.append(float((lg.argmax(1) == np.array(j["preds"])).mean())); np.savez(out, logits=lg, y=y)
            print(f"  {M.TAG}_{v} sl={sl} S{sid:02d} saved, argmax agrees with JSON preds on {ag[-1]:.3f}", flush=True)
        if ag: print(f"== {M.TAG}_{v} sl={sl}: {len(ag)} folds, mean agreement {np.mean(ag):.4f}, min {np.min(ag):.3f}", flush=True)
