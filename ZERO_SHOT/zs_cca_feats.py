"""
zs_cca_feats.py - save the CCA branch of CCA-Net once per LOSO fold (network-independent, no GPU).
  f1 = FBCCA (sin/cos refs)   f2 = Modified FBtt-CCA (SST)   f3 = FBCCA (SST)      each (240,40)
Output: results/ccafeats_sXX_{0p5|1p0}.npz  (f1,f2,f3,y).  ~1 min per fold.
Check: if results/cca_net_off_sXX_*.json exists, re-derives acc_tt/acc_cca/acc_mod and compares.
Usage (from ZERO_SHOT/):  python3 zs_cca_feats.py --sl 0.5 1.0 --resume
"""
import argparse, json, time
import numpy as np
import zs_config as C, zs_data as D
from zs_cca_net import per_subject_lst, fit_gmlst, FBTtCCA
from zs_cca_official import cca_reference, fbcca_official

def feat_path(sid, sl): return C.RESULTS_DIR / f"ccafeats_s{sid:02d}_{str(sl).replace('.', 'p')}.npz"

def run(sid, sl):
    T = int(sl * C.FS); freqs, _, phases = D.load_freq_phase()
    fold = D.make_fold(sid); parts = [D.get_trials(s, sl) for s in fold["train"] + fold["val"]]
    X = np.concatenate([p[0] for p in parts]).astype(np.float64); y = np.concatenate([p[1] for p in parts]).astype(int)
    Xt, yt, _ = D.get_trials(sid, sl); Xt = Xt.astype(np.float64); yt = np.asarray(yt).astype(int)
    L = np.concatenate([per_subject_lst(p[0].astype(np.float64), np.asarray(p[1]).astype(int), freqs, phases) for p in parts])
    sst = fit_gmlst(X, L, y, freqs, phases)[1]
    f2 = FBTtCCA(1).fit(X, y, sst).predict(Xt)[1]
    f3 = fbcca_official(Xt, sst); f1 = fbcca_official(Xt, cca_reference(freqs, T))
    return f1, f2, f3, yt

def check_vs_json(sid, sl, f1, f2, f3, y):
    j = C.RESULTS_DIR / f"cca_net_off_s{sid:02d}_{str(sl).replace('.', 'p')}.json"
    if not j.exists(): return "no cca_net_off json to compare"
    r = json.loads(j.read_text()); A = lambda s: float((s.argmax(1) == y).mean())
    mm = lambda s: (s - s.min(1, keepdims=True)) / np.where(np.ptp(s, 1, keepdims=True) == 0, 1, np.ptp(s, 1, keepdims=True))
    mine = dict(acc_tt=A(f2), acc_cca=A(mm(f3)), acc_mod=A(mm(f2) + mm(f3)))
    bad = {k: (round(v, 4), round(r[k], 4)) for k, v in mine.items() if abs(v - r[k]) > 1e-9}
    return "matches json" if not bad else f"MISMATCH {bad}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl", nargs="+", type=float, default=[0.5, 1.0]); ap.add_argument("--subjects", nargs="+", type=int, default=C.SUBJECTS)
    ap.add_argument("--resume", action="store_true"); a = ap.parse_args(); C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    for sl in a.sl:
        for sid in a.subjects:
            p = feat_path(sid, sl)
            if a.resume and p.exists(): print(f"  S{sid:02d} sl={sl} [skip]", flush=True); continue
            t = time.time(); f1, f2, f3, y = run(sid, sl)
            np.savez_compressed(p, f1=f1, f2=f2, f3=f3, y=y)
            print(f"  S{sid:02d} sl={sl} saved ({time.time()-t:.0f}s)  {check_vs_json(sid, sl, f1, f2, f3, y)}", flush=True)
