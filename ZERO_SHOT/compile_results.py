"""compile_results.py -- summarise every per-fold JSON in results/ and run paired tests vs a reference.
Usage: python3 compile_results.py [--ref gnnv2c_g2]
Reads results/*.json that contain subject/method/sl/acc. Writes results/compiled_summary.csv"""
import argparse, glob, json
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser(); ap.add_argument("--ref", default="gnnv2c_g2"); a = ap.parse_args()
R = defaultdict(dict)                                   # (method, sl) -> {subject: (acc, itr)}
for f in glob.glob("results/*.json"):
    try: d = json.load(open(f))
    except Exception: continue
    if isinstance(d, dict) and {"subject", "method", "sl", "acc"} <= d.keys():
        R[(d["method"], d["sl"])][d["subject"]] = (d["acc"], d.get("itr", float("nan")))

rows = []
for sl in sorted({k[1] for k in R}):
    print(f"\n=== window {sl} s   (paired vs {a.ref}) ===")
    print(f"{'method':22s}{'n':>3s}{'acc%':>8s}{'SE':>6s}{'ITR':>8s}{'ahead':>8s}{'p':>8s}")
    ref = R.get((a.ref, sl), {})
    for (m, s), d in sorted(R.items(), key=lambda kv: -np.mean([v[0] for v in kv[1].values()])):
        if s != sl: continue
        acc = np.array([v[0] for v in d.values()]); itr = np.array([v[1] for v in d.values()])
        common = sorted(set(d) & set(ref)); ahead = p = ""
        if m != a.ref and len(common) >= 6:
            x = np.array([d[i][0] for i in common]); y = np.array([ref[i][0] for i in common])
            ahead = f"{(x > y).sum()}/{len(common)}"
            p = f"{wilcoxon(x, y).pvalue:.3f}" if np.any(x != y) else "1.000"
        se = acc.std() / np.sqrt(len(acc))
        print(f"{m:22s}{len(acc):3d}{acc.mean()*100:8.2f}{se*100:6.2f}{np.nanmean(itr):8.2f}{ahead:>8s}{p:>8s}")
        rows.append((m, sl, len(acc), acc.mean(), se, np.nanmean(itr), ahead, p))
with open("results/compiled_summary.csv", "w") as f:
    f.write("method,sl,n,acc,se,itr,ahead_vs_ref,wilcoxon_p\n")
    for r in rows: f.write(",".join(str(x) for x in r) + "\n")
print("\nsaved results/compiled_summary.csv   (n < 35 means the run is incomplete)")
