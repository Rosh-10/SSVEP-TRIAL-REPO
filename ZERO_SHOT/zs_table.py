"""zs_table.py -- one row per method: n folds, acc / SE / ITR at 0.5 s and 1.0 s (acc in %). Writes results/all_models.csv.
Run from ZERO_SHOT/ after any job:  python3 zs_table.py     (n < 35 means the run is incomplete)"""
import glob, json
import numpy as np, pandas as pd
R = {}
for f in glob.glob("results/*.json"):
    try: d = json.load(open(f))
    except Exception: continue
    if isinstance(d, dict) and {"subject", "method", "sl", "acc"} <= d.keys():
        R.setdefault((d["method"], d["sl"]), {})[d["subject"]] = (d["acc"], d.get("itr", np.nan))
rows = []
for (m, sl), s in R.items():
    a = np.array([v[0] for v in s.values()]) * 100; i = np.array([v[1] for v in s.values()], float)
    rows.append(dict(method=m, sl=sl, n=len(a), acc=a.mean(), se=a.std() / np.sqrt(len(a)), itr=np.nanmean(i)))
w = pd.DataFrame(rows).pivot(index="method", columns="sl", values=["n", "acc", "se", "itr"])
w.columns = [f"{a}_{str(b).replace('.', '')}" for a, b in w.columns]            # acc_05, acc_10, ...
w = w.sort_values("acc_10", ascending=False, na_position="last")
w.round(2).to_csv("results/all_models.csv"); print(w.round(2).to_string())
