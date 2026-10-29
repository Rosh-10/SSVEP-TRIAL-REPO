"""zs_consolidate.py -- consolidate EVERY result on disk into results/consolidated/ (read-only on results/).
Run from ZERO_SHOT/ after any job:  python3 zs_consolidate.py
Writes: all_models.csv, per_subject_acc.csv, seeds.csv, fusion.csv, subjects.csv, tests.csv, report_tables.md, all_results.xlsx (needs openpyxl)
Inputs: results/*.json (one per fold), results/logits_*.npz, results/ccafeats_*.npz.  Missing pieces are skipped, never invented."""
import glob, json, re
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, ttest_ind
import zs_config as C

OUT = C.RESULTS_DIR / "consolidated"; OUT.mkdir(parents=True, exist_ok=True)
SL = {"0p5": 0.5, "1p0": 1.0}; rng = np.random.default_rng(0)
HARD = [11, 19, 29, 33]; EXP = list(range(1, 9)); NAIVE = list(range(9, 36))
# method -> (family, description). "INVALID" in the description = never use as evidence.
L = {"cca_net_off": ("baseline", "CCA-Net replication: CNN + Modified FBtt-CCA + FBCCA(SST)"), "fbttcca_yf": ("baseline", "FBtt-CCA (classical)"),
     "fbcca": ("baseline", "FBCCA (classical)"), "cnn_ref": ("baseline", "CCA-Net CNN branch alone (CSDuDoFN-like)"), "gnn": ("v1", "first conv+GCN"),
     "gnnv2_g2": ("v2", "conv+GCN, learned graph"), "gnnv2_g3": ("v2", "conv, no graph"), "gnnv2b_g2": ("v2", "v2 + GroupNorm, graph"), "gnnv2b_g3": ("v2", "v2 + GroupNorm, no graph"),
     "gnnv2c_g2": ("v2", "v2c reference, learned graph"), "gnnv2c_g3": ("v2", "v2c, no graph"), "gnnv2d_g2_identity": ("v2", "graph ablation: identity"),
     "gnnv2d_g2_uniform": ("v2", "graph ablation: uniform"), "gnnv2d_g2_anatomical": ("v2", "INVALID: polar-coordinate bug in anatomical adjacency"),
     "gnnv3a_pb": ("v3", "per-band + layer-wise adjacency (conv front end)"), "gnnv3b3_fg3": ("v3", "template front end, no graph"), "gnnv3b3_fg2": ("v3", "learned gated adjacency"),
     "gnnv3c_ng": ("v3", "per-node gate"), "gnnv3c_h5": ("v3", "5 harmonics"), "gnnv3c_ngh5": ("v3", "per-node gate + 5 harmonics"), "gnnv3d_fgu": ("v3", "uniform adjacency + gate"),
     "gnnv3e_cm": ("v3", "fixed channel-mean contrast"), "gnnv3e_lr": ("v3", "learned signed 9x9 map"), "gnnv3g_knn9": ("v3", "kNN signed map, 9 ch"), "gnnv3g_knn19": ("v3", "kNN signed map, 19 ch"),
     "gnnv3g_knn64": ("v3", "kNN signed map, 64 ch"), "gnnv3g_dense19": ("v3", "free signed map, 19 ch"), "gnnv3g_dense64": ("v3", "free signed map, 64 ch"), "gnnv3h_fga": ("v3", "anatomical adjacency + gate"),
     "gnnv3i_wide19": ("v3", "dense19, 4x wider"), "gnnv3j_aw9": ("v3", "trial-adaptive node weights, 9 ch"), "gnnv3j_aw19": ("v3", "trial-adaptive node weights, 19 ch"),
     "gnnv3k_s42d19": ("v3", "dense19 seed 42"), "gnnv3k_s43d19": ("v3", "dense19 seed 43"), "gnnv3l_b5d9": ("v3", "dense, 5-band bank C, 9 ch"), "gnnv3l_b5d19": ("v3", "dense, 5-band bank C, 19 ch"),
     "gnnv3m_s42b5d19": ("v3", "b5d19 seed 42"), "gnnv3m_s43b5d19": ("v3", "b5d19 seed 43")}
GROUPS = {"dense19": ["gnnv3g_dense19", "gnnv3k_s42d19", "gnnv3k_s43d19"], "b5d19": ["gnnv3l_b5d19", "gnnv3m_s42b5d19", "gnnv3m_s43b5d19"]}
PAIRS = [("gnnv3b3_fg2", "gnnv3b3_fg3"), ("gnnv3d_fgu", "gnnv3b3_fg3"), ("gnnv3e_lr", "gnnv3b3_fg2"), ("gnnv3e_cm", "gnnv3b3_fg3"), ("gnnv3h_fga", "gnnv3d_fgu"), ("gnnv3g_dense19", "gnnv3e_lr"),
         ("gnnv3g_dense19", "gnnv3g_knn19"), ("gnnv3i_wide19", "gnnv3g_dense19"), ("gnnv3j_aw19", "gnnv3g_dense19"), ("gnnv3j_aw9", "gnnv3e_lr"), ("gnnv3l_b5d19", "gnnv3g_dense19"),
         ("gnnv3l_b5d9", "gnnv3e_lr"), ("gnnv3k_s42d19", "gnnv3g_dense19"), ("gnnv3k_s43d19", "gnnv3g_dense19"), ("cca_net_off", "gnnv3g_dense19"), ("cca_net_off", "gnnv3l_b5d19"),
         ("gnnv3g_dense19", "cnn_ref"), ("gnnv3l_b5d19", "cnn_ref"), ("fbttcca_yf", "gnnv3g_dense19")]
FUSE = {"cnnref": ["cnnref"], "gnnv2c_g2": ["gnnv2c_g2"], "dense19": ["gnnv3g_dense19"], "b5d19": ["gnnv3l_b5d19"], "b5d9 (9 ch)": ["gnnv3l_b5d9"], "lr (9 ch)": ["gnnv3e_lr"],
        "dense19 x3 seeds": GROUPS["dense19"], "b5d19 x3 seeds": GROUPS["b5d19"]}

# ---- load every fold JSON ----
R = {}
for f in glob.glob(str(C.RESULTS_DIR / "*.json")):
    try: d = json.load(open(f))
    except Exception: continue
    if isinstance(d, dict) and {"subject", "method", "sl", "acc"} <= d.keys():
        R.setdefault((d["method"], d["sl"]), {})[d["subject"]] = (d["acc"] * 100, d.get("itr", np.nan), d.get("params", np.nan))
def A(m, sl):   # (n_subjects,) accuracies % in subject order, or None if incomplete
    d = R.get((m, SL[sl]), {}); return np.array([d[i][0] for i in sorted(d)]) if len(d) == 35 else None
def chans(m): g = re.search(r"(\d+)$", m); return int(g.group(1)) if g and m.startswith("gnnv3") and g.group(1) in ("9", "19", "64") else 9
def boot(df): return np.percentile([rng.choice(df, len(df)).mean() for _ in range(5000)], [2.5, 97.5])
def md(df): return "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in df.values) + "\n"

# ---- 1. all_models.csv + per_subject_acc.csv ----
rows, ps = [], []
for m in sorted({k[0] for k in R}):
    r = {"method": m, "family": L.get(m, ("other", ""))[0], "channels": chans(m), "description": L.get(m, ("", ""))[1]}
    for k, sl in SL.items():
        d = R.get((m, sl), {}); a = np.array([v[0] for v in d.values()]); i = np.array([v[1] for v in d.values()], float); p = np.array([v[2] for v in d.values()], float)
        r.update({f"n_{k}": len(d), f"acc_{k}": a.mean() if len(a) else np.nan, f"se_{k}": a.std() / np.sqrt(len(a)) if len(a) else np.nan, f"itr_{k}": np.nanmean(i) if len(i) else np.nan})
        if len(d): r["params"] = np.nanmax(p) if np.isfinite(p).any() else r.get("params", np.nan); ps.append({"method": m, "sl": sl, **{f"S{s:02d}": d[s][0] for s in sorted(d)}})
    rows.append(r)
M = pd.DataFrame(rows).sort_values("acc_1p0", ascending=False, na_position="last").round(2); M.to_csv(OUT / "all_models.csv", index=False)
pd.DataFrame(ps).round(4).to_csv(OUT / "per_subject_acc.csv", index=False)

# ---- 2. seeds.csv ----
S = []
for k in SL:
    g = {n: [A(m, k) for m in ms] for n, ms in GROUPS.items()}
    if any(x is None for v in g.values() for x in v): continue
    b, d = np.array(g["b5d19"]), np.array(g["dense19"]); df = b.mean(0) - d.mean(0); ci = boot(df)
    for n, v in (("dense19", d), ("b5d19", b)): S.append({"window": k, "group": n, "seed_means": " / ".join(f"{x:.2f}" for x in v.mean(1)), "mean": v.mean(), "sd_of_seed_means": v.mean(1).std(ddof=1)})
    S.append({"window": k, "group": "b5d19 - dense19", "seed_means": f"CI [{ci[0]:.2f}, {ci[1]:.2f}]  wins {int((df > 0).sum())}/35  Wilcoxon p={wilcoxon(b.mean(0), d.mean(0)).pvalue:.3f}  Welch p={ttest_ind(b.mean(1), d.mean(1), equal_var=False).pvalue:.3f}", "mean": df.mean(), "sd_of_seed_means": np.nan})
SD = pd.DataFrame(S).round(3); SD.to_csv(OUT / "seeds.csv", index=False)

# ---- 3. tests.csv (paired Wilcoxon + bootstrap CI; p-values are NOMINAL, ~85 tests were run in the project) ----
T = []
for a_, b_ in PAIRS:
    for k in SL:
        x, y = A(a_, k), A(b_, k)
        if x is None or y is None: continue
        df = x - y; ci = boot(df); T.append({"A": a_, "B": b_, "window": k, "diff_A_minus_B": df.mean(), "ci_lo": ci[0], "ci_hi": ci[1], "A_ahead": int((df > 0).sum()), "p_wilcoxon": wilcoxon(x, y).pvalue if np.any(df) else 1.0})
TD = pd.DataFrame(T).round(3); TD.to_csv(OUT / "tests.csv", index=False)

# ---- 4. subjects.csv ----
U = []
for m in ["cca_net_off", "fbttcca_yf", "cnn_ref", "gnnv3e_lr", "gnnv3g_dense19", "gnnv3l_b5d19"]:
    for k in SL:
        a = A(m, k)
        if a is not None: U.append({"method": m, "window": k, "all": a.mean(), "experienced_S01-08": a[[s - 1 for s in EXP]].mean(), "naive_S09-35": a[[s - 1 for s in NAIVE]].mean(),
                                   "hard4_S11_19_29_33": a[[s - 1 for s in HARD]].mean(), "other31": np.delete(a, [s - 1 for s in HARD]).mean()})
UD = pd.DataFrame(U).round(2); UD.to_csv(OUT / "subjects.csv", index=False)

# ---- 5. fusion.csv (needs logits + ccafeats; exact official decision rule via zs_fuse_ens) ----
FD = pd.DataFrame()
try:
    from zs_fuse_ens import ens_acc
    from zs_fuse import itr as itr_fn
    F_ = []
    for k, sl in SL.items():
        ref, g0 = ens_acc(["cnnref"], sl)
        for name, tags in FUSE.items():
            r, got = ens_acc(tags, sl)
            if len(got) != 35: continue
            row = {"net": name, "window": k, "network_alone": r["net"].mean() * 100, "CCA_only_f2f3": r["cca"].mean() * 100, "fuse4": r["fuse4"].mean() * 100, "fuse_full": r["fuse_full"].mean() * 100,
                   "fuse4_ITR": itr_fn(r["fuse4"], sl + 0.5).mean()}
            if len(g0) == 35 and name != "cnnref":
                for q in ("fuse4", "fuse_full"):
                    df = r[q] - ref[q]; row[f"{q}_minus_cnn"] = df.mean() * 100; row[f"{q}_p"] = wilcoxon(r[q], ref[q]).pvalue if np.any(df) else 1.0
            F_.append(row)
    FD = pd.DataFrame(F_).round(3); FD.to_csv(OUT / "fusion.csv", index=False)
except ImportError as e: print("fusion skipped (zs_fuse.py / zs_fuse_ens.py not found):", e)

# ---- 6. report_tables.md ----
def tab(ms, title):
    d = M[M.method.isin(ms)].copy(); d["0.5 s"] = d.apply(lambda r: f"{r.acc_0p5:.2f} +- {r.se_0p5:.2f}" if r.n_0p5 == 35 else "-", axis=1); d["1.0 s"] = d.apply(lambda r: f"{r.acc_1p0:.2f} +- {r.se_1p0:.2f}" if r.n_1p0 == 35 else "-", axis=1)
    return f"### {title}\n" + md(d.sort_values("acc_1p0", ascending=False)[["method", "description", "channels", "0.5 s", "1.0 s"]])
v3 = [m for m, v in L.items() if v[0] == "v3"]; old = [m for m, v in L.items() if v[0] in ("v1", "v2")]
with open(OUT / "report_tables.md", "w") as o:
    o.write("# Consolidated results (35-fold leave-one-subject-out, mean accuracy % +- SE; zero-shot, Benchmark dataset)\nNominal p-values; about 85 paired tests were run. 9-channel rows are protocol-comparable with CCA-Net; 19/64-channel rows are a separate table.\n\n")
    o.write(tab([m for m, v in L.items() if v[0] == "baseline"], "Table 1: baselines")); o.write("\n" + tab([m for m in v3 if chans(m) == 9], "Table 2: GNN family, 9 channels (protocol-comparable)"))
    o.write("\n" + tab([m for m in v3 if chans(m) != 9], "Table 3: GNN family, 19 / 64 channels (separate table)")); o.write("\n" + tab(old, "Table 4: earlier GNN generations (v1/v2)"))
    o.write("\n### Table 5: seeds (b5d19 vs dense19)\n" + md(SD)); o.write("\n### Table 6: hybrid GNN + CCA fusion (fuse4 = net+(f2+f3)/2, official rule; full adds FBCCA sin/cos)\n" + (md(FD) if len(FD) else "not available\n"))
    o.write("\n### Table 7: subject groups (1.0 s / 0.5 s)\n" + md(UD)); o.write("\n### Table 8: paired tests (nominal p)\n" + md(TD))
try:
    with pd.ExcelWriter(OUT / "all_results.xlsx") as xw:
        for n, d in (("all_models", M), ("seeds", SD), ("fusion", FD), ("subjects", UD), ("tests", TD)): d.to_excel(xw, sheet_name=n, index=False)
except Exception as e: print("xlsx skipped:", e)
print("wrote", sorted(p.name for p in OUT.iterdir()), "\n", M[["method", "n_0p5", "acc_0p5", "n_1p0", "acc_1p0"]].head(12).to_string(index=False))
