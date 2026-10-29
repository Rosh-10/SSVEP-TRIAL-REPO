"""zs_compile_all.py -- compile results of every model: CNN baselines, GNN variants, and
all CCA/fusion combinations into a single set of tables and an Excel workbook.

Run from ZERO_SHOT/  (same directory as zs_config.py):
    python3 zs_compile_all.py [--out results/compiled] [--ref cnnref]

Reads
  results/*.json          per-fold accuracy/ITR records (subject, method, sl, acc, itr)
  results/logits_*.npz   per-fold logits for fusion  (keys: logits, y)
  results/ccafeats_*.npz per-fold CCA feature scores  (keys: f1, f2, f3, y)

Writes (to --out, default results/compiled/)
  all_models.csv          one row per (method, window)
  per_subject_acc.csv     one row per (method, window, subject)
  seeds.csv               3-seed comparison dense19 vs b5d19
  fusion.csv              hybrid GNN+CCA rows (needs logits + ccafeats)
  subjects.csv            experienced / naive / hard-4 breakdown
  tests.csv               paired Wilcoxon for every pre-specified pair
  report_tables.md        ready-to-paste markdown tables
  all_results.xlsx        all sheets in one Excel file (needs openpyxl)

Notes
  - p-values are NOMINAL; about 85 tests were run on the same 35 subjects.
  - 9-channel rows are protocol-comparable with CCA-Net; 19 / 64-channel rows are separate.
  - Rows whose description contains INVALID are kept in CSVs but flagged in markdown.
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_ind

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=None, help="output directory (default: results/compiled)")
ap.add_argument("--ref", default="cnnref", help="reference tag for paired fusion tests")
args = ap.parse_args()

# ---------------------------------------------------------------------------
# Locate ZERO_SHOT/ via zs_config
# ---------------------------------------------------------------------------
try:
    import zs_config as C
    RESULTS = C.RESULTS_DIR
except ModuleNotFoundError:
    RESULTS = Path("results")
    print("[warn] zs_config not found; assuming results/ is ./results/", file=sys.stderr)

OUT = Path(args.out) if args.out else RESULTS / "compiled"
OUT.mkdir(parents=True, exist_ok=True)

SL_MAP = {"0p5": 0.5, "1p0": 1.0}   # key used in filenames
rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Method registry
#   key  : method string stored in the JSON  (also the logits/features tag)
#   value: (family, n_channels, description)
# INVALID in the description = keep in CSVs, flag in markdown, never use as evidence
# ---------------------------------------------------------------------------
REGISTRY = {
    # ---- classical baselines ---------------------------------------------------
    "fbcca":                ("baseline",  9, "FBCCA (classical)"),
    "fbttcca_yf":           ("baseline",  9, "FBtt-CCA (classical)"),
    # ---- CCA-Net replication ---------------------------------------------------
    "cnn_ref":              ("cnn",       9, "CCA-Net CNN branch alone (CSDuDoFN-like)"),
    "cca_net_off":          ("cnn+cca",   9, "CCA-Net replication: CNN + Modified FBtt-CCA + FBCCA(SST)"),
    # ---- early GNN generations (v1 / v2) ---------------------------------------
    "gnn":                  ("v1",        9, "v1: first conv+GCN"),
    "gnnv2_g2":             ("v2",        9, "v2: conv+GCN, learned graph"),
    "gnnv2_g3":             ("v2",        9, "v2: conv, no graph"),
    "gnnv2b_g2":            ("v2",        9, "v2b: GroupNorm, graph"),
    "gnnv2b_g3":            ("v2",        9, "v2b: GroupNorm, no graph"),
    "gnnv2c_g2":            ("v2",        9, "v2c: reference, learned graph"),
    "gnnv2c_g3":            ("v2",        9, "v2c: no graph"),
    "gnnv2d_g2_identity":   ("v2",        9, "v2d ablation: identity adjacency"),
    "gnnv2d_g2_uniform":    ("v2",        9, "v2d ablation: uniform adjacency"),
    "gnnv2d_g2_anatomical": ("v2",        9, "INVALID: v2d anatomical (polar-coordinate bug)"),
    # ---- v3 template-based GNN family (9 channels) -----------------------------
    "gnnv3a_pb":            ("v3",        9, "v3a: per-band + layer-wise adj (conv front end)"),
    "gnnv3b3_fg3":          ("v3",        9, "v3 fg3: template front end, no graph"),
    "gnnv3b3_fg2":          ("v3",        9, "v3 fg2: learned gated adjacency"),
    "gnnv3c_ng":            ("v3",        9, "v3c: per-node gate"),
    "gnnv3c_h5":            ("v3",        9, "v3c: 5 harmonics"),
    "gnnv3c_ngh5":          ("v3",        9, "v3c: per-node gate + 5 harmonics"),
    "gnnv3d_fgu":           ("v3",        9, "v3d: fixed uniform adjacency + gate"),
    "gnnv3e_cm":            ("v3",        9, "v3e: fixed channel-mean contrast"),
    "gnnv3e_lr":            ("v3",        9, "v3e lr: learned signed 9x9 spatial map (best 9-ch)"),
    "gnnv3g_knn9":          ("v3",        9, "v3g knn9: kNN signed map, 9 ch"),
    "gnnv3h_fga":           ("v3",        9, "v3h: anatomical inverse-dist adjacency + gate"),
    "gnnv3j_aw9":           ("v3",        9, "v3j: trial-adaptive node weights, 9 ch"),
    "gnnv3l_b5d9":          ("v3",        9, "v3l b5d9: dense + 5-band bank C, 9 ch"),
    # ---- v3 family (19 channels) ------------------------------------------------
    "gnnv3g_dense19":       ("v3",       19, "v3g dense19: free signed map, 19 ch  [seed 41]"),
    "gnnv3g_knn19":         ("v3",       19, "v3g knn19: kNN signed map, 19 ch"),
    "gnnv3i_wide19":        ("v3",       19, "v3i wide19: dense19 4x wider"),
    "gnnv3j_aw19":          ("v3",       19, "v3j aw19: trial-adaptive node weights, 19 ch"),
    "gnnv3k_s42d19":        ("v3",       19, "v3k: dense19 seed 42"),
    "gnnv3k_s43d19":        ("v3",       19, "v3k: dense19 seed 43"),
    "gnnv3l_b5d19":         ("v3",       19, "v3l b5d19: dense + 5-band bank C, 19 ch  [seed 41]"),
    "gnnv3m_s42b5d19":      ("v3",       19, "v3m: b5d19 seed 42"),
    "gnnv3m_s43b5d19":      ("v3",       19, "v3m: b5d19 seed 43"),
    # ---- v3 family (64 channels) ------------------------------------------------
    "gnnv3g_dense64":       ("v3",       64, "v3g dense64: free signed map, 64 ch"),
    "gnnv3g_knn64":         ("v3",       64, "v3g knn64: kNN signed map, 64 ch"),
}

# 3-seed groups for the seed comparison
SEED_GROUPS = {
    "dense19": ["gnnv3g_dense19", "gnnv3k_s42d19", "gnnv3k_s43d19"],
    "b5d19":   ["gnnv3l_b5d19",   "gnnv3m_s42b5d19", "gnnv3m_s43b5d19"],
}

# Tags used in logits files for the fusion analysis
# map display-name -> list of logit tags whose logits are averaged before fusion
FUSE_GROUPS = {
    "cnnref":           ["cnnref"],
    "GNN lr 9ch":       ["gnnv3e_lr"],
    "GNN b5d9 9ch":     ["gnnv3l_b5d9"],
    "GNN dense19":      ["gnnv3g_dense19"],
    "GNN b5d19":        ["gnnv3l_b5d19"],
    "GNN dense19 x3":   SEED_GROUPS["dense19"],
    "GNN b5d19 x3":     SEED_GROUPS["b5d19"],
}

# Pre-specified paired tests (A vs B = A minus B)
PAIRS = [
    # graph ablation
    ("gnnv3b3_fg2",   "gnnv3b3_fg3"),   # graph on vs off
    ("gnnv3d_fgu",    "gnnv3b3_fg3"),   # uniform gate vs no graph
    ("gnnv3e_lr",     "gnnv3b3_fg2"),   # free map vs gated learned adj
    ("gnnv3e_cm",     "gnnv3b3_fg3"),   # full contrast vs no graph
    ("gnnv3h_fga",    "gnnv3d_fgu"),    # anatomical vs uniform
    # channel / capacity
    ("gnnv3g_dense19","gnnv3e_lr"),      # 19ch vs 9ch best
    ("gnnv3g_dense19","gnnv3g_knn19"),   # free vs kNN at 19ch
    ("gnnv3i_wide19", "gnnv3g_dense19"), # 4x wider vs standard
    # adaptive weighting
    ("gnnv3j_aw19",   "gnnv3g_dense19"),
    ("gnnv3j_aw9",    "gnnv3e_lr"),
    # 5-band bank
    ("gnnv3l_b5d19",  "gnnv3g_dense19"),
    ("gnnv3l_b5d9",   "gnnv3e_lr"),
    # seeds
    ("gnnv3k_s42d19", "gnnv3g_dense19"),
    ("gnnv3k_s43d19", "gnnv3g_dense19"),
    # vs baselines
    ("cca_net_off",   "gnnv3g_dense19"),
    ("cca_net_off",   "gnnv3l_b5d19"),
    ("gnnv3g_dense19","cnn_ref"),
    ("gnnv3l_b5d19",  "cnn_ref"),
    ("fbttcca_yf",    "gnnv3g_dense19"),
]

# Subject groups
HARD    = [11, 19, 29, 33]
EXP     = list(range(1, 9))
NAIVE   = list(range(9, 36))

# Methods shown in subject-group breakdown
SUBJ_METHODS = ["cca_net_off", "fbttcca_yf", "cnn_ref",
                 "gnnv3e_lr", "gnnv3g_dense19", "gnnv3l_b5d19"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slkey(sl: float) -> str:
    return "0p5" if sl == 0.5 else "1p0"

def boot95(diff: np.ndarray, n: int = 5000) -> tuple:
    samples = rng.choice(diff, (n, len(diff)), replace=True).mean(1)
    return tuple(np.percentile(samples, [2.5, 97.5]))

def md_table(df: pd.DataFrame) -> str:
    hdr  = "| " + " | ".join(df.columns) + " |"
    sep  = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([hdr, sep] + rows) + "\n"

def mm(s: np.ndarray) -> np.ndarray:
    """Min-max normalise rows (official CCA-Net rule)."""
    lo = s.min(1, keepdims=True)
    rng_ = np.ptp(s, axis=1, keepdims=True)
    return (s - lo) / np.where(rng_ == 0, 1.0, rng_)

def itr_bpm(acc: np.ndarray, T: float, M: int = 40) -> np.ndarray:
    p = np.clip(acc, 1e-9, 1 - 1e-9)
    return (np.log2(M) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (M - 1))) * 60 / T

# ---------------------------------------------------------------------------
# 1. Load all per-fold JSON records
# ---------------------------------------------------------------------------
print("Loading JSON records …")
R: dict[tuple, dict] = {}   # (method, sl) -> {subject_int: (acc%, itr, params)}

for fpath in glob.glob(str(RESULTS / "*.json")):
    try:
        d = json.load(open(fpath))
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    if not {"subject", "method", "sl", "acc"} <= d.keys():
        continue
    key  = (d["method"], float(d["sl"]))
    subj = int(d["subject"])
    acc  = float(d["acc"]) * 100          # store as %
    itr  = float(d.get("itr", float("nan")))
    pars = float(d.get("params", float("nan")))
    R.setdefault(key, {})[subj] = (acc, itr, pars)

found_methods = sorted({k[0] for k in R})
print(f"  Found {len(found_methods)} unique methods in {len(R)} (method, sl) groups.")


def acc_array(method: str, sl: float) -> np.ndarray | None:
    """Return per-subject accuracy array (35,) or None if not complete."""
    d = R.get((method, sl), {})
    if len(d) < 35:
        return None
    return np.array([d[s][0] for s in sorted(d)])   # sorted by subject index


# ---------------------------------------------------------------------------
# 2. all_models.csv  +  per_subject_acc.csv
# ---------------------------------------------------------------------------
print("Building all_models.csv …")
rows_model, rows_subj = [], []

for method in sorted({k[0] for k in R}):
    fam, nchan, desc = REGISTRY.get(method, ("other", 9, method))
    row = {"method": method, "family": fam, "channels": nchan, "description": desc}
    params_seen = []

    for k, sl in SL_MAP.items():
        d = R.get((method, sl), {})
        accs = np.array([v[0] for v in d.values()])
        itrs = np.array([v[1] for v in d.values()], dtype=float)
        pars = np.array([v[2] for v in d.values()], dtype=float)
        n = len(accs)
        row[f"n_{k}"]   = n
        row[f"acc_{k}"] = round(accs.mean(), 2) if n else float("nan")
        row[f"se_{k}"]  = round(accs.std() / np.sqrt(n), 2) if n > 1 else float("nan")
        row[f"itr_{k}"] = round(float(np.nanmean(itrs)), 2) if n else float("nan")
        if np.isfinite(pars).any():
            params_seen.append(np.nanmax(pars))
        # per-subject rows
        for subj in sorted(d):
            rows_subj.append({"method": method, "sl": sl, "subject": subj,
                               "acc": round(d[subj][0], 4)})

    row["params"] = int(max(params_seen)) if params_seen else float("nan")
    rows_model.append(row)

M = (pd.DataFrame(rows_model)
       .sort_values("acc_1p0", ascending=False, na_position="last")
       .reset_index(drop=True))
M.to_csv(OUT / "all_models.csv", index=False)

PS = pd.DataFrame(rows_subj)
PS.to_csv(OUT / "per_subject_acc.csv", index=False)
print(f"  {len(M)} methods → all_models.csv")

# ---------------------------------------------------------------------------
# 3. seeds.csv  -- 3-seed comparison dense19 vs b5d19
# ---------------------------------------------------------------------------
print("Building seeds.csv …")
seed_rows = []
for k, sl in SL_MAP.items():
    arrays = {g: [acc_array(m, sl) for m in ms] for g, ms in SEED_GROUPS.items()}
    if any(a is None for v in arrays.values() for a in v):
        continue
    d19 = np.array(arrays["dense19"])   # (3, 35)
    b19 = np.array(arrays["b5d19"])
    diff = b19.mean(0) - d19.mean(0)    # per-subject difference of 3-seed means
    ci = boot95(diff)
    p_wilcox = wilcoxon(b19.mean(0), d19.mean(0)).pvalue
    p_welch  = ttest_ind(b19.mean(1),  d19.mean(1), equal_var=False).pvalue
    for name, arr in (("dense19", d19), ("b5d19", b19)):
        seed_rows.append({
            "window":   k,
            "group":    name,
            "seed41":   round(arr[0].mean(), 2),
            "seed42":   round(arr[1].mean(), 2),
            "seed43":   round(arr[2].mean(), 2),
            "mean":     round(arr.mean(), 2),
            "sd_seeds": round(arr.mean(1).std(ddof=1), 3),
        })
    seed_rows.append({
        "window":   k,
        "group":    "b5d19 - dense19",
        "seed41":   "",
        "seed42":   "",
        "seed43":   "",
        "mean":     round(diff.mean(), 3),
        "sd_seeds": "",
        "ci_lo":    round(ci[0], 3),
        "ci_hi":    round(ci[1], 3),
        "wins_35":  int((diff > 0).sum()),
        "p_wilcox": round(p_wilcox, 4),
        "p_welch":  round(p_welch, 4),
    })

SD = pd.DataFrame(seed_rows)
SD.to_csv(OUT / "seeds.csv", index=False)
print(f"  seeds.csv written ({len(seed_rows)} rows)")

# ---------------------------------------------------------------------------
# 4. tests.csv  -- paired Wilcoxon + bootstrap 95 % CI
# ---------------------------------------------------------------------------
print("Building tests.csv …")
test_rows = []
for a_tag, b_tag in PAIRS:
    for k, sl in SL_MAP.items():
        x = acc_array(a_tag, sl)
        y = acc_array(b_tag, sl)
        if x is None or y is None:
            continue
        diff = x - y
        ci   = boot95(diff)
        p    = wilcoxon(x, y).pvalue if np.any(diff) else 1.0
        test_rows.append({
            "A":           a_tag,
            "B":           b_tag,
            "window":      k,
            "A_mean":      round(x.mean(), 2),
            "B_mean":      round(y.mean(), 2),
            "diff_A-B":    round(diff.mean(), 2),
            "ci_lo":       round(ci[0], 2),
            "ci_hi":       round(ci[1], 2),
            "A_ahead_35":  int((diff > 0).sum()),
            "p_wilcoxon":  round(p, 4),
        })

TD = pd.DataFrame(test_rows)
TD.to_csv(OUT / "tests.csv", index=False)
print(f"  tests.csv written ({len(test_rows)} rows over {len(PAIRS)} pairs × 2 windows)")

# ---------------------------------------------------------------------------
# 5. fusion.csv  -- GNN + CCA hybrid results
#    Reads logits_<tag>_sXX_<sl>.npz and ccafeats_sXX_<sl>.npz
# ---------------------------------------------------------------------------
print("Building fusion.csv …")

def load_fuse_fold(tag: str, subj: int, sl: float):
    """Return (logits, y) or None."""
    k = slkey(sl)
    lp = RESULTS / f"logits_{tag}_s{subj:02d}_{k}.npz"
    if not lp.exists():
        return None
    d = np.load(lp)
    return d["logits"], d["y"]

def load_cca_fold(subj: int, sl: float):
    """Return (f1, f2, f3, y) or None."""
    k = slkey(sl)
    fp = RESULTS / f"ccafeats_s{subj:02d}_{k}.npz"
    if not fp.exists():
        return None
    d = np.load(fp)
    return d["f1"], d["f2"], d["f3"], d["y"]

def ensemble_fuse(tags: list[str], sl: float, sids=range(1, 36)):
    """Average normalised logits of several tags, then apply fusion rules.
    Returns dict of per-subject arrays, and the list of found subjects."""
    net_per_s, cca_per_s = [], []
    got = []
    for s in sids:
        # load all logits for this subject
        ldata = [load_fuse_fold(t, s, sl) for t in tags]
        if any(x is None for x in ldata):
            continue
        cca = load_cca_fold(s, sl)
        if cca is None:
            continue
        f1, f2, f3, y = cca
        # verify label consistency
        for lo, yo in ldata:
            assert (yo == y).all(), f"label mismatch s{s} tag {tags}"
        # average normalised logits
        avg_net = np.mean([mm(lo) for lo, _ in ldata], axis=0)
        net_per_s.append(avg_net)
        cca_per_s.append((mm(f1), mm(f2), mm(f3), y))
        got.append(s)
    if not got:
        return None, []

    def acc1d(scores_list):
        return np.array([(s.argmax(1) == y_).mean()
                         for s, (_, _, _, y_) in zip(scores_list, cca_per_s)])

    net_s = net_per_s
    cca_only = [f2 + f3          for _, f2, f3, _ in cca_per_s]
    fuse4    = [n + (f2 + f3)/2  for n, (_, f2, f3, _) in zip(net_s, cca_per_s)]
    full_    = [n + (f1+f2+f3)/3 for n, (f1, f2, f3, _) in zip(net_s, cca_per_s)]

    return {
        "net":       acc1d(net_s),
        "cca":       acc1d(cca_only),
        "fuse4":     acc1d(fuse4),
        "fuse_full": acc1d(full_),
    }, got

fuse_rows = []
ref_results = {}   # cache reference (cnnref) per window for paired tests

for k, sl in SL_MAP.items():
    T = sl + 0.5   # total trial time including 0.5 s gaze shift

    # load reference first
    ref_tags = FUSE_GROUPS.get(args.ref, [args.ref])
    ref_r, ref_got = ensemble_fuse(ref_tags, sl)

    if ref_r is not None:
        ref_results[k] = ref_r

    for name, tags in FUSE_GROUPS.items():
        r, got = ensemble_fuse(tags, sl)
        if r is None or len(got) != 35:
            print(f"  [fusion] {name} sl={sl}: only {len(got)}/35 folds available – skipped")
            continue

        row = {
            "net_label":   name,
            "window":      k,
            "n_folds":     len(got),
            "net_alone":   round(r["net"].mean()  * 100, 2),
            "cca_only":    round(r["cca"].mean()  * 100, 2),
            "fuse4":       round(r["fuse4"].mean()* 100, 2),
            "full":        round(r["fuse_full"].mean()*100, 2),
            "fuse4_ITR":   round(itr_bpm(r["fuse4"], T).mean(), 2),
            "full_ITR":    round(itr_bpm(r["fuse_full"], T).mean(), 2),
        }

        # paired vs reference
        if ref_r is not None and name != args.ref and len(got) == 35:
            for rule in ("fuse4", "fuse_full"):
                diff = r[rule] - ref_r[rule]
                ci = boot95(diff)
                p  = wilcoxon(r[rule], ref_r[rule]).pvalue if np.any(diff) else 1.0
                row[f"{rule}_vs_{args.ref}_diff"] = round(diff.mean() * 100, 2)
                row[f"{rule}_vs_{args.ref}_ci"]   = f"[{ci[0]*100:.2f}, {ci[1]*100:.2f}]"
                row[f"{rule}_vs_{args.ref}_p"]    = round(p, 4)
                row[f"{rule}_ahead_35"]            = int((diff > 0).sum())

        fuse_rows.append(row)

FD = pd.DataFrame(fuse_rows)
FD.to_csv(OUT / "fusion.csv", index=False)
print(f"  fusion.csv written ({len(fuse_rows)} rows)")

# ---------------------------------------------------------------------------
# 6. subjects.csv  -- experienced / naive / hard-4 breakdown
# ---------------------------------------------------------------------------
print("Building subjects.csv …")
subj_rows = []
for method in SUBJ_METHODS:
    for k, sl in SL_MAP.items():
        a = acc_array(method, sl)
        if a is None:
            continue
        hard_idx  = [s - 1 for s in HARD]
        other_idx = [i for i in range(35) if i not in hard_idx]
        exp_idx   = [s - 1 for s in EXP]
        naive_idx = [s - 1 for s in NAIVE]
        subj_rows.append({
            "method":       method,
            "description":  REGISTRY.get(method, ("", 9, method))[2],
            "window":       k,
            "all_35":       round(a.mean(), 2),
            "exp_S01-08":   round(a[exp_idx].mean(), 2),
            "naive_S09-35": round(a[naive_idx].mean(), 2),
            "hard4_mean":   round(a[hard_idx].mean(), 2),
            "hard_S11":     round(a[10], 2),
            "hard_S19":     round(a[18], 2),
            "hard_S29":     round(a[28], 2),
            "hard_S33":     round(a[32], 2),
            "other_31":     round(np.delete(a, hard_idx).mean(), 2),
        })

UD = pd.DataFrame(subj_rows)
UD.to_csv(OUT / "subjects.csv", index=False)
print(f"  subjects.csv written ({len(subj_rows)} rows)")

# ---------------------------------------------------------------------------
# 7. report_tables.md  -- ready to paste into a paper / poster
# ---------------------------------------------------------------------------
print("Building report_tables.md …")

def acc_str(row, k):
    n = row[f"n_{k}"]
    if n == 35:
        return f"{row[f'acc_{k}']:.2f} ± {row[f'se_{k}']:.2f}"
    elif n > 0:
        return f"{row[f'acc_{k}']:.2f} (n={n})"
    return "–"

def build_table(methods, title):
    sub = M[M.method.isin(methods)].copy()
    if sub.empty:
        return f"### {title}\n_no data_\n\n"
    sub["0.5 s acc ± SE"] = sub.apply(lambda r: acc_str(r, "0p5"), axis=1)
    sub["1.0 s acc ± SE"] = sub.apply(lambda r: acc_str(r, "1p0"), axis=1)
    sub["note"] = sub["description"].apply(
        lambda d: "⚠ INVALID" if "INVALID" in str(d) else "")
    disp = (sub.sort_values("acc_1p0", ascending=False, na_position="last")
               [["method", "description", "channels", "params",
                 "0.5 s acc ± SE", "1.0 s acc ± SE", "note"]])
    return f"### {title}\n" + md_table(disp) + "\n"

v3_9ch  = [m for m, (fam, ch, _) in REGISTRY.items() if fam == "v3" and ch == 9]
v3_19ch = [m for m, (fam, ch, _) in REGISTRY.items() if fam == "v3" and ch == 19]
v3_64ch = [m for m, (fam, ch, _) in REGISTRY.items() if fam == "v3" and ch == 64]
base    = [m for m, (fam, *_) in REGISTRY.items() if fam in ("baseline", "cnn", "cnn+cca")]
v1v2    = [m for m, (fam, *_) in REGISTRY.items() if fam in ("v1", "v2")]

lines = [
    "# Compiled results – 35-fold LOSO, mean accuracy % ± SE  (zero-shot, Benchmark dataset)\n",
    "> Nominal p-values; ~85 paired tests were run on the same 35 subjects.  \n",
    "> 9-channel rows are protocol-comparable with CCA-Net.  \n",
    "> 19/64-channel rows belong in a SEPARATE table with matching-channel baselines.  \n\n",
    build_table(base,   "Table 1 – Classical baselines and CCA-Net replication (9 ch)"),
    build_table(v3_9ch, "Table 2 – GNN v3 family, 9 channels (protocol-comparable)"),
    build_table(v3_19ch,"Table 3 – GNN v3 family, 19 channels (separate)"),
    build_table(v3_64ch,"Table 4 – GNN v3 family, 64 channels (separate)"),
    build_table(v1v2,   "Table 5 – Earlier GNN generations v1/v2"),
]

# --- seeds table ---
lines.append("### Table 6 – Seed comparison: dense19 vs b5d19 (3 seeds each)\n")
if not SD.empty:
    lines.append(md_table(SD) + "\n")
else:
    lines.append("_not enough seeds found_\n\n")

# --- fusion table ---
lines.append("### Table 7 – Hybrid GNN + CCA fusion  (fuse4 = net+(f2+f3)/2; full adds FBCCA sin/cos)\n")
if not FD.empty:
    lines.append(md_table(FD) + "\n")
else:
    lines.append("_no logits/ccafeats found – run zs_export_logits.py and zs_cca_feats.py first_\n\n")

# --- subject groups ---
lines.append("### Table 8 – Subject-group breakdown\n")
if not UD.empty:
    lines.append(md_table(UD) + "\n")

# --- paired tests ---
lines.append("### Table 9 – Paired tests (nominal Wilcoxon p; ~85 tests total)\n")
if not TD.empty:
    lines.append(md_table(TD) + "\n")

(OUT / "report_tables.md").write_text("".join(lines))
print("  report_tables.md written")

# ---------------------------------------------------------------------------
# 8. all_results.xlsx
# ---------------------------------------------------------------------------
try:
    xl_path = OUT / "all_results.xlsx"
    with pd.ExcelWriter(xl_path, engine="openpyxl") as xw:
        for name, df in [("all_models",   M),
                          ("per_subject",  PS),
                          ("seeds",        SD),
                          ("fusion",       FD),
                          ("subjects",     UD),
                          ("tests",        TD)]:
            if df is not None and not df.empty:
                df.to_excel(xw, sheet_name=name, index=False)
    print(f"  all_results.xlsx written ({xl_path})")
except Exception as e:
    print(f"  [warn] xlsx skipped ({e}); install openpyxl with: pip install openpyxl --break-system-packages")

# ---------------------------------------------------------------------------
# 9. Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
print("SUMMARY – top 15 methods by 1.0 s accuracy")
print("=" * 68)
cols = ["method", "channels", "n_0p5", "acc_0p5", "n_1p0", "acc_1p0"]
have = [c for c in cols if c in M.columns]
print(M[have].head(15).to_string(index=False))

print(f"\nAll outputs written to:  {OUT}/")
print("  all_models.csv        per_subject_acc.csv    seeds.csv")
print("  fusion.csv            subjects.csv           tests.csv")
print("  report_tables.md      all_results.xlsx")
print()
print("NOTE: p-values are NOMINAL (~85 tests). Only paired 35-subject tests count.")
print("      Rows marked INVALID in description should NOT be used as evidence.")
