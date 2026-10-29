import json, glob, numpy as np
from scipy.stats import wilcoxon, ttest_ind
B = ("gnnv3l_b5d19", "gnnv3m_s42b5d19", "gnnv3m_s43b5d19"); D = ("gnnv3g_dense19", "gnnv3k_s42d19", "gnnv3k_s43d19")
def acc(m, sl):
    d = {}
    for f in glob.glob(f"results/{m}_s*_{sl}.json"):
        j = json.load(open(f)); d[j["subject"]] = j["acc"] * 100
    assert len(d) == 35, f"{m} {sl}: {len(d)} folds"
    return np.array([d[i] for i in sorted(d)])
rng = np.random.default_rng(0)
for sl in ("1p0", "0p5"):
    b, d = [np.array([acc(m, sl) for m in g]) for g in (B, D)]           # (3 seeds, 35 subjects)
    print(f"\n== {sl}  seed means  b5d19: {b.mean(1).round(2)}  dense19: {d.mean(1).round(2)}")
    print(f"   group mean (SD of seed means)  b5d19 {b.mean():.2f} ({b.mean(1).std(ddof=1):.2f})   dense19 {d.mean():.2f} ({d.mean(1).std(ddof=1):.2f})   Welch t on 3 vs 3 seed means: p={ttest_ind(b.mean(1), d.mean(1), equal_var=False).pvalue:.3f}")
    df = b.mean(0) - d.mean(0); ci = np.percentile([rng.choice(df, 35).mean() for _ in range(5000)], [2.5, 97.5])
    print(f"   per-subject 3-seed means: diff {df.mean():+.2f}  CI [{ci[0]:.2f}, {ci[1]:.2f}]  wins {int((df>0).sum())}/35  Wilcoxon p={wilcoxon(b.mean(0), d.mean(0)).pvalue:.3f}")
    ok = ci[0] > 0 and (df.mean() >= 0.5 if sl == "1p0" else True)
    print("   RULE (CI excludes 0" + (" and gain >= 0.5" if sl == "1p0" else "") + f"): {'PASS' if ok else 'FAIL'}")
