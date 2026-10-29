import json,glob,numpy as np
from scipy.stats import wilcoxon
def acc(m,sl):
    d={}
    for f in glob.glob(f"results/{m}_s*_{sl}.json"):
        j=json.load(open(f)); d[j["subject"]]=j["acc"]*100
    return np.array([d[i] for i in sorted(d)])
rng=np.random.default_rng(0)
for sl in ("0p5","1p0"):
    base=np.mean([acc(m,sl) for m in ("gnnv3g_dense19","gnnv3k_s42d19","gnnv3k_s43d19")],0)
    lr=acc("gnnv3e_lr",sl)
    for m,r in (("gnnv3l_b5d19",base),("gnnv3j_aw19",base),("gnnv3i_wide19",base),("gnnv3l_b5d9",lr),("gnnv3j_aw9",lr)):
        x=acc(m,sl); df=x-r
        ci=np.percentile([rng.choice(df,35).mean() for _ in range(5000)],[2.5,97.5]).round(2)
        print(sl,m,"d=%.2f"%df.mean(),"wins",(df>0).sum(),"p=%.3f"%wilcoxon(x,r).pvalue,"CI",ci)
