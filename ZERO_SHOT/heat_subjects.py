import json, glob, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zs_config as C
tags = {"CNN": "cnn_ref", "GNN": "gnnv2c_g2"}
M = np.full((35, len(tags)), np.nan)
for j, t in enumerate(tags.values()):
    for s in range(1, 36):
        f = glob.glob(str(C.RESULTS_DIR / f"{t}_s{s:02d}_1p0*.json"))
        if f: M[s-1, j] = json.load(open(f[0]))["acc"] * 100
print("missing:", int(np.isnan(M).sum()), "(expect 0)")
fig, ax = plt.subplots(figsize=(4.5, 9)); im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(tags)), list(tags), fontsize=9); ax.set_yticks(range(35), [f"S{i}" for i in range(1, 36)], fontsize=7)
for i in range(35):
    for j in range(len(tags)): ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax, label="accuracy (%)", fraction=0.06); ax.set_title("Per-subject accuracy, 1.0 s, network alone", fontsize=9)
plt.savefig("heat_subjects.png", dpi=200, bbox_inches="tight")
