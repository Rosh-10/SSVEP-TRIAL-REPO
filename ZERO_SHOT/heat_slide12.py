import numpy as np, torch, glob, sys, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import zs_config as C
from zs_gnn_v2c import _sym_row_norm
win = sys.argv[1] if len(sys.argv) > 1 else "1p0"
names = ["Pz","PO5","PO3","POz","PO4","PO6","O1","Oz","O2"]
fs = sorted(glob.glob(str(C.RESULTS_DIR / f"adj_gnnv2c_g2_s*_{win}.npz"))); print(len(fs), "subject(s)")
L = np.stack([_sym_row_norm(torch.tensor(np.load(f)["gcn1_A_raw"])).numpy() for f in fs]).mean(0)
P = {"Identity (A = I/9)": np.eye(9)/9, "Uniform (A = 1/9)": np.full((9, 9), 1/9),
     f"Learned (GCN layer 1, n={len(fs)} fold)": L}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
for a, (k, A) in zip(ax, P.items()):
    im = a.imshow(A, cmap="viridis", vmin=0, vmax=0.15); a.set_title(k, fontsize=10)
    a.set_xticks(range(9), names, rotation=45, fontsize=8); a.set_yticks(range(9), names, fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.02, label="edge weight")
plt.savefig("heat_slide12.png", dpi=200, bbox_inches="tight")
