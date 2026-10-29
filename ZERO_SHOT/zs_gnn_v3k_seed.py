"""zs_gnn_v3k_seed.py -- seed-variance check: dense19 retrained with other seeds. Tag gnnv3k.
sNNd19 = gnnv3g dense19 with C.SEED=NN. Variant name must end in the channel count (19)."""
import sys
import numpy as np, torch
import zs_config as C
import zs_gnn_v3g_graph as G
import zs_gnn_v3_common as K

TAG, VARIANTS = "gnnv3k", ("s42d19", "s43d19", "s44d19")

def build_model(variant, T, freqs=None, phases=None):
    if variant not in VARIANTS: raise ValueError(variant)
    return G.build_model("dense19", T, freqs, phases)

_prev = K.run_fold
def _run(tag, bm, sid, sl, variant, device):
    old = C.SEED; C.SEED = int(variant[1:3])
    try: return _prev(tag, bm, sid, sl, variant, device)
    finally: C.SEED = old
K.run_fold = _run

if __name__ == "__main__":
    if "--smoke" in sys.argv:
        m = build_model(VARIANTS[0], 250, 8.0 + 0.2 * np.arange(40), C.PHASE).to(K.DEVICE)
        assert m(torch.randn(4, 3, 19, 250, device=K.DEVICE)).shape == (4, 40); print("PASSED"); sys.exit(0)
    K.cli(TAG, VARIANTS, build_model)
