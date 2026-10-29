"""zs_gnn_v3m_bank5seed.py -- seed replication of b5d19 (dense19 + 5-band bank C). Tag gnnv3m.
sNNb5d19 = gnnv3l b5d19 with C.SEED=NN. Names are not parsed; build_model always returns b5d19."""
import sys
import numpy as np, torch
import zs_config as C
import zs_gnn_v3g_graph as G
import zs_gnn_v3l_bank5 as L          # importing it installs the bank-C data shim on K.D
import zs_gnn_v3_common as K

TAG, VARIANTS = "gnnv3m", ("s42b5d19", "s43b5d19")

def build_model(variant, T, freqs=None, phases=None):
    if variant not in VARIANTS: raise ValueError(variant)
    return L.build_model("b5d19", T, freqs, phases)

def _run(tag, bm, sid, sl, variant, device):
    old = C.SEED; C.SEED = int(variant[1:3]); L.CH = G.chans(19)      # L.CH selects the 19-channel bank-C cache
    try: return G._run_fold(tag, bm, sid, sl, variant, device)         # G._run_fold = the original K.run_fold
    finally: C.SEED = old
K.run_fold = _run

if __name__ == "__main__":
    if "--smoke" in sys.argv:
        m = build_model(VARIANTS[0], 250, 8.0 + 0.2 * np.arange(40), C.PHASE).to(K.DEVICE)
        assert m(torch.randn(4, 5, 19, 250, device=K.DEVICE)).shape == (4, 40); print("PASSED"); sys.exit(0)
    K.cli(TAG, VARIANTS, build_model)
