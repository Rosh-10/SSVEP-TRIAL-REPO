"""zs_gnn_v3d_unif.py -- control for v3b3 fg2: FIXED uniform adjacency (1/9), same learned scalar gate. Tag gnnv3d."""
import numpy as np, torch, torch.nn as nn, sys
import zs_config as C
import zs_gnn_v3b_freq2 as B
import zs_gnn_v3_common as K

TAG, VARIANTS = "gnnv3d", ("fgu",)

class UGCN(nn.Module):
    def __init__(self, n, lin):
        super().__init__(); self.learned = True; self.lin = lin    # reuse fg2's Linear -> same init
        self.gate = nn.Parameter(torch.zeros(1)); self.register_buffer("A", torch.full((n, n), 1.0 / n))
    def adjacency(self): return self.A
    def forward(self, x): return self.lin(x + self.gate * torch.einsum("nm,bkmf->bknf", self.A, x))

def build_model(variant, T, freqs=None, phases=None):
    if variant not in VARIANTS: raise ValueError(variant)
    m = B.GNNv3b(T, True, freqs, phases)
    for nm in ("g1", "g2"): setattr(m, nm, UGCN(m.N, getattr(m, nm).lin))
    return m

if __name__ == "__main__":
    if "--smoke" in sys.argv:
        K.smoke(TAG, VARIANTS, build_model, freqs=8.0 + 0.2 * np.arange(40), phases=C.PHASE); sys.exit(0)
    K.cli(TAG, VARIANTS, build_model)
