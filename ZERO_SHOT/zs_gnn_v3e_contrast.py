"""zs_gnn_v3e_contrast.py -- is the gain a plain spatial contrast? No graph, no gate. Tag gnnv3e.
cm: x - mean over channels (fixed, full strength) before each layer.  lr: learned 9x9 map (init identity) before each layer."""
import numpy as np, torch, torch.nn as nn, sys
import zs_config as C
import zs_gnn_v3b_freq2 as B
import zs_gnn_v3_common as K

TAG, VARIANTS = "gnnv3e", ("cm", "lr")

class Contrast(nn.Module):
    def __init__(self, n, lin, learned_map):
        super().__init__(); self.learned = False; self.lin = lin; self.n = n    # learned=False -> export_graph writes gates=None
        self.M = nn.Parameter(torch.eye(n)) if learned_map else None
    def adjacency(self): return self.M if self.M is not None else torch.eye(self.n) - 1.0 / self.n
    def forward(self, x):
        x = torch.einsum("nm,bkmf->bknf", self.M, x) if self.M is not None else x - x.mean(2, keepdim=True)
        return self.lin(x)

def build_model(variant, T, freqs=None, phases=None):
    if variant not in VARIANTS: raise ValueError(variant)
    m = B.GNNv3b(T, False, freqs, phases)
    for nm in ("g1", "g2"): setattr(m, nm, Contrast(m.N, getattr(m, nm).lin, variant == "lr"))
    return m

if __name__ == "__main__":
    if "--smoke" in sys.argv:
        K.smoke(TAG, VARIANTS, build_model, freqs=8.0 + 0.2 * np.arange(40), phases=C.PHASE); sys.exit(0)
    K.cli(TAG, VARIANTS, build_model)
