"""zs_gnn_v3h_anat.py -- fgu with a FIXED anatomical adjacency (inverse electrode distance, correct polar->xy), learned scalar gate. Tag gnnv3h."""
import numpy as np, torch, torch.nn as nn, sys
import zs_config as C, zs_elec as E
import zs_gnn_v3b_freq2 as B
import zs_gnn_v3_common as K

TAG, VARIANTS = "gnnv3h", ("fga",)

def anat_adj():
    xy = E.load_loc()[1][C.CHANNELS]
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1)); np.fill_diagonal(d, np.inf)
    A = 1.0 / d; return A / A.sum(1, keepdims=True)

class AGCN(nn.Module):
    def __init__(self, A, lin):
        super().__init__(); self.learned = True; self.lin = lin
        self.gate = nn.Parameter(torch.zeros(1)); self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
    def adjacency(self): return self.A
    def forward(self, x): return self.lin(x + self.gate * torch.einsum("nm,bkmf->bknf", self.A, x))

def build_model(variant, T, freqs=None, phases=None):
    if variant not in VARIANTS: raise ValueError(variant)
    m = B.GNNv3b(T, True, freqs, phases); A = anat_adj()
    for nm in ("g1", "g2"): setattr(m, nm, AGCN(A, getattr(m, nm).lin))
    return m

if __name__ == "__main__":
    if "--smoke" in sys.argv:
        K.smoke(TAG, VARIANTS, build_model, freqs=8.0 + 0.2 * np.arange(40), phases=C.PHASE); sys.exit(0)
    K.cli(TAG, VARIANTS, build_model)
