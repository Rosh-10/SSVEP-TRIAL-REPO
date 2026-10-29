"""zs_gnn_v3g_graph.py -- signed graph over N channels (9/19/64), frequency-aware candidate scoring. Tag gnnv3g.
knn{N}  : M = I + W*mask, mask = k nearest electrodes (k=4, real positions from zs_elec), W signed, init 0
dense{N}: M = I + W (free signed map, like gnnv3e lr but N nodes)
Layer: x -> lin(M x). Front end, z-scoring over candidates and recipe = gnnv3b3/gnnv3e."""
import re, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import zs_config as C, zs_data as D, zs_data64 as D64, zs_elec as E
import zs_gnn_v3b_freq2 as B
import zs_gnn_v3_common as K

TAG = "gnnv3g"
VARIANTS = ("knn9", "knn19", "knn64", "dense19", "dense64")
RED = {9: 32, 19: 16, 64: 8}     # per-node features after layer 2 (keeps the flatten small); fixed a priori
K_NN = 4

def chans(n): return {9: list(C.CHANNELS), 19: E.region(), 64: list(range(64))}[n]

def knn_mask(xy, k=K_NN):
    d = np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1)); np.fill_diagonal(d, np.inf)
    m = np.zeros_like(d); np.put_along_axis(m, np.argsort(d, 1)[:, :k], 1.0, 1)
    return np.maximum(m, m.T)                                   # symmetric neighbour structure

class SignedGraph(nn.Module):
    def __init__(self, n, lin, mask=None):
        super().__init__(); self.lin, self.learned = lin, False
        self.W = nn.Parameter(torch.zeros(n, n))                # starts as the no-mixing model
        self.register_buffer("I", torch.eye(n))
        self.register_buffer("mask", torch.ones(n, n) if mask is None else torch.tensor(mask, dtype=torch.float32))
    def adjacency(self): return self.I + self.W * self.mask
    def forward(self, x): return self.lin(torch.einsum("nm,bkmf->bknf", self.adjacency(), x))

class Net(B.GNNv3b):
    @torch.no_grad()
    def export_graph(self):
        return dict(layer1=self.g1.adjacency().cpu().tolist(), layer2=self.g2.adjacency().cpu().tolist(), channels=self.idx)

def build_model(variant, T, freqs=None, phases=None):
    g, n = re.match(r"(knn|dense)(\d+)$", variant).groups(); idx = chans(int(n)); N = len(idx)
    m = Net(T, False, freqs, phases, n_nodes=N); m.idx = idx
    mask = knn_mask(E.load_loc()[1][idx]) if g == "knn" else None
    r = RED[N]
    m.g1 = SignedGraph(N, m.g1.lin, mask)
    m.g2 = SignedGraph(N, nn.Linear(B.HID, r), mask)
    m.head1 = nn.Linear(N * r + B.EMB, 64)
    return m

CH = None
class _D:                                                       # shim: same API as zs_data, channels chosen by the running variant
    make_fold = staticmethod(D.make_fold)
    @staticmethod
    def prep(sid, sl): return D64.prep64(sid, sl, CH)
_run_fold = K.run_fold
def _run(tag, bm, sid, sl, variant, device):
    global CH; CH = chans(int(re.search(r"\d+$", variant).group())); return _run_fold(tag, bm, sid, sl, variant, device)
K.D, K.run_fold = _D, _run

def smoke():
    fr, ph = 8.0 + 0.2 * np.arange(40), C.PHASE
    for v in VARIANTS:
        for T in (125, 250):
            m = build_model(v, T, fr, ph).to(K.DEVICE); x = torch.randn(4, 3, m.N, T, device=K.DEVICE)
            out = m(x); assert out.shape == (4, 40), out.shape
            F.cross_entropy(m(x), torch.randint(0, 40, (4,), device=K.DEVICE)).backward()
            bad = [k for k, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]; assert not bad, bad
            print(f"  {v} T={T} N={m.N} params={sum(p.numel() for p in m.parameters()):,} OK")
    print("PASSED")

if __name__ == "__main__":
    if "--smoke" in sys.argv: smoke(); sys.exit(0)
    K.cli(TAG, VARIANTS, build_model)
