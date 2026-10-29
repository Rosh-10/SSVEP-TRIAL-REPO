"""zs_gnn_v3i_cap.py -- capacity / training-length test on dense19. Tag gnnv3i.
wide19: layer widths 64/32 per node, head 128 (~85k params).  long19: dense19 unchanged, 200 epochs instead of 100."""
import sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import zs_config as C
import zs_gnn_v3g_graph as G          # importing it installs the multi-channel data shim on K
import zs_gnn_v3b_freq2 as B
import zs_gnn_v3_common as K

TAG, VARIANTS = "gnnv3i", ("wide19", "long19")

def build_model(variant, T, freqs=None, phases=None):
    m = G.build_model("dense19", T, freqs, phases)
    if variant == "wide19":
        m.g1 = G.SignedGraph(m.N, nn.Linear(m.nb * m.H * 2, 64))
        m.g2 = G.SignedGraph(m.N, nn.Linear(64, 32))
        m.head1 = nn.Linear(m.N * 32 + B.EMB, 128); m.head2 = nn.Linear(128, 1)
    elif variant != "long19": raise ValueError(variant)
    return m

_prev = K.run_fold
def _run(tag, bm, sid, sl, variant, device):
    old = K.EPOCHS; K.EPOCHS = 200 if variant == "long19" else old
    try: return _prev(tag, bm, sid, sl, variant, device)
    finally: K.EPOCHS = old
K.run_fold = _run

if __name__ == "__main__":
    if "--smoke" in sys.argv:
        fr, ph = 8.0 + 0.2 * np.arange(40), C.PHASE
        for v in VARIANTS:
            m = build_model(v, 250, fr, ph).to(K.DEVICE); x = torch.randn(4, 3, m.N, 250, device=K.DEVICE)
            assert m(x).shape == (4, 40)
            F.cross_entropy(m(x), torch.randint(0, 40, (4,), device=K.DEVICE)).backward()
            bad = [k for k, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]; assert not bad, bad
            print(f"  {v} params={sum(p.numel() for p in m.parameters()):,} OK")
        print("PASSED"); sys.exit(0)
    K.cli(TAG, VARIANTS, build_model)
