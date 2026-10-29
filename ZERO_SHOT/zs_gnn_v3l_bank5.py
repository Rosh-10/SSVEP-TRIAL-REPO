"""zs_gnn_v3l_bank5.py -- dense signed graph, 5-band filter bank (bank C). Tag gnnv3l.
b5d9 / b5d19 = dense9 / dense19 with n_bands=5 (input (5,N,T)). Per-subject cache for the 9 or 19 channels."""
import sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import zs_config as C, zs_data as D
import zs_gnn_v3b_freq2 as B
import zs_gnn_v3g_graph as G
import zs_gnn_v3_common as K

TAG, VARIANTS = "gnnv3l", ("b5d9", "b5d19")

def prepC(sid, sl, idx):
    f = C.CACHE_DIR / f"S{sid}_BC{len(idx)}_{str(sl).replace('.', 'p')}.npy"
    if not f.exists():
        s, e, T = D.window_idx(sl)
        w = D.load_raw64(sid)[:, s:e][idx].astype(np.float64)                 # (n,T,40,6)
        X = w.transpose(2, 3, 0, 1).reshape(C.N_CLS * C.N_BLK, len(idx), T)   # (240,n,T), same trial order as D.prep
        C.CACHE_DIR.mkdir(exist_ok=True, parents=True)
        tmp = f.with_suffix(".tmp.npy"); np.save(tmp, D.fb_stand(D.apply_bank_C(X)).astype(np.float32))
        for _ in range(15):                                                   # /mnt/c can briefly lock a just-written file
            try: tmp.replace(f); break
            except PermissionError: time.sleep(2)
        else: raise RuntimeError(f"cannot rename {tmp}")
    return dict(A=np.load(f), y=np.repeat(np.arange(C.N_CLS), C.N_BLK))      # A (240,5,n,T)

def build_model(variant, T, freqs=None, phases=None):
    if variant not in VARIANTS: raise ValueError(variant)
    idx = G.chans(int(variant[3:])); N = len(idx); r = G.RED[N]
    m = G.Net(T, False, freqs, phases, n_nodes=N, n_bands=5); m.idx = idx
    m.g1 = G.SignedGraph(N, m.g1.lin, None)
    m.g2 = G.SignedGraph(N, nn.Linear(B.HID, r), None)
    m.head1 = nn.Linear(N * r + B.EMB, 64)
    return m

CH = None
class _D:                                                   # shim: bank-C data for the channel set of the running variant
    make_fold = staticmethod(D.make_fold)
    @staticmethod
    def prep(sid, sl): return prepC(sid, sl, CH)
def _run(tag, bm, sid, sl, variant, device):
    global CH; CH = G.chans(int(variant[3:])); return G._run_fold(tag, bm, sid, sl, variant, device)
K.D, K.run_fold = _D, _run

if __name__ == "__main__":
    if "--cache" in sys.argv:                               # CPU only: pre-build all caches
        for n in (9, 19):
            for sl in (0.5, 1.0):
                for sid in C.SUBJECTS: prepC(sid, sl, G.chans(n)); print(f"S{sid:02d} {n}ch sl={sl} cached", flush=True)
        sys.exit(0)
    if "--smoke" in sys.argv:
        fr, ph = 8.0 + 0.2 * np.arange(40), C.PHASE
        for v in VARIANTS:
            for T in (125, 250):
                m = build_model(v, T, fr, ph).to(K.DEVICE); x = torch.randn(4, 5, m.N, T, device=K.DEVICE)
                assert m(x).shape == (4, 40)
                F.cross_entropy(m(x), torch.randint(0, 40, (4,), device=K.DEVICE)).backward()
                bad = [k for k, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]; assert not bad, bad
                print(f"  {v} T={T} N={m.N} params={sum(p.numel() for p in m.parameters()):,} OK")
        print("PASSED"); sys.exit(0)
    K.cli(TAG, VARIANTS, build_model)
