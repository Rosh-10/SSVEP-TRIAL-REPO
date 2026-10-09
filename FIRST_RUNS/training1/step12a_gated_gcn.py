"""STEP 11d: candidate-scoring GCN (shared weights across the 40 candidate targets).
Run: python training1/step11d_gcn.py [subjects...] [--noadj]     (next to data_loader.py)
Per trial & candidate k: node features = log-SNR at f_k,2f_k,3f_k,4f_k per channel (64,4) -> 2 GCN layers -> attention pool -> score_k.
Loss: cross-entropy over the 40 scores. Protocol: leave-one-block-out, val block picks the epoch, test block untouched."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.io import loadmat
from data_loader import load_subject, cfg

args = sys.argv[1:]; NOADJ, OCC, GATE = '--noadj' in args, '--occ' in args, '--gate' in args
GRAPH = next((a[8:] for a in args if a.startswith('--graph=')), 'electrode_adjacency'); SEED = int(next((a[7:] for a in args if a.startswith('--seed=')), 0))
assert all(a in ('--noadj', '--occ', '--gate') or a.startswith(('--graph=', '--seed=', '--init=')) or not a.startswith('--') for a in args), f'unknown flag in {args}'
assert not (NOADJ and GATE), '--gate and --noadj are mutually exclusive'
subjects = [int(a) for a in args if not a.startswith('--')] or [1, 15, 30]
f = loadmat(cfg.PROJECT_ROOT/'data/raw/benchmark/Freq_Phase.mat')['freqs'].ravel()
idx = np.round((np.arange(1, 5)[None] * f[:, None] - 8) / 0.2).astype(int)               # (40,4) bin of m*f_k
G = np.load(cfg.PROJECT_ROOT/'graph_construction/electrode_adjacency.npz')
occ = [i for i, l in enumerate(G['labels']) if str(l).upper().startswith(('O', 'PO'))]      # same 10 channels as step11c
N = len(occ) if OCC else 64
A = np.load(cfg.PROJECT_ROOT/'graph_construction'/f'{GRAPH}.npz')['adjacency_matrix'].astype(np.float64); A = (A + A.T > 0) * 1.0; np.fill_diagonal(A, 0)   # swappable graph: symmetric 0/1, no self-loops
A0 = A[np.ix_(occ, occ)] if OCC else A; A_nb = torch.tensor(A0 / np.maximum(A0.sum(1, keepdims=True), 1), dtype=torch.float32)   # neighbour mean, no self
A = A + np.eye(64)
if OCC: A = A[np.ix_(occ, occ)]                                                              # induced occipital subgraph
d = A.sum(1); A_hat = torch.tensor(np.eye(N) if NOADJ else A / np.sqrt(d[:, None] * d[None]), dtype=torch.float32)

GATE_INIT = float(next((a[7:] for a in args if a.startswith('--init=')), -1.))   # default sigmoid(-1)=0.27; --init=-20 = graph off (reproduction check)
class Net(nn.Module):
    def __init__(s, fin=4, h=16):
        super().__init__(); s.l1, s.l2, s.att, s.out, s.drop = nn.Linear(fin, h), nn.Linear(h, h), nn.Linear(h, 1), nn.Linear(h, 1), nn.Dropout(0.3)
        if GATE:
            with torch.random.fork_rng(devices=[]): s.n1, s.n2 = nn.Linear(fin, h), nn.Linear(h, h)   # fork: RNG stream stays identical to the non-gated net
            s.gam = nn.Parameter(torch.full((2,), GATE_INIT))
    def forward(s, x):                                           # x (B,40,64,4)
        B = x.shape[0]; x = x.reshape(B * 40, N, -1)
        if GATE:
            g = torch.sigmoid(s.gam)
            h = s.drop(F.relu(s.l1(x) + g[0] * A_nb @ s.n1(x)))
            h = F.relu(s.l2(h) + g[1] * A_nb @ s.n2(h) + h)
            return s.out((torch.softmax(s.att(h), dim=1) * h).sum(1)).reshape(B, 40)
        h = s.drop(F.relu(A_hat @ s.l1(x)))                      # (B*40,64,h)
        h = F.relu(A_hat @ s.l2(h) + h)                          # residual
        w = torch.softmax(s.att(h), dim=1)                       # attention over nodes
        return s.out((w * h).sum(1)).reshape(B, 40)              # (B,40) candidate scores

def run(Fc, y, b, t, epochs=100):
    torch.manual_seed(t + 100 * SEED); v = (t + 1) % 6
    tr, va, te = (torch.tensor(m) for m in (~np.isin(b, [t, v]), b == v, b == t))
    mu, sd = Fc[tr].mean((0, 1)), Fc[tr].std((0, 1)) + 1e-6      # train-only statistics, (64,4)
    Z = (Fc - mu) / sd
    net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2); best = (-1, None)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(int(tr.sum())); Xtr, ytr = Z[tr], y[tr]
        for i in range(0, len(perm), 32):
            j = perm[i:i + 32]; opt.zero_grad(); F.cross_entropy(net(Xtr[j]), ytr[j]).backward(); opt.step()
        net.eval()
        with torch.no_grad(): a = (net(Z[va]).argmax(1) == y[va]).float().mean().item()
        if a > best[0]: best = (a, {k: p.clone() for k, p in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad():
        acc = (net(Z[te]).argmax(1) == y[te]).float().mean().item(); a0 = None
        if GATE: g0 = net.gam.clone(); net.gam.fill_(-20.); a0 = (net(Z[te]).argmax(1) == y[te]).float().mean().item(); net.gam.copy_(g0)   # test-time ablation: graph path off
    return acc, a0, (torch.sigmoid(net.gam).tolist() if GATE else None)

for s in subjects:
    X, yn, b = load_subject(s)
    Fc = torch.tensor(X[:, :, idx].transpose(0, 2, 1, 3)); y = torch.tensor(yn)           # (240,40,64,4)
    if OCC: Fc = Fc[:, :, occ]
    assert Fc.shape == (240, 40, N, 4) and torch.isfinite(Fc).all()
    res = [run(Fc, y, b, t) for t in range(6)]; accs = [r[0] for r in res]
    print(f'S{s} [{"occ" if OCC else "all64"} {"no graph" if NOADJ else "gated GCN" if GATE else "GCN"} {GRAPH} seed{SEED}]: per-block {np.round(accs, 3)} | mean {np.mean(accs):.3f} +- {np.std(accs):.3f}')
    if GATE: print(f'   ablated (g=0) mean {np.mean([r[1] for r in res]):.3f} | g layer1/layer2 mean {np.round(np.mean([r[2] for r in res], 0), 3).tolist()}')