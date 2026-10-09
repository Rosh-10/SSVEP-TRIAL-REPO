"""STEP 11d: candidate-scoring GCN (shared weights across the 40 candidate targets).
Run: python training1/step11d_gcn.py [subjects...] [--noadj]     (next to data_loader.py)
Per trial & candidate k: node features = log-SNR at f_k,2f_k,3f_k,4f_k per channel (64,4) -> 2 GCN layers -> attention pool -> score_k.
Loss: cross-entropy over the 40 scores. Protocol: leave-one-block-out, val block picks the epoch, test block untouched."""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.io import loadmat
from data_loader import load_subject, cfg

args = sys.argv[1:]; NOADJ, OCC = '--noadj' in args, '--occ' in args
assert all(a in ('--noadj', '--occ') or not a.startswith('--') for a in args), f'unknown flag in {args}'
subjects = [int(a) for a in args if not a.startswith('--')] or [1, 15, 30]
f = loadmat(cfg.PROJECT_ROOT/'data/raw/benchmark/Freq_Phase.mat')['freqs'].ravel()
idx = np.round((np.arange(1, 5)[None] * f[:, None] - 8) / 0.2).astype(int)               # (40,4) bin of m*f_k
G = np.load(cfg.PROJECT_ROOT/'graph_construction/electrode_adjacency.npz')
occ = [i for i, l in enumerate(G['labels']) if str(l).upper().startswith(('O', 'PO'))]      # same 10 channels as step11c
N = len(occ) if OCC else 64
A = G['adjacency_matrix'].astype(np.float64) + np.eye(64)
if OCC: A = A[np.ix_(occ, occ)]                                                              # induced occipital subgraph
d = A.sum(1); A_hat = torch.tensor(np.eye(N) if NOADJ else A / np.sqrt(d[:, None] * d[None]), dtype=torch.float32)

class Net(nn.Module):
    def __init__(s, fin=4, h=16):
        super().__init__(); s.l1, s.l2, s.att, s.out, s.drop = nn.Linear(fin, h), nn.Linear(h, h), nn.Linear(h, 1), nn.Linear(h, 1), nn.Dropout(0.3)
    def forward(s, x):                                           # x (B,40,64,4)
        B = x.shape[0]; x = x.reshape(B * 40, N, -1)
        h = s.drop(F.relu(A_hat @ s.l1(x)))                      # (B*40,64,h)
        h = F.relu(A_hat @ s.l2(h) + h)                          # residual
        w = torch.softmax(s.att(h), dim=1)                       # attention over nodes
        return s.out((w * h).sum(1)).reshape(B, 40)              # (B,40) candidate scores

def run(Fc, y, b, t, epochs=100):
    torch.manual_seed(t); v = (t + 1) % 6
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
    with torch.no_grad(): return (net(Z[te]).argmax(1) == y[te]).float().mean().item()

for s in subjects:
    X, yn, b = load_subject(s)
    Fc = torch.tensor(X[:, :, idx].transpose(0, 2, 1, 3)); y = torch.tensor(yn)           # (240,40,64,4)
    if OCC: Fc = Fc[:, :, occ]
    assert Fc.shape == (240, 40, N, 4) and torch.isfinite(Fc).all()
    accs = [run(Fc, y, b, t) for t in range(6)]
    print(f'S{s} [{"occ" if OCC else "all64"} {"no graph" if NOADJ else "GCN"}]: per-block {np.round(accs, 3)} | mean {np.mean(accs):.3f} +- {np.std(accs):.3f}')