import torch, time
from zs_gnn_v2e import BandGCNLayer
d = 'cuda'
for fi, fo in [(16, 64), (64, 64)]:
    L = BandGCNLayer(fi, fo).to(d)
    H = torch.randn(128, 3, fi, 9, 250, device=d, requires_grad=True)
    def t(f):
        for _ in range(3): f(H).sum().backward()
        torch.cuda.synchronize(); s = time.time()
        for _ in range(20): f(H).sum().backward()
        torch.cuda.synchronize(); return (time.time() - s) / 20
    print(fi, fo, 'ref', t(L.forward_ref), 'fast', t(L.forward),
          'maxdiff', (L.forward_ref(H) - L.forward(H)).abs().max().item())
