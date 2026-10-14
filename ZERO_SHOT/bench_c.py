import torch, time
from zs_gnn_v2e import BandGCNLayer
d='cuda'; L=BandGCNLayer(64,64).to(d); Lc=torch.compile(L)
H=torch.randn(128,3,64,9,250,device=d,requires_grad=True)
def t(f):
    for _ in range(3): f(H).sum().backward()
    torch.cuda.synchronize(); s=time.time()
    for _ in range(20): f(H).sum().backward()
    torch.cuda.synchronize(); return (time.time()-s)/20
print('eager',t(L),'compiled',t(Lc),'maxdiff',(L.forward_ref(H)-Lc(H)).abs().max().item())
