"""zs_gnn_v3j_adapt.py -- dense signed graph + trial-adaptive channel weights. Tag gnnv3j.
aw9 / aw19: w = N*softmax(beta*s + bias), s = per-trial in-band energy per channel, standardised across channels; beta, bias init 0
(uniform weights -> identical to dense9 / dense19 at init). Weights applied after the cross-candidate z-score, before graph layers."""
import sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import zs_config as C
import zs_gnn_v3g_graph as G          # installs the multi-channel data shim; variant name's trailing digits pick the channel set
import zs_gnn_v3_common as K

TAG, VARIANTS = "gnnv3j", ("aw9", "aw19")

class AW(G.Net):
    def forward(self, x):                                              # x (B,nb,N,T)
        B = x.shape[0]
        z = torch.matmul(x, self.basis)                                # (B,nb,N,K*H*2)
        s = (z ** 2).sum(-1).mean(1)                                   # (B,N) in-band energy per channel
        z = z.view(B, self.nb, self.N, self.Kc, self.H, 2).permute(0, 3, 2, 1, 4, 5)
        z = z.reshape(B, self.Kc, self.N, self.nb * self.H * 2)
        z = (z - z.mean(1, keepdim=True)) / (z.std(1, keepdim=True) + 1e-6)
        s = (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-6)
        w = self.N * torch.softmax(self.beta * s + self.bias, dim=1)   # (B,N); all ones at init
        z = z * w[:, None, :, None]
        z = F.elu(self.g1(z)); z = F.elu(self.g2(z))
        z = z.reshape(B, self.Kc, -1)
        e = self.cls_emb.weight.unsqueeze(0).expand(B, -1, -1)
        z = self.drop(F.elu(self.head1(torch.cat([z, e], -1))))
        return self.head2(z).squeeze(-1)

    @torch.no_grad()
    def export_graph(self):
        d = super().export_graph(); d["beta"] = float(self.beta); d["bias"] = self.bias.cpu().tolist(); return d

def build_model(variant, T, freqs=None, phases=None):
    if variant not in VARIANTS: raise ValueError(variant)
    m = G.build_model("dense" + variant[2:], T, freqs, phases); m.__class__ = AW
    m.beta = nn.Parameter(torch.zeros(1)); m.bias = nn.Parameter(torch.zeros(m.N))
    return m

if __name__ == "__main__":
    if "--smoke" in sys.argv:
        fr, ph = 8.0 + 0.2 * np.arange(40), C.PHASE
        for v in VARIANTS:
            m1 = build_model(v, 250, fr, ph).to(K.DEVICE).eval(); m0 = G.build_model("dense" + v[2:], 250, fr, ph).to(K.DEVICE).eval()
            m1.load_state_dict(m0.state_dict(), strict=False)
            x = torch.randn(4, 3, m1.N, 250, device=K.DEVICE)
            assert torch.allclose(m0(x), m1(x), atol=1e-5), "does not start as the dense model"
            m1.train(); F.cross_entropy(m1(x), torch.randint(0, 40, (4,), device=K.DEVICE)).backward()
            bad = [k for k, p in m1.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]; assert not bad, bad
            print(f"  {v} N={m1.N} params={sum(p.numel() for p in m1.parameters()):,} starts-as-dense OK")
        print("PASSED"); sys.exit(0)
    K.cli(TAG, VARIANTS, build_model)
