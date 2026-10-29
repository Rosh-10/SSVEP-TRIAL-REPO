"""
zs_gnn_v3b_freq.py -- frequency-aware candidate-scoring GNN (no CCA).

Idea: the label IS the stimulus frequency/phase, so make the network frequency-aware by construction.
  1. Fixed (non-trainable) sin/cos projection of every (band, channel) signal onto the 40 known stimulus
     frequencies x Nh=4 harmonics, with the known stimulus phases (cos/sin(2*pi*h*f_k*t + h*phi_k)).
     These are constants from Freq_Phase / config, NOT canonical correlation, no CCA inside.
  2. For each candidate class k the 9 channel nodes get a feature vector (bands x harmonics x {cos,sin}).
  3. Graph layers over the 9 channels, weights SHARED across candidates k (learned 9x9 adjacency like v2c).
  4. Node features are flattened (keeps channel identity), a learned class embedding is added, and a
     shared MLP outputs one score per candidate -> 40 logits.
Input/recipe identical to v2c (bank-A standardized (3,9,T), 100 ep, Adam 2e-4, bs 128, seed 41).

Variants:
  fg2 : learned symmetric 9x9 adjacency (2 layers)
  fg3 : identity adjacency (no graph) -- the matched ablation
v3b2 fix: no LayerNorm on candidate features; cross-candidate z-scoring instead.
Note: far fewer parameters than v2c (no hidden*T FC), so expect a different capacity/accuracy profile.

Usage:  python3 zs_gnn_v3b_freq.py --smoke
        python3 zs_gnn_v3b_freq.py --sl 1.0 --subjects 1 --variant fg2 fg3
        python3 zs_gnn_v3b_freq.py --sl 0.5 1.0 --variant fg2 fg3 --resume
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zs_config as C
import zs_data as D
import zs_gnn_v3_common as K

TAG = "gnnv3b2"   # new tag so old (broken, LayerNorm) S01 results are not reused by --resume
VARIANTS = ("fg2", "fg3")
HID, EMB, DROPOUT = 32, 16, 0.3


class FreqGCN(nn.Module):
    """x (B,K,N,F) -> (B,K,N,Fout); adjacency over N shared across K. learned=False -> identity."""
    def __init__(self, f_in, f_out, n_nodes, learned):
        super().__init__()
        self.learned = learned
        if learned: self.A_raw = nn.Parameter(0.1 * torch.randn(n_nodes, n_nodes))
        else: self.register_buffer("I", torch.eye(n_nodes))
        self.lin = nn.Linear(f_in, f_out)

    def adjacency(self):
        if not self.learned: return self.I
        A = F.softplus((self.A_raw + self.A_raw.T) / 2)
        return A / A.sum(-1, keepdim=True)

    def forward(self, x):
        return self.lin(torch.einsum("nm,bkmf->bknf", self.adjacency(), x))


class GNNv3b(nn.Module):
    def __init__(self, T, learned_graph=True, freqs=None, phases=None, n_bands=3, n_nodes=9,
                 n_classes=40, n_harm=4):
        super().__init__()
        if freqs is None: freqs, phases, _ = D.load_freq_phase()   # real keyboard-order freqs + phases
        assert len(freqs) == n_classes == len(phases)
        self.nb, self.N, self.Kc, self.H, self.T = n_bands, n_nodes, n_classes, n_harm, T
        t = np.arange(T) / C.FS                                    # from window start, as in CCA-Net templates
        h = np.arange(1, n_harm + 1)
        arg = 2 * np.pi * freqs[:, None, None] * h[None, :, None] * t[None, None, :] \
            + h[None, :, None] * np.asarray(phases)[:, None, None]  # (K,H,T)
        basis = np.stack([np.cos(arg), np.sin(arg)], -1) * np.sqrt(2.0 / T)   # (K,H,T,2)
        basis = torch.tensor(basis, dtype=torch.float32).permute(2, 0, 1, 3).reshape(T, -1)
        self.register_buffer("basis", basis)                       # fixed, not trained
        f_in = n_bands * n_harm * 2
        self.g1 = FreqGCN(f_in, HID, n_nodes, learned_graph)
        self.g2 = FreqGCN(HID, HID, n_nodes, learned_graph)
        self.cls_emb = nn.Embedding(n_classes, EMB)
        self.head1 = nn.Linear(n_nodes * HID + EMB, 64)
        self.head2 = nn.Linear(64, 1)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):                                          # x (B,nb,N,T)
        B = x.shape[0]
        z = torch.matmul(x, self.basis)                            # (B,nb,N,K*H*2)
        z = z.view(B, self.nb, self.N, self.Kc, self.H, 2).permute(0, 3, 2, 1, 4, 5)
        z = z.reshape(B, self.Kc, self.N, self.nb * self.H * 2)    # (B,K,N,F)
        # contrast across the 40 candidates (per node, per feature): true class = positive outlier.
        # Replaces the old LayerNorms, which removed the match magnitude that carries the label.
        z = (z - z.mean(1, keepdim=True)) / (z.std(1, keepdim=True) + 1e-6)
        z = F.elu(self.g1(z))
        z = F.elu(self.g2(z))
        z = z.reshape(B, self.Kc, -1)                              # flatten nodes (keeps channel identity)
        e = self.cls_emb.weight.unsqueeze(0).expand(B, -1, -1)
        z = self.drop(F.elu(self.head1(torch.cat([z, e], -1))))
        return self.head2(z).squeeze(-1)                           # (B,K) logits

    @torch.no_grad()
    def export_graph(self):
        return dict(layer1=self.g1.adjacency().cpu().tolist(), layer2=self.g2.adjacency().cpu().tolist(),
                    channels=[47, 53, 54, 55, 56, 57, 60, 61, 62])


def build_model(variant, T, freqs=None, phases=None):
    if variant not in VARIANTS: raise ValueError(variant)
    return GNNv3b(T, learned_graph=(variant == "fg2"), freqs=freqs, phases=phases)


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:       # smoke uses dummy freqs so it runs without the dataset; real runs load Freq_Phase.mat
        fr = 8.0 + 0.2 * np.arange(40); ph = C.PHASE
        K.smoke(TAG, VARIANTS, build_model, freqs=fr, phases=ph); sys.exit(0)
    K.cli(TAG, VARIANTS, build_model)