"""
zs_gnn_v3a_band.py -- per-band + layer-wise adjacency GNN (no CCA).

Same front end as v2c (depthwise temporal conv per (band,node), time kept, GroupNorm, ELU, FC on
hidden*T), but the graph part is different:
  layer 1: one learned 9x9 adjacency PER FREQUENCY BAND, and per-band feature weights
  fuse   : 1x1 conv mixes the 3 bands into `hidden` features
  layer 2: its own learned 9x9 adjacency (layer-wise graph, as in DDGCNN-style models)
Adjacency construction is the same as v2c: A=softplus((A_raw+A_raw^T)/2), rows normalised.

Variants:
  pb : per-band adjacency in layer 1 (3 matrices) + layer-2 matrix      <- the proposed model
  sh : same model but ONE adjacency shared by all bands in layer 1      <- ablation (isolates per-band)
(no-graph ablation = zs_gnn_v2c.py --variant g3, same recipe)

Usage:  python3 zs_gnn_v3a_band.py --smoke
        python3 zs_gnn_v3a_band.py --sl 1.0 --subjects 1 --variant pb sh
        python3 zs_gnn_v3a_band.py --sl 0.5 1.0 --variant pb sh --resume
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import zs_gnn_v3_common as K

TAG = "gnnv3a"
VARIANTS = ("pb", "sh")
D_T, KERNEL, HID_BAND, HIDDEN, DROPOUT = 16, 16, 32, 64, 0.3   # D_T/KERNEL/HIDDEN/DROPOUT = v2c


class BandGCNLayer(nn.Module):
    """x (B, nb, Fin, N, T) -> (B, nb, Fout, N, T). Adjacency per band (per_band) or shared."""
    def __init__(self, n_bands, f_in, f_out, n_nodes, per_band=True):
        super().__init__()
        self.nb, self.per_band = n_bands, per_band
        self.A_raw = nn.Parameter(0.1 * torch.randn(n_bands if per_band else 1, n_nodes, n_nodes))
        self.W = nn.Parameter(torch.empty(n_bands, f_in, f_out))
        for b in range(n_bands): nn.init.xavier_uniform_(self.W.data[b])
        self.bias = nn.Parameter(torch.zeros(n_bands, f_out, 1, 1))

    def adjacency(self):
        A = F.softplus((self.A_raw + self.A_raw.transpose(-1, -2)) / 2)
        A = A / A.sum(-1, keepdim=True)
        return A.expand(self.nb, -1, -1) if not self.per_band else A    # (nb,N,N)

    def forward(self, x):
        x = torch.einsum("cnm,bcfmt->bcfnt", self.adjacency(), x)       # propagate over nodes
        x = torch.einsum("cfo,bcfnt->bcont", self.W, x)                  # per-band feature map
        return x + self.bias


class GNNv3a(nn.Module):
    def __init__(self, T, per_band=True, n_bands=3, n_nodes=9, n_classes=40):
        super().__init__()
        self.nb, self.N, self.T = n_bands, n_nodes, T
        g = n_bands * n_nodes
        self.conv = nn.Conv1d(g, g * D_T, KERNEL, padding=KERNEL // 2, groups=g)
        self.bn0 = nn.GroupNorm(8, g * D_T)
        self.gcn1 = BandGCNLayer(n_bands, D_T, HID_BAND, n_nodes, per_band)
        self.bn1 = nn.GroupNorm(n_bands, n_bands * HID_BAND)
        self.fuse = nn.Conv2d(n_bands * HID_BAND, HIDDEN, 1)
        self.bn2 = nn.GroupNorm(8, HIDDEN)
        self.gcn2 = BandGCNLayer(1, HIDDEN, HIDDEN, n_nodes, per_band=False)   # layer-wise graph
        self.bn3 = nn.GroupNorm(8, HIDDEN)
        self.drop = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN * T, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu"); nn.init.zeros_(m.bias)

    def forward(self, x):                                   # x (B, nb, N, T)
        B, nb, N, T = x.shape
        h = self.conv(x.reshape(B, nb * N, T))[..., :T]
        h = F.elu(self.bn0(h))
        h = h.view(B, nb, N, D_T, T).permute(0, 1, 3, 2, 4)                 # (B,nb,d_t,N,T)
        h = self.gcn1(h)                                                     # (B,nb,hb,N,T)
        h = F.elu(self.bn1(h.reshape(B, nb * HID_BAND, N, T)))
        h = F.elu(self.bn2(self.fuse(h)))                                    # (B,hidden,N,T)
        h = self.gcn2(h.unsqueeze(1)).squeeze(1)
        h = F.elu(self.bn3(h))
        h = h.mean(2)                                                        # pool nodes -> (B,hidden,T)
        return self.fc(self.drop(h.reshape(B, -1)))

    @torch.no_grad()
    def export_graph(self):
        return dict(layer1=self.gcn1.adjacency().cpu().tolist(), layer2=self.gcn2.adjacency()[0].cpu().tolist(),
                    channels=[47, 53, 54, 55, 56, 57, 60, 61, 62])


def build_model(variant, T):
    if variant not in VARIANTS: raise ValueError(variant)
    return GNNv3a(T, per_band=(variant == "pb"))


if __name__ == "__main__":
    K.cli(TAG, VARIANTS, build_model)
