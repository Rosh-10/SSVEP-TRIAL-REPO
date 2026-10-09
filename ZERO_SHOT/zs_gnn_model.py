"""
zs_gnn_model.py  —  9-node dense-adjacency GCN for zero-shot SSVEP (BETA dataset).

Tensor dimensions (all comments assume batch_size = B, n_nodes = 9):

  Input from zs_data.prep():
    raw_bank_A  :  (B, 3, 9, T)   fb_stand'd filterbank-A features
                   bands=3, nodes=9, time=T

  Stage 1 — temporal encoder (per node, per band):
    conv1d over time axis: (B, 3, 9, T) -> (B, 3, 9, d_t)
    then flatten bands: (B, 3*d_t, 9) — treat as node features
    node_dim = 3 * d_t

  Stage 2 — dense-adjacency GCN:
    adjacency A: (9, 9) learned, symmetric, row-normalised
    two GCN layers: node_dim -> hidden_dim -> hidden_dim
    activation: ELU

  Stage 3 — readout + classifier:
    global mean pool over nodes: (B, hidden_dim)
    linear -> 40 logits

Design choices:
  - Dense (fully-connected) adjacency: all 9 nodes connected; weights learned.
  - Adjacency is shared across samples (not input-dependent); symmetric + softplus
    positive to avoid sign ambiguity; row-normalised (D^{-1} A style).
  - No self-loops added explicitly — diagonal is also learned.
  - Temporal encoder uses depthwise (groups=3*9) 1-D conv to keep parameter
    count low and avoid cross-channel mixing before the graph step.
  - Everything in float32 for GPU compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sym_row_norm(W: torch.Tensor) -> torch.Tensor:
    """Make W symmetric, apply softplus (positive), row-normalise.

    W: (9, 9) raw parameter matrix
    Returns: (9, 9) normalised adjacency
    """
    # symmetrise
    A = (W + W.T) / 2.0
    # positive weights via softplus  (no hard zeros, smooth gradient)
    A = F.softplus(A)
    # row-normalise: D^{-1} A
    D_inv = 1.0 / A.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return A * D_inv


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """Encode (B, n_bands, n_nodes, T) -> (B, n_bands*d_t, n_nodes).

    Uses a 1-D conv along the time axis applied identically to every
    (band, node) pair — implemented as a grouped conv where each group
    is one (band, node) channel.

    Parameters
    ----------
    n_bands : int   (3 for bank-A)
    n_nodes : int   (9 EEG channels)
    T       : int   window length in samples (125 for sl=0.5, 250 for sl=1.0)
    d_t     : int   output time-features per (band, node)  [MY CHOICE: 16]
    kernel  : int   conv kernel width  [MY CHOICE: 16]
    """

    def __init__(self, n_bands: int, n_nodes: int, T: int,
                 d_t: int = 16, kernel: int = 16):
        super().__init__()
        self.n_bands = n_bands
        self.n_nodes = n_nodes
        self.d_t = d_t
        groups = n_bands * n_nodes           # depthwise over (band, node)
        # conv: in_channels = groups, out_channels = groups * d_t, groups = groups
        # -> each (band,node) channel gets its own d_t filters
        self.conv = nn.Conv1d(
            in_channels=groups,
            out_channels=groups * d_t,
            kernel_size=kernel,
            padding=kernel // 2,
            groups=groups,
            bias=True,
        )
        self.bn   = nn.BatchNorm1d(groups * d_t)
        self.act  = nn.ELU()
        # global average pool over remaining time -> scalar per filter
        # (we do this manually in forward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, n_bands, n_nodes, T)
        returns : (B, n_bands * d_t, n_nodes)
        """
        B = x.size(0)
        n_bands, n_nodes, T = self.n_bands, self.n_nodes, x.size(-1)

        # -> (B, n_bands*n_nodes, T)  for Conv1d
        x2 = x.view(B, n_bands * n_nodes, T)

        # -> (B, n_bands*n_nodes*d_t, T')
        h = self.conv(x2)
        # trim to original T if padding caused overshoot
        h = h[..., :T]
        # batchnorm + activation
        h = self.act(self.bn(h))

        # global average pool over time  ->  (B, n_bands*n_nodes*d_t)
        h = h.mean(dim=-1)

        # reshape to (B, n_bands*d_t, n_nodes)
        # current layout: (B, [n_bands*n_nodes] * d_t)
        # after depthwise conv, channels are ordered as
        #   (band0_node0)*d_t, (band0_node1)*d_t, ..., (band2_node8)*d_t
        # We want (B, n_bands*d_t, n_nodes).
        # Current: (B, n_bands * n_nodes * d_t)
        # view as (B, n_bands, n_nodes, d_t) then permute -> (B, n_bands, d_t, n_nodes)
        # then (B, n_bands*d_t, n_nodes)
        h = h.view(B, n_bands, n_nodes, self.d_t)   # (B, 3, 9, d_t)
        h = h.permute(0, 1, 3, 2)                   # (B, 3, d_t, 9)
        h = h.reshape(B, n_bands * self.d_t, n_nodes)  # (B, 3*d_t, 9)
        return h


class DenseGCNLayer(nn.Module):
    """One GCN layer with a learned dense 9×9 adjacency.

    H_out = σ( A_norm  @  H_in  @  W + b )

    where  @  on (B, d_in, 9) is applied as a graph-convolution:
      aggregated[b, :, i] = sum_j A_norm[i,j] * H[b, :, j]
      -> H_agg = H @ A_norm.T    (broadcast over batch & feature dims)
    then a linear transform over features.

    Parameters
    ----------
    in_dim  : int
    out_dim : int
    n_nodes : int  (9)
    """

    def __init__(self, in_dim: int, out_dim: int, n_nodes: int = 9):
        super().__init__()
        self.n_nodes = n_nodes
        # raw adjacency parameter (shared across samples)
        self.A_raw  = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1)
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.bn     = nn.BatchNorm1d(n_nodes)  # over node dim
        self.act    = nn.ELU()

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H : (B, d_in, n_nodes)
        returns : (B, d_out, n_nodes)
        """
        A = _sym_row_norm(self.A_raw)    # (9, 9)

        # graph aggregation: H_agg[b, f, i] = sum_j A[i,j] H[b, f, j]
        #   H: (B, d_in, 9)  ->  H_agg: (B, d_in, 9)
        #   einsum "bfi, ji -> bfj" ... simpler: H @ A^T
        H_agg = torch.einsum('bfn,mn->bfm', H, A)   # (B, d_in, 9)

        # linear over feature dim: (B, 9, d_in) -> (B, 9, d_out)
        H_agg = H_agg.permute(0, 2, 1)              # (B, 9, d_in)
        H_out = self.linear(H_agg)                   # (B, 9, d_out)

        # batchnorm over node dim
        H_out = self.bn(H_out)                       # BN expects (B, n_nodes, d_out)
                                                     # — BN1d on last dim: OK if we treat
                                                     # node as the "channel"
        H_out = self.act(H_out)                      # (B, 9, d_out)
        H_out = H_out.permute(0, 2, 1)               # (B, d_out, 9) for next layer
        return H_out


# ---------------------------------------------------------------------------
# full model
# ---------------------------------------------------------------------------

class SSVEPGraphNet(nn.Module):
    """9-node dense-adjacency GCN for zero-shot SSVEP classification.

    Input  : (B, n_bands, n_nodes, T)   — bank-A fb_stand'd features
    Output : (B, n_classes)             — unnormalised logits

    Architecture
    ------------
    TemporalEncoder    (B, 3, 9, T)   ->  (B, 3*d_t, 9)
    GCNLayer-1         (B, 3*d_t, 9)  ->  (B, hidden, 9)
    GCNLayer-2         (B, hidden, 9) ->  (B, hidden, 9)
    GlobalMeanPool                        (B, hidden)
    Dropout
    Linear                                (B, n_classes)

    Parameters
    ----------
    T          : int   window samples (125 or 250)
    n_bands    : int   3  (bank A)
    n_nodes    : int   9  (EEG channels)
    n_classes  : int   40
    d_t        : int   16  temporal features per (band, node)
    hidden     : int   64  GCN hidden dim
    kernel     : int   16  temporal conv kernel
    dropout    : float 0.3
    """

    def __init__(
        self,
        T: int,
        n_bands: int = 3,
        n_nodes: int = 9,
        n_classes: int = 40,
        d_t: int = 16,
        hidden: int = 64,
        kernel: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.T = T

        node_feat_dim = n_bands * d_t   # 3*16 = 48

        self.enc   = TemporalEncoder(n_bands, n_nodes, T, d_t=d_t, kernel=kernel)
        self.gcn1  = DenseGCNLayer(node_feat_dim, hidden, n_nodes)
        self.gcn2  = DenseGCNLayer(hidden, hidden, n_nodes)
        self.drop  = nn.Dropout(p=dropout)
        self.head  = nn.Linear(hidden, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, n_bands, n_nodes, T)   float32
        returns logits : (B, n_classes)
        """
        # --- temporal encoder ---
        h = self.enc(x)          # (B, 3*d_t, 9)

        # --- graph convolutions ---
        h = self.gcn1(h)         # (B, hidden, 9)
        h = self.gcn2(h)         # (B, hidden, 9)

        # --- global mean pool over nodes ---
        h = h.mean(dim=-1)       # (B, hidden)

        h = self.drop(h)
        logits = self.head(h)    # (B, 40)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices (B,) without gradient."""
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)


# ---------------------------------------------------------------------------
# quick shape smoke-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    def check(name, tensor, expected_shape):
        ok = tuple(tensor.shape) == tuple(expected_shape)
        tag = "OK" if ok else f"FAIL (got {tuple(tensor.shape)})"
        print(f"  {name:<40} {str(tuple(tensor.shape)):<25} {tag}")
        assert ok, f"Shape mismatch for {name}"

    print("\n=== SSVEPGraphNet shape smoke-test ===\n")

    for sl, T in [(0.5, 125), (1.0, 250)]:
        print(f"--- sl={sl}  T={T} ---")
        B = 4   # batch of 4 trials
        x = torch.randn(B, 3, 9, T)

        model = SSVEPGraphNet(T=T)
        model.eval()

        # trace through sub-modules
        with torch.no_grad():
            h_enc = model.enc(x)
            check(f"TemporalEncoder (B,3*16,9)", h_enc, (B, 48, 9))

            h1 = model.gcn1(h_enc)
            check(f"GCNLayer-1 (B,64,9)", h1, (B, 64, 9))

            h2 = model.gcn2(h1)
            check(f"GCNLayer-2 (B,64,9)", h2, (B, 64, 9))

            pool = h2.mean(dim=-1)
            check(f"GlobalMeanPool (B,64)", pool, (B, 64))

            logits = model(x)
            check(f"logits (B,40)", logits, (B, 40))

            preds = model.predict(x)
            check(f"predictions (B,)", preds, (B,))
            assert preds.min() >= 0 and preds.max() < 40, "prediction out of range"

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_params:,}")
        print()

    print("All shape checks passed.\n")

    # gradient check on a single batch
    print("=== Gradient check (sl=0.5) ===")
    model = SSVEPGraphNet(T=125)
    model.train()
    x = torch.randn(8, 3, 9, 125)
    y = torch.randint(0, 40, (8,))
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    bad = [(n, p) for n, p in model.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all()]
    if bad:
        print("  FAIL — non-finite or missing grads:", [n for n, _ in bad])
        sys.exit(1)
    print(f"  loss={loss.item():.4f}  all grads finite  OK\n")
