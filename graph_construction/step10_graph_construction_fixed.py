"""
STEP 10: GRAPH CONSTRUCTION - electrode adjacency (k-NN on 3D scalp positions)

Input : 64-channels.loc  -> columns: index  theta(deg)  rho  label
        theta: 0 deg = nose, + = right; rho: 0 = vertex, 0.5 = head equator (90 deg from vertex)
Output: graph_construction/electrode_adjacency.npz
        positions (64,3) mm, distance_matrix (64,64), adjacency_matrix (64,64) symmetric 0/1 (no self-loops),
        labels, k, head_radius
Assumption: spherical head, radius HEAD_RADIUS (k-NN ordering is scale-invariant, so K result doesn't depend on it).
"""
import numpy as np
from pathlib import Path
from datetime import datetime
import importlib.util

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config.py'
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"config.py not found at {CONFIG_PATH}")
spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

PROJECT_ROOT = config.PROJECT_ROOT
RESULTS_DIR = config.RESULTS_DIR
OUTPUT_DIR = PROJECT_ROOT / 'graph_construction'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CHANNELS = 64
K = 5                # neighbours per node (before symmetrizing)
HEAD_RADIUS = 85.0   # mm

ELECTRODE_CANDIDATES = [
    PROJECT_ROOT / '64-channels.loc',
    PROJECT_ROOT / 'data' / 'raw' / 'benchmark' / '64-channels.loc',
    Path('64-channels.loc'),
]


def load_electrode_positions():
    """Returns positions (64,3) in mm and list of 64 labels."""
    loc_file = next((c for c in ELECTRODE_CANDIDATES if c.exists()), None)
    if loc_file is None:
        raise FileNotFoundError(f"64-channels.loc not found. Tried: {ELECTRODE_CANDIDATES}")
    print(f"[INFO] Found: {loc_file}")

    positions, labels = [], []
    with open(loc_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:          # index theta rho label
                continue
            theta = np.radians(float(parts[1]))        # 0 = nose, + = right
            phi = np.radians(float(parts[2]) * 180.0)  # polar angle from vertex
            positions.append([HEAD_RADIUS * np.sin(phi) * np.sin(theta),
                              HEAD_RADIUS * np.sin(phi) * np.cos(theta),   # +y = nose
                              HEAD_RADIUS * np.cos(phi)])
            labels.append(parts[3])

    positions = np.array(positions, dtype=np.float32)
    if len(positions) != NUM_CHANNELS:
        raise ValueError(f"Expected {NUM_CHANNELS} electrodes, parsed {len(positions)}")
    return positions, labels


def build_adjacency_matrix(positions, k=K):
    """Symmetrized k-NN graph. Returns distance_matrix, adjacency_matrix, edge_count."""
    n = len(positions)
    distance_matrix = np.linalg.norm(positions[:, None] - positions[None], axis=-1).astype(np.float32)
    d = np.round(distance_matrix, 3) + np.diag(np.full(n, np.inf))      # exclude self; round so ties are exact
    nn = np.argsort(d, axis=1, kind='stable')[:, :k]
    adjacency_matrix = np.zeros((n, n), np.float32)
    adjacency_matrix[np.arange(n)[:, None], nn] = 1
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)   # i~j if either is in the other's k-NN
    edge_count = int(np.triu(adjacency_matrix, 1).sum())
    return distance_matrix, adjacency_matrix, edge_count


def check_graph(A):
    """Sanity checks; raises AssertionError on failure."""
    from scipy.sparse.csgraph import connected_components
    assert A.shape == (NUM_CHANNELS, NUM_CHANNELS)
    assert (A == A.T).all(), "adjacency not symmetric"
    assert np.diag(A).sum() == 0, "self-loops present"
    assert A.sum(1).min() >= 1, "isolated node"
    ncomp = connected_components(A)[0]
    assert ncomp == 1, f"graph has {ncomp} components"


def build_graph():
    log = ["=" * 80, "STEP 10: GRAPH CONSTRUCTION (k-NN, 3D spherical positions)", "=" * 80,
           f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}", f"K = {K}, head radius = {HEAD_RADIUS} mm", ""]
    try:
        positions, labels = load_electrode_positions()
        dist, A, edges = build_adjacency_matrix(positions, K)
        check_graph(A)
        deg = A.sum(1)
        log += [f"Loaded {len(positions)} electrodes",
                f"Distance range: {dist.min():.1f}-{dist.max():.1f} mm",
                f"Edges: {edges}  density: {edges / (NUM_CHANNELS * (NUM_CHANNELS - 1) / 2):.2%}",
                f"Degree min/mean/max: {int(deg.min())}/{deg.mean():.1f}/{int(deg.max())}",
                "Checks passed: symmetric, no self-loops, no isolated nodes, 1 connected component", ""]
        for name in ('O1', 'Oz', 'O2', 'POz', 'M1', 'Cz'):
            if name in labels:
                i = labels.index(name)
                log.append(f"  {name}: {[labels[j] for j in np.where(A[i])[0]]}")
        log.append("")

        out = OUTPUT_DIR / 'electrode_adjacency.npz'
        np.savez_compressed(out, positions=positions, distance_matrix=dist, adjacency_matrix=A,
                            labels=np.array(labels), k=K, head_radius=HEAD_RADIUS)
        log.append(f"Saved: {out}")

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(15, 7))
            # top view: nose up; drawn for all electrodes (x right, y nose)
            for i, j in zip(*np.where(np.triu(A, 1))):
                ax[0].plot(positions[[i, j], 0], positions[[i, j], 1], 'k-', lw=0.6, alpha=0.5)
            sc = ax[0].scatter(positions[:, 0], positions[:, 1], c=deg, cmap='viridis', s=180, zorder=3)
            for i, l in enumerate(labels):
                ax[0].text(positions[i, 0], positions[i, 1], l, ha='center', va='center', fontsize=6, zorder=4)
            ax[0].set_aspect('equal'); ax[0].set_title(f'k-NN graph (K={K}), top view, nose up\ncolour = degree')
            ax[0].set_xlabel('X (mm, right +)'); ax[0].set_ylabel('Y (mm, nose +)')
            plt.colorbar(sc, ax=ax[0], label='degree')
            ax[1].imshow(A, cmap='binary'); ax[1].set_title(f'Adjacency ({edges} edges)')
            ax[1].set_xlabel('Channel'); ax[1].set_ylabel('Channel')
            plt.tight_layout()
            viz = RESULTS_DIR / 'step10_electrode_graph.png'
            plt.savefig(viz, dpi=120, bbox_inches='tight'); plt.close()
            log.append(f"Saved plot: {viz}")
        except ImportError:
            log.append("[INFO] matplotlib not available, skipping plot")

        log += ["", "=" * 80, "STEP 10 COMPLETE", "=" * 80,
                "Next: Step 11 (GNN). Node features: 280-bin spectrum from Step 9d, shape (64, 280)."]
    except Exception as e:
        import traceback
        log += [f"[ERROR] {e}", traceback.format_exc()]

    (RESULTS_DIR / "step10_graph_construction_log.txt").write_text('\n'.join(log), encoding='utf-8')
    print('\n'.join(log))


if __name__ == "__main__":
    build_graph()