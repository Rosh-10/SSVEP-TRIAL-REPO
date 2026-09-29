"""
STEP 10: GRAPH CONSTRUCTION - FINAL VERSION (Tab-delimited parsing)

Format: index<TAB>theta<TAB>rho<TAB>label
Example: 1	-18	0.51111	FP1
"""

import numpy as np
from pathlib import Path
from datetime import datetime
import importlib.util

# ============================================================================
# LOAD CONFIG
# ============================================================================

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

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_CHANNELS = 64
DISTANCE_THRESHOLD = 30.0  # mm
HEAD_RADIUS = 85.0  # mm

ELECTRODE_CANDIDATES = [
    PROJECT_ROOT / '64-channels.loc',
    PROJECT_ROOT / 'data' / 'raw' / 'benchmark' / '64-channels.loc',
    Path('64-channels.loc'),
]

# ============================================================================
# LOAD ELECTRODE POSITIONS
# ============================================================================

def load_electrode_positions():
    """
    Load 64-channel electrode positions from tab-delimited .loc file.
    
    Format: index<TAB>theta<TAB>rho<TAB>label
    Converts spherical (theta, rho) to Cartesian (X, Y, Z)
    """
    loc_file = None
    for candidate in ELECTRODE_CANDIDATES:
        if candidate.exists():
            loc_file = candidate
            print(f"[INFO] Found: {loc_file}")
            break
    
    if loc_file is None:
        raise FileNotFoundError(f"64-channels.loc not found")
    
    positions = []
    labels = []
    
    with open(loc_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Split by tabs
            parts = line.split('\t')
            parts = [p.strip() for p in parts if p.strip()]  # Remove extra whitespace
            
            if len(parts) < 4:
                continue
            
            try:
                idx = int(parts[0])
                theta_deg = float(parts[1])  # Azimuth angle
                rho = float(parts[2])  # Normalized radius (0-1)
                label = parts[3]
                
                # Convert spherical to Cartesian
                theta_rad = np.radians(theta_deg)
                radius_mm = rho * HEAD_RADIUS
                x = radius_mm * np.cos(theta_rad)
                y = radius_mm * np.sin(theta_rad)
                z = 0.0
                
                labels.append(label)
                positions.append([x, y, z])
                
            except (ValueError, IndexError):
                continue
    
    positions = np.array(positions, dtype=np.float32)
    
    print(f"[INFO] Loaded {len(positions)} electrode positions")
    if len(positions) > 0:
        print(f"[INFO] X range: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] mm")
        print(f"[INFO] Y range: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}] mm")
    
    return positions, labels

# ============================================================================
# BUILD ADJACENCY MATRIX
# ============================================================================

def build_adjacency_matrix(positions, threshold=30.0):
    """Build adjacency matrix based on Euclidean distance."""
    n_channels = len(positions)
    
    # Compute pairwise distances
    distance_matrix = np.zeros((n_channels, n_channels), dtype=np.float32)
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            dist = np.linalg.norm(positions[i] - positions[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    # Binary adjacency
    adjacency_matrix = (distance_matrix <= threshold).astype(np.float32)
    np.fill_diagonal(adjacency_matrix, 0)
    
    # Count edges
    edge_count = int(np.sum(np.triu(adjacency_matrix, k=1)))
    
    return distance_matrix, adjacency_matrix, edge_count

# ============================================================================
# MAIN
# ============================================================================

def build_graph():
    """Build and save electrode graph."""
    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("STEP 10: GRAPH CONSTRUCTION")
    log_lines.append("=" * 80)
    log_lines.append(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Output directory: {OUTPUT_DIR}")
    log_lines.append(f"Distance threshold: {DISTANCE_THRESHOLD} mm")
    log_lines.append("")
    
    try:
        # Load electrode positions
        positions, labels = load_electrode_positions()
        log_lines.append(f"Loaded {len(positions)} electrodes")
        log_lines.append("")
        
        if len(positions) != NUM_CHANNELS:
            raise ValueError(f"Expected {NUM_CHANNELS} electrodes, got {len(positions)}")
        
        # Build adjacency matrix
        log_lines.append("Building adjacency matrix...")
        distance_matrix, adjacency_matrix, edge_count = build_adjacency_matrix(
            positions, threshold=DISTANCE_THRESHOLD
        )
        
        log_lines.append(f"Distance range: {distance_matrix.min():.2f} - {distance_matrix.max():.2f} mm")
        log_lines.append(f"Edges (distance <= {DISTANCE_THRESHOLD} mm): {edge_count}")
        log_lines.append(f"Graph density: {edge_count / (NUM_CHANNELS * (NUM_CHANNELS - 1) / 2):.1%}")
        log_lines.append("")
        
        # Statistics
        degrees = np.sum(adjacency_matrix, axis=1)
        log_lines.append("Degree distribution:")
        log_lines.append(f"  Min: {int(degrees.min())} neighbors")
        log_lines.append(f"  Max: {int(degrees.max())} neighbors")
        log_lines.append(f"  Mean: {degrees.mean():.1f} neighbors")
        log_lines.append("")
        
        # Save graph
        output_file = OUTPUT_DIR / 'electrode_adjacency.npz'
        np.savez_compressed(
            output_file,
            positions=positions,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            labels=np.array(labels),
            threshold=DISTANCE_THRESHOLD
        )
        
        log_lines.append(f"Saved: {output_file}")
        log_lines.append("")
        
        # Visualization
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Electrode positions
            ax = axes[0]
            scatter = ax.scatter(positions[:, 0], positions[:, 1], c=degrees, cmap='viridis', s=100, alpha=0.7)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_title(f'Electrode Positions\nColored by degree')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            plt.colorbar(scatter, ax=ax, label='Neighbors')
            
            # Adjacency heatmap
            ax = axes[1]
            im = ax.imshow(adjacency_matrix, cmap='binary', aspect='auto')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Channel')
            ax.set_title(f'Adjacency Matrix\n({edge_count} edges)')
            plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            viz_file = RESULTS_DIR / 'step10_electrode_positions.png'
            plt.savefig(viz_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            log_lines.append(f"Saved visualization: {viz_file}")
            log_lines.append("")
        except ImportError:
            log_lines.append("[INFO] Matplotlib not available, skipping visualization")
            log_lines.append("")
        
        # Summary
        log_lines.append("=" * 80)
        log_lines.append("STEP 10: COMPLETE")
        log_lines.append("=" * 80)
        log_lines.append("")
        log_lines.append("Summary:")
        log_lines.append(f"  - Loaded {NUM_CHANNELS} electrode positions")
        log_lines.append(f"  - Built adjacency matrix (threshold {DISTANCE_THRESHOLD} mm)")
        log_lines.append(f"  - {edge_count} edges, density {edge_count / (NUM_CHANNELS * (NUM_CHANNELS - 1) / 2):.1%}")
        log_lines.append(f"  - Average degree: {degrees.mean():.1f} neighbors/node")
        log_lines.append("")
        log_lines.append("Graph saved to: graph_construction/electrode_adjacency.npz")
        log_lines.append("")
        log_lines.append("READY FOR STEP 11 (GNN Model Training)")
        
    except Exception as e:
        import traceback
        log_lines.append(f"[ERROR] {str(e)}")
        log_lines.append("")
        log_lines.append(traceback.format_exc())
    
    # Write log
    log_file = RESULTS_DIR / "step10_graph_construction_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    
    # Print to console
    print('\n'.join(log_lines))


if __name__ == "__main__":
    build_graph()