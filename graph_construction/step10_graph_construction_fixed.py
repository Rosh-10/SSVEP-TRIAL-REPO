"""
STEP 10: GRAPH CONSTRUCTION - BUILD ELECTRODE ADJACENCY MATRIX (FIXED)

Objective:
    Load 64-channel electrode positions and build distance-based graphs.
    Each graph is a static adjacency matrix for spatial neighborhood of EEG channels.

Input:
    Electrode positions: 64-channels.loc (spherical coordinates on 10-20 system)
    Format: index, index_repeat, theta (degrees), rho (0-1 normalized), label

Output:
    Graph files: graph_construction/electrode_adjacency.npz
    - Contains distance matrix and binary adjacency (threshold 30mm)

Coordinate System:
    Input: Spherical (theta, rho) on standard 10-20 electrode layout
    Output: Cartesian (X, Y, Z) in 3D space (mm)
    Head radius: 85 mm (standard reference for scalp surface)
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
DISTANCE_THRESHOLD = 30.0  # mm (functional neighborhood)
HEAD_RADIUS = 85.0  # mm (standard reference for scalp surface)

# Try multiple possible locations for electrode file
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
    Load 64-channel electrode positions from .loc file.
    
    Format (spherical coordinates on 10-20 layout):
    index, index_repeat, theta (degrees), rho (0-1), label
    
    Converts to Cartesian 3D: (X, Y, Z) in mm on head surface
    
    Returns:
        positions: Shape (64, 3) — X, Y, Z coordinates in mm
        labels: List of 64 channel names
    """
    print(f"[INFO] Loading electrode positions from .loc file...")
    
    loc_file = None
    for candidate in ELECTRODE_CANDIDATES:
        if candidate.exists():
            loc_file = candidate
            print(f"[INFO] Found: {loc_file}")
            break
    
    if loc_file is None:
        print(f"[WARNING] Tried: {ELECTRODE_CANDIDATES}")
        raise FileNotFoundError(f"64-channels.loc not found in any expected location")
    
    positions = []
    labels = []
    
    with open(loc_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                # Format: index, index_repeat, theta, rho, label
                idx = int(parts[0])
                theta_deg = float(parts[2])  # Azimuth angle in degrees
                rho = float(parts[3])  # Normalized radius (0-1, where 1 = scalp surface)
                label = parts[4]
                
                # Convert spherical to Cartesian
                # Theta: 0° = right (positive X), 90° = back (negative Y), 180° = left, 270° = front
                # Rho: 0 = center, 1 = scalp surface
                theta_rad = np.radians(theta_deg)
                
                # 3D Cartesian (assuming on sphere surface)
                # Using standard spherical coords: theta (azimuth), phi would be elevation
                # For 10-20 system, we'll use 2D projection on scalp plane with Z=0 at center
                radius_mm = rho * HEAD_RADIUS
                x = radius_mm * np.cos(theta_rad)
                y = radius_mm * np.sin(theta_rad)
                z = 0  # All on same plane (scalp surface approximation)
                
                labels.append(label)
                positions.append([x, y, z])
                
            except (ValueError, IndexError) as e:
                continue
    
    positions = np.array(positions, dtype=np.float32)
    
    if len(positions) != NUM_CHANNELS:
        print(f"[WARNING] Expected {NUM_CHANNELS} channels, found {len(positions)}")
    
    print(f"[INFO] Loaded {len(positions)} electrode positions")
    print(f"[INFO] Position range: X [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] mm")
    print(f"[INFO] Position range: Y [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}] mm")
    print(f"[INFO] Position range: Z [{positions[:, 2].min():.1f}, {positions[:, 2].max():.1f}] mm")
    
    return positions, labels

# ============================================================================
# BUILD ADJACENCY MATRIX
# ============================================================================

def build_adjacency_matrix(positions, threshold=30.0):
    """
    Build adjacency matrix based on Euclidean distance.
    
    Args:
        positions: Shape (64, 3) — electrode positions
        threshold: Distance threshold (mm) for connectivity
    
    Returns:
        distance_matrix: Shape (64, 64) — pairwise distances
        adjacency_matrix: Shape (64, 64) — binary (0/1), 1 if distance <= threshold
        edge_count: Number of edges in graph
    """
    n_channels = len(positions)
    
    # Compute pairwise distances
    distance_matrix = np.zeros((n_channels, n_channels), dtype=np.float32)
    
    for i in range(n_channels):
        for j in range(i, n_channels):
            dist = np.linalg.norm(positions[i] - positions[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    # Binary adjacency: 1 if distance <= threshold
    adjacency_matrix = (distance_matrix <= threshold).astype(np.float32)
    
    # Diagonal should be 0 (no self-loops)
    np.fill_diagonal(adjacency_matrix, 0)
    
    # Count edges (upper triangle only to avoid double-counting)
    edge_count = int(np.sum(np.triu(adjacency_matrix, k=1)))
    
    return distance_matrix, adjacency_matrix, edge_count

# ============================================================================
# MAIN
# ============================================================================

def build_graph():
    """Build and save electrode graph."""
    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("STEP 10: GRAPH CONSTRUCTION (Fixed - Spherical Coordinates)")
    log_lines.append("=" * 80)
    log_lines.append(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Output directory: {OUTPUT_DIR}")
    log_lines.append(f"Distance threshold: {DISTANCE_THRESHOLD} mm")
    log_lines.append(f"Head radius (scalp surface): {HEAD_RADIUS} mm")
    log_lines.append("")
    
    try:
        # Load electrode positions
        positions, labels = load_electrode_positions()
        log_lines.append(f"Loaded {len(positions)} electrodes")
        log_lines.append("")
        
        # Build adjacency matrix
        log_lines.append("Building adjacency matrix...")
        distance_matrix, adjacency_matrix, edge_count = build_adjacency_matrix(
            positions, 
            threshold=DISTANCE_THRESHOLD
        )
        
        log_lines.append(f"Distance matrix shape: {distance_matrix.shape}")
        log_lines.append(f"Adjacency matrix shape: {adjacency_matrix.shape}")
        log_lines.append(f"Distance range: {distance_matrix.min():.2f} - {distance_matrix.max():.2f} mm")
        log_lines.append(f"Edges (distance <= {DISTANCE_THRESHOLD} mm): {edge_count}")
        log_lines.append(f"Graph density: {edge_count / (NUM_CHANNELS * (NUM_CHANNELS - 1) / 2):.2%}")
        log_lines.append("")
        
        # Statistics
        log_lines.append("Degree distribution (neighbors per node):")
        degrees = np.sum(adjacency_matrix, axis=1)
        log_lines.append(f"  Min: {int(degrees.min())} neighbors")
        log_lines.append(f"  Max: {int(degrees.max())} neighbors")
        log_lines.append(f"  Mean: {degrees.mean():.1f} neighbors")
        log_lines.append(f"  Std: {degrees.std():.1f}")
        log_lines.append("")
        
        # Save graph
        output_file = OUTPUT_DIR / 'electrode_adjacency.npz'
        np.savez_compressed(
            output_file,
            positions=positions,
            distance_matrix=distance_matrix,
            adjacency_matrix=adjacency_matrix,
            labels=labels,
            threshold=DISTANCE_THRESHOLD
        )
        
        log_lines.append(f"Saved: {output_file}")
        log_lines.append("")
        
        # Generate visualization (if matplotlib available)
        try:
            import matplotlib.pyplot as plt
            
            # Plot 1: Electrode positions
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Top view: X-Y plane
            ax = axes[0]
            scatter = ax.scatter(positions[:, 0], positions[:, 1], c=degrees, cmap='viridis', s=100, alpha=0.7)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_title(f'Electrode Positions (Top View)\nColored by degree (neighbors)')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Number of Neighbors')
            
            # Plot 2: Adjacency matrix heatmap
            ax = axes[1]
            im = ax.imshow(adjacency_matrix, cmap='binary', aspect='auto')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Channel')
            ax.set_title(f'Adjacency Matrix\n({edge_count} edges, threshold={DISTANCE_THRESHOLD}mm)')
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
        log_lines.append("STEP 10: GRAPH CONSTRUCTION COMPLETE")
        log_lines.append("=" * 80)
        log_lines.append("")
        log_lines.append("Summary:")
        log_lines.append(f"  - Loaded {NUM_CHANNELS} electrode positions (10-20 system)")
        log_lines.append(f"  - Converted from spherical to Cartesian coordinates")
        log_lines.append(f"  - Built adjacency matrix (threshold {DISTANCE_THRESHOLD} mm)")
        log_lines.append(f"  - Graph has {edge_count} edges, density {edge_count / (NUM_CHANNELS * (NUM_CHANNELS - 1) / 2):.1%}")
        log_lines.append(f"  - Average degree: {degrees.mean():.1f} neighbors/node")
        log_lines.append("")
        log_lines.append("Graph saved to: graph_construction/electrode_adjacency.npz")
        log_lines.append("")
        log_lines.append("Next step: Step 11 (GNN Model Training)")
        log_lines.append("  - Use spectrum features from Step 9d")
        log_lines.append("  - Use electrode graph from this step")
        log_lines.append("  - Train GraphConv and DDGCNN models")
        
    except Exception as e:
        import traceback
        log_lines.append(f"[ERROR] {str(e)}")
        log_lines.append(traceback.format_exc())
        log_lines.append("")
    
    # Write log
    log_file = RESULTS_DIR / "step10_graph_construction_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    
    # Print to console
    print('\n'.join(log_lines))


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    build_graph()
