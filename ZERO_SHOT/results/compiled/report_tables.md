# Compiled results – 35-fold LOSO, mean accuracy % ± SE  (zero-shot, Benchmark dataset)
> Nominal p-values; ~85 paired tests were run on the same 35 subjects.  
> 9-channel rows are protocol-comparable with CCA-Net.  
> 19/64-channel rows belong in a SEPARATE table with matching-channel baselines.  

### Table 1 – Classical baselines and CCA-Net replication (9 ch)
| method | description | channels | params | 0.5 s acc ± SE | 1.0 s acc ± SE | note |
|---|---|---|---|---|---|---|
| cca_net_off | CCA-Net replication: CNN + Modified FBtt-CCA + FBCCA(SST) | 9 | nan | 51.95 ± 3.54 | 82.48 ± 3.03 |  |
| fbttcca_yf | FBtt-CCA (classical) | 9 | nan | 39.15 ± 3.21 | 77.71 ± 3.28 |  |
| cnn_ref | CCA-Net CNN branch alone (CSDuDoFN-like) | 9 | nan | 41.54 ± 3.66 | 68.32 ± 3.96 |  |
| fbcca | FBCCA (classical) | 9 | nan | 20.64 ± 1.80 | 63.49 ± 3.53 |  |

### Table 2 – GNN v3 family, 9 channels (protocol-comparable)
| method | description | channels | params | 0.5 s acc ± SE | 1.0 s acc ± SE | note |
|---|---|---|---|---|---|---|
| gnnv3l_b5d9 | v3l b5d9: dense + 5-band bank C, 9 ch | 9 | nan | 40.99 ± 3.39 | 76.92 ± 3.08 |  |
| gnnv3j_aw9 | v3j: trial-adaptive node weights, 9 ch | 9 | nan | 40.76 ± 3.31 | 76.18 ± 3.19 |  |
| gnnv3e_lr | v3e lr: learned signed 9x9 spatial map (best 9-ch) | 9 | nan | 39.83 ± 3.35 | 76.10 ± 3.18 |  |
| gnnv3g_knn9 | v3g knn9: kNN signed map, 9 ch | 9 | nan | 39.08 ± 3.23 | 75.93 ± 3.15 |  |
| gnnv3h_fga | v3h: anatomical inverse-dist adjacency + gate | 9 | nan | 38.54 ± 3.20 | 74.86 ± 3.29 |  |
| gnnv3c_h5 | v3c: 5 harmonics | 9 | nan | 38.02 ± 3.27 | 74.69 ± 3.29 |  |
| gnnv3c_ng | v3c: per-node gate | 9 | nan | 37.30 ± 3.10 | 74.21 ± 3.21 |  |
| gnnv3b3_fg2 | v3 fg2: learned gated adjacency | 9 | nan | 38.15 ± 3.19 | 74.18 ± 3.34 |  |
| gnnv3c_ngh5 | v3c: per-node gate + 5 harmonics | 9 | nan | 37.25 ± 3.27 | 74.01 ± 3.25 |  |
| gnnv3d_fgu | v3d: fixed uniform adjacency + gate | 9 | nan | 38.06 ± 3.19 | 73.93 ± 3.35 |  |
| gnnv3b3_fg3 | v3 fg3: template front end, no graph | 9 | nan | 34.69 ± 3.18 | 71.52 ± 3.27 |  |
| gnnv3e_cm | v3e: fixed channel-mean contrast | 9 | nan | 32.08 ± 2.96 | 68.50 ± 3.56 |  |
| gnnv3a_pb | v3a: per-band + layer-wise adj (conv front end) | 9 | nan | 33.29 ± 2.98 | 67.33 ± 3.53 |  |

### Table 3 – GNN v3 family, 19 channels (separate)
| method | description | channels | params | 0.5 s acc ± SE | 1.0 s acc ± SE | note |
|---|---|---|---|---|---|---|
| gnnv3l_b5d19 | v3l b5d19: dense + 5-band bank C, 19 ch  [seed 41] | 19 | nan | 42.95 ± 3.39 | 78.24 ± 3.12 |  |
| gnnv3m_s42b5d19 | v3m: b5d19 seed 42 | 19 | nan | 42.52 ± 3.39 | 77.95 ± 3.03 |  |
| gnnv3m_s43b5d19 | v3m: b5d19 seed 43 | 19 | nan | 43.20 ± 3.35 | 77.88 ± 3.02 |  |
| gnnv3k_s42d19 | v3k: dense19 seed 42 | 19 | nan | 41.21 ± 3.24 | 77.71 ± 3.09 |  |
| gnnv3j_aw19 | v3j aw19: trial-adaptive node weights, 19 ch | 19 | nan | 41.62 ± 3.27 | 77.62 ± 3.01 |  |
| gnnv3g_dense19 | v3g dense19: free signed map, 19 ch  [seed 41] | 19 | nan | 41.92 ± 3.30 | 77.54 ± 3.01 |  |
| gnnv3k_s43d19 | v3k: dense19 seed 43 | 19 | nan | 40.96 ± 3.32 | 77.26 ± 3.10 |  |
| gnnv3i_wide19 | v3i wide19: dense19 4x wider | 19 | nan | 41.55 ± 3.21 | 77.17 ± 3.09 |  |
| gnnv3g_knn19 | v3g knn19: kNN signed map, 19 ch | 19 | nan | 39.89 ± 3.32 | 76.96 ± 3.07 |  |

### Table 4 – GNN v3 family, 64 channels (separate)
| method | description | channels | params | 0.5 s acc ± SE | 1.0 s acc ± SE | note |
|---|---|---|---|---|---|---|
| gnnv3g_dense64 | v3g dense64: free signed map, 64 ch | 64 | nan | 40.11 ± 3.14 | – |  |
| gnnv3g_knn64 | v3g knn64: kNN signed map, 64 ch | 64 | nan | 39.69 ± 3.28 | – |  |

### Table 5 – Earlier GNN generations v1/v2
| method | description | channels | params | 0.5 s acc ± SE | 1.0 s acc ± SE | note |
|---|---|---|---|---|---|---|
| gnnv2d_g2_uniform | v2d ablation: uniform adjacency | 9 | nan | – | 67.25 ± 3.50 |  |
| gnnv2c_g2 | v2c: reference, learned graph | 9 | nan | 32.56 ± 3.00 | 66.98 ± 3.50 |  |
| gnnv2b_g2 | v2b: GroupNorm, graph | 9 | nan | 33.06 ± 2.97 | 66.05 ± 3.56 |  |
| gnnv2_g2 | v2: conv+GCN, learned graph | 9 | nan | 32.13 ± 2.97 | 65.65 ± 3.54 |  |
| gnnv2d_g2_anatomical | INVALID: v2d anatomical (polar-coordinate bug) | 9 | nan | – | 62.77 ± 3.48 | ⚠ INVALID |
| gnnv2d_g2_identity | v2d ablation: identity adjacency | 9 | nan | – | 59.87 ± 3.39 |  |
| gnnv2c_g3 | v2c: no graph | 9 | nan | 29.14 ± 2.65 | 54.73 ± 3.48 |  |
| gnnv2_g3 | v2: conv, no graph | 9 | nan | 29.93 ± 2.70 | 52.98 ± 3.46 |  |
| gnnv2b_g3 | v2b: GroupNorm, no graph | 9 | nan | 29.87 ± 2.69 | 52.98 ± 3.46 |  |
| gnn | v1: first conv+GCN | 9 | nan | 13.38 ± 1.18 | 22.76 ± 1.94 |  |

### Table 6 – Seed comparison: dense19 vs b5d19 (3 seeds each)
| window | group | seed41 | seed42 | seed43 | mean | sd_seeds | ci_lo | ci_hi | wins_35 | p_wilcox | p_welch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0p5 | dense19 | 41.92 | 41.21 | 40.96 | 41.37 | 0.494 | nan | nan | nan | nan | nan |
| 0p5 | b5d19 | 42.95 | 42.52 | 43.2 | 42.89 | 0.343 | nan | nan | nan | nan | nan |
| 0p5 | b5d19 - dense19 |  |  |  | 1.528 |  | 0.829 | 2.294 | 27.0 | 0.0001 | 0.0151 |
| 1p0 | dense19 | 77.54 | 77.71 | 77.26 | 77.5 | 0.228 | nan | nan | nan | nan | nan |
| 1p0 | b5d19 | 78.24 | 77.95 | 77.88 | 78.02 | 0.189 | nan | nan | nan | nan | nan |
| 1p0 | b5d19 - dense19 |  |  |  | 0.52 |  | 0.0 | 1.044 | 24.0 | 0.0445 | 0.0401 |

### Table 7 – Hybrid GNN + CCA fusion  (fuse4 = net+(f2+f3)/2; full adds FBCCA sin/cos)
| net_label | window | n_folds | net_alone | cca_only | fuse4 | full | fuse4_ITR | full_ITR | fuse4_vs_cnnref_diff | fuse4_vs_cnnref_ci | fuse4_vs_cnnref_p | fuse4_ahead_35 | fuse_full_vs_cnnref_diff | fuse_full_vs_cnnref_ci | fuse_full_vs_cnnref_p | fuse_full_ahead_35 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnnref | 0p5 | 35 | 41.54 | 43.96 | 51.65 | 51.86 | 114.59 | 115.47 | nan | nan | nan | nan | nan | nan | nan | nan |
| GNN lr 9ch | 0p5 | 35 | 39.83 | 43.96 | 51.05 | 50.46 | 112.54 | 110.78 | -0.61 | [-2.96, 2.32] | 0.2044 | 13.0 | -1.39 | [-3.92, 1.69] | 0.0103 | 8.0 |
| GNN b5d9 9ch | 0p5 | 35 | 40.99 | 43.96 | 51.74 | 50.75 | 114.74 | 111.48 | 0.08 | [-2.38, 3.07] | 0.4328 | 14.0 | -1.11 | [-3.77, 2.08] | 0.0346 | 7.0 |
| GNN dense19 | 0p5 | 35 | 41.92 | 43.96 | 52.57 | 52.1 | 117.12 | 115.57 | 0.92 | [-1.43, 3.95] | 0.7896 | 20.0 | 0.24 | [-2.30, 3.54] | 0.2132 | 12.0 |
| GNN b5d19 | 0p5 | 35 | 42.95 | 43.96 | 52.65 | 51.94 | 117.65 | 115.3 | 1.0 | [-1.50, 4.19] | 0.7817 | 19.0 | 0.08 | [-2.33, 3.32] | 0.2171 | 11.0 |
| GNN dense19 x3 | 0p5 | 35 | 42.13 | 43.96 | 52.65 | 51.96 | 117.64 | 115.23 | 1.0 | [-1.49, 4.33] | 0.9329 | 16.0 | 0.11 | [-2.45, 3.37] | 0.1438 | 12.0 |
| GNN b5d19 x3 | 0p5 | 35 | 44.02 | 43.96 | 53.25 | 52.4 | 119.55 | 116.65 | 1.6 | [-0.83, 4.96] | 0.5494 | 20.0 | 0.55 | [-1.88, 3.86] | 0.2756 | 12.0 |
| cnnref | 1p0 | 35 | 68.32 | 80.77 | 82.31 | 83.39 | 154.62 | 157.71 | nan | nan | nan | nan | nan | nan | nan | nan |
| GNN lr 9ch | 1p0 | 35 | 76.1 | 80.77 | 84.67 | 84.63 | 160.94 | 160.69 | 2.36 | [0.14, 5.24] | 0.2181 | 20.0 | 1.24 | [-0.87, 3.91] | 0.8374 | 16.0 |
| GNN b5d9 9ch | 1p0 | 35 | 76.92 | 80.77 | 85.12 | 84.92 | 162.19 | 161.4 | 2.81 | [0.63, 5.75] | 0.0563 | 18.0 | 1.52 | [-0.76, 4.56] | 0.9004 | 15.0 |
| GNN dense19 | 1p0 | 35 | 77.54 | 80.77 | 85.51 | 85.42 | 163.56 | 163.22 | 3.2 | [0.93, 6.37] | 0.0028 | 26.0 | 2.02 | [-0.42, 5.24] | 0.2305 | 19.0 |
| GNN b5d19 | 1p0 | 35 | 78.24 | 80.77 | 85.62 | 85.39 | 164.14 | 163.39 | 3.31 | [0.89, 6.54] | 0.0086 | 23.0 | 2.0 | [-0.43, 5.24] | 0.4688 | 17.0 |
| GNN dense19 x3 | 1p0 | 35 | 78.19 | 80.77 | 85.67 | 85.52 | 164.23 | 163.78 | 3.36 | [0.95, 6.56] | 0.0057 | 24.0 | 2.13 | [-0.26, 5.51] | 0.4311 | 19.0 |
| GNN b5d19 x3 | 1p0 | 35 | 79.04 | 80.77 | 85.96 | 85.81 | 165.01 | 164.47 | 3.65 | [1.27, 7.00] | 0.0013 | 26.0 | 2.42 | [-0.04, 5.87] | 0.2533 | 16.0 |

### Table 8 – Subject-group breakdown
| method | description | window | all_35 | exp_S01-08 | naive_S09-35 | hard4_mean | hard_S11 | hard_S19 | hard_S29 | hard_S33 | other_31 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cca_net_off | CCA-Net replication: CNN + Modified FBtt-CCA + FBCCA(SST) | 0p5 | 51.95 | 61.09 | 49.24 | 21.04 | 9.58 | 16.25 | 26.25 | 32.08 | 55.94 |
| cca_net_off | CCA-Net replication: CNN + Modified FBtt-CCA + FBCCA(SST) | 1p0 | 82.48 | 91.3 | 79.86 | 45.73 | 25.0 | 52.5 | 58.75 | 46.67 | 87.22 |
| fbttcca_yf | FBtt-CCA (classical) | 0p5 | 39.15 | 43.91 | 37.75 | 13.44 | 8.75 | 8.33 | 14.17 | 22.5 | 42.47 |
| fbttcca_yf | FBtt-CCA (classical) | 1p0 | 77.71 | 85.83 | 75.31 | 39.17 | 21.67 | 37.92 | 48.75 | 48.33 | 82.69 |
| cnn_ref | CCA-Net CNN branch alone (CSDuDoFN-like) | 0p5 | 41.54 | 49.01 | 39.32 | 15.73 | 6.25 | 17.08 | 17.5 | 22.08 | 44.87 |
| cnn_ref | CCA-Net CNN branch alone (CSDuDoFN-like) | 1p0 | 68.32 | 79.84 | 64.91 | 31.35 | 13.75 | 42.08 | 37.5 | 32.08 | 73.09 |
| gnnv3e_lr | v3e lr: learned signed 9x9 spatial map (best 9-ch) | 0p5 | 39.83 | 43.8 | 38.66 | 16.15 | 5.83 | 16.25 | 15.0 | 27.5 | 42.89 |
| gnnv3e_lr | v3e lr: learned signed 9x9 spatial map (best 9-ch) | 1p0 | 76.1 | 83.39 | 73.94 | 36.77 | 18.75 | 36.25 | 38.75 | 53.33 | 81.17 |
| gnnv3g_dense19 | v3g dense19: free signed map, 19 ch  [seed 41] | 0p5 | 41.92 | 46.3 | 40.62 | 15.83 | 6.67 | 12.92 | 15.83 | 27.92 | 45.28 |
| gnnv3g_dense19 | v3g dense19: free signed map, 19 ch  [seed 41] | 1p0 | 77.54 | 84.22 | 75.56 | 37.81 | 17.5 | 45.42 | 40.0 | 48.33 | 82.66 |
| gnnv3l_b5d19 | v3l b5d19: dense + 5-band bank C, 19 ch  [seed 41] | 0p5 | 42.95 | 45.78 | 42.11 | 16.88 | 5.83 | 12.5 | 20.83 | 28.33 | 46.32 |
| gnnv3l_b5d19 | v3l b5d19: dense + 5-band bank C, 19 ch  [seed 41] | 1p0 | 78.24 | 84.58 | 76.36 | 36.46 | 15.0 | 39.58 | 40.83 | 50.42 | 83.63 |

### Table 9 – Paired tests (nominal Wilcoxon p; ~85 tests total)
| A | B | window | A_mean | B_mean | diff_A-B | ci_lo | ci_hi | A_ahead_35 | p_wilcoxon |
|---|---|---|---|---|---|---|---|---|---|
| gnnv3b3_fg2 | gnnv3b3_fg3 | 0p5 | 38.15 | 34.69 | 3.46 | 2.39 | 4.61 | 29 | 0.0 |
| gnnv3b3_fg2 | gnnv3b3_fg3 | 1p0 | 74.18 | 71.52 | 2.65 | 1.61 | 3.74 | 28 | 0.0001 |
| gnnv3d_fgu | gnnv3b3_fg3 | 0p5 | 38.06 | 34.69 | 3.37 | 2.26 | 4.49 | 29 | 0.0 |
| gnnv3d_fgu | gnnv3b3_fg3 | 1p0 | 73.93 | 71.52 | 2.4 | 1.43 | 3.39 | 27 | 0.0002 |
| gnnv3e_lr | gnnv3b3_fg2 | 0p5 | 39.83 | 38.15 | 1.68 | 0.58 | 2.79 | 26 | 0.0096 |
| gnnv3e_lr | gnnv3b3_fg2 | 1p0 | 76.1 | 74.18 | 1.92 | 0.93 | 3.02 | 25 | 0.0011 |
| gnnv3e_cm | gnnv3b3_fg3 | 0p5 | 32.08 | 34.69 | -2.61 | -5.64 | 0.33 | 12 | 0.084 |
| gnnv3e_cm | gnnv3b3_fg3 | 1p0 | 68.5 | 71.52 | -3.02 | -6.79 | 0.42 | 11 | 0.1438 |
| gnnv3h_fga | gnnv3d_fgu | 0p5 | 38.54 | 38.06 | 0.48 | 0.1 | 0.88 | 17 | 0.0289 |
| gnnv3h_fga | gnnv3d_fgu | 1p0 | 74.86 | 73.93 | 0.93 | 0.54 | 1.38 | 23 | 0.0001 |
| gnnv3g_dense19 | gnnv3e_lr | 0p5 | 41.92 | 39.83 | 2.08 | 0.94 | 3.24 | 24 | 0.0011 |
| gnnv3g_dense19 | gnnv3e_lr | 1p0 | 77.54 | 76.1 | 1.44 | -0.13 | 3.16 | 19 | 0.081 |
| gnnv3g_dense19 | gnnv3g_knn19 | 0p5 | 41.92 | 39.89 | 2.02 | 1.02 | 3.01 | 24 | 0.0008 |
| gnnv3g_dense19 | gnnv3g_knn19 | 1p0 | 77.54 | 76.96 | 0.57 | -0.11 | 1.29 | 22 | 0.2008 |
| gnnv3i_wide19 | gnnv3g_dense19 | 0p5 | 41.55 | 41.92 | -0.37 | -1.1 | 0.36 | 14 | 0.3125 |
| gnnv3i_wide19 | gnnv3g_dense19 | 1p0 | 77.17 | 77.54 | -0.37 | -1.21 | 0.49 | 15 | 0.4208 |
| gnnv3j_aw19 | gnnv3g_dense19 | 0p5 | 41.62 | 41.92 | -0.3 | -0.71 | 0.12 | 10 | 0.1009 |
| gnnv3j_aw19 | gnnv3g_dense19 | 1p0 | 77.62 | 77.54 | 0.08 | -0.24 | 0.39 | 16 | 0.4925 |
| gnnv3j_aw9 | gnnv3e_lr | 0p5 | 40.76 | 39.83 | 0.93 | 0.06 | 1.77 | 25 | 0.03 |
| gnnv3j_aw9 | gnnv3e_lr | 1p0 | 76.18 | 76.1 | 0.08 | -0.64 | 0.79 | 18 | 0.5306 |
| gnnv3l_b5d19 | gnnv3g_dense19 | 0p5 | 42.95 | 41.92 | 1.04 | 0.06 | 2.04 | 20 | 0.0934 |
| gnnv3l_b5d19 | gnnv3g_dense19 | 1p0 | 78.24 | 77.54 | 0.7 | -0.14 | 1.52 | 22 | 0.0588 |
| gnnv3l_b5d9 | gnnv3e_lr | 0p5 | 40.99 | 39.83 | 1.15 | 0.02 | 2.26 | 22 | 0.037 |
| gnnv3l_b5d9 | gnnv3e_lr | 1p0 | 76.92 | 76.1 | 0.82 | 0.05 | 1.67 | 22 | 0.0625 |
| gnnv3k_s42d19 | gnnv3g_dense19 | 0p5 | 41.21 | 41.92 | -0.7 | -1.49 | 0.05 | 12 | 0.1078 |
| gnnv3k_s42d19 | gnnv3g_dense19 | 1p0 | 77.71 | 77.54 | 0.18 | -0.55 | 0.9 | 16 | 0.64 |
| gnnv3k_s43d19 | gnnv3g_dense19 | 0p5 | 40.96 | 41.92 | -0.95 | -1.73 | -0.14 | 10 | 0.0205 |
| gnnv3k_s43d19 | gnnv3g_dense19 | 1p0 | 77.26 | 77.54 | -0.27 | -1.14 | 0.54 | 16 | 0.9374 |
| cca_net_off | gnnv3g_dense19 | 0p5 | 51.95 | 41.92 | 10.04 | 6.82 | 13.35 | 33 | 0.0 |
| cca_net_off | gnnv3g_dense19 | 1p0 | 82.48 | 77.54 | 4.94 | 2.33 | 7.46 | 29 | 0.0001 |
| cca_net_off | gnnv3l_b5d19 | 0p5 | 51.95 | 42.95 | 9.0 | 5.82 | 12.37 | 30 | 0.0 |
| cca_net_off | gnnv3l_b5d19 | 1p0 | 82.48 | 78.24 | 4.24 | 1.5 | 6.92 | 26 | 0.0013 |
| gnnv3g_dense19 | cnn_ref | 0p5 | 41.92 | 41.54 | 0.38 | -2.74 | 3.81 | 15 | 0.5329 |
| gnnv3g_dense19 | cnn_ref | 1p0 | 77.54 | 68.32 | 9.21 | 5.36 | 13.94 | 30 | 0.0 |
| gnnv3l_b5d19 | cnn_ref | 0p5 | 42.95 | 41.54 | 1.42 | -1.87 | 4.96 | 18 | 0.8042 |
| gnnv3l_b5d19 | cnn_ref | 1p0 | 78.24 | 68.32 | 9.92 | 5.8 | 15.18 | 29 | 0.0 |
| fbttcca_yf | gnnv3g_dense19 | 0p5 | 39.15 | 41.92 | -2.76 | -6.86 | 1.55 | 10 | 0.07 |
| fbttcca_yf | gnnv3g_dense19 | 1p0 | 77.71 | 77.54 | 0.18 | -3.38 | 3.92 | 17 | 0.9929 |

