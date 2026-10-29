# Consolidated results (35-fold leave-one-subject-out, mean accuracy % +- SE; zero-shot, Benchmark dataset)
Nominal p-values; about 85 paired tests were run. 9-channel rows are protocol-comparable with CCA-Net; 19/64-channel rows are a separate table.

### Table 1: baselines
| method | description | channels | 0.5 s | 1.0 s |
|---|---|---|---|---|
| cca_net_off | CCA-Net replication: CNN + Modified FBtt-CCA + FBCCA(SST) | 9 | 51.95 +- 3.54 | 82.48 +- 3.03 |
| fbttcca_yf | FBtt-CCA (classical) | 9 | 39.15 +- 3.21 | 77.71 +- 3.28 |
| cnn_ref | CCA-Net CNN branch alone (CSDuDoFN-like) | 9 | 41.54 +- 3.66 | 68.32 +- 3.96 |
| fbcca | FBCCA (classical) | 9 | 20.64 +- 1.80 | 63.49 +- 3.53 |

### Table 2: GNN family, 9 channels (protocol-comparable)
| method | description | channels | 0.5 s | 1.0 s |
|---|---|---|---|---|
| gnnv3l_b5d9 | dense, 5-band bank C, 9 ch | 9 | 40.99 +- 3.39 | 76.92 +- 3.08 |
| gnnv3j_aw9 | trial-adaptive node weights, 9 ch | 9 | 40.76 +- 3.31 | 76.18 +- 3.19 |
| gnnv3e_lr | learned signed 9x9 map | 9 | 39.83 +- 3.35 | 76.10 +- 3.18 |
| gnnv3g_knn9 | kNN signed map, 9 ch | 9 | 39.08 +- 3.23 | 75.93 +- 3.15 |
| gnnv3h_fga | anatomical adjacency + gate | 9 | 38.54 +- 3.20 | 74.86 +- 3.29 |
| gnnv3c_h5 | 5 harmonics | 9 | 38.02 +- 3.27 | 74.69 +- 3.29 |
| gnnv3c_ng | per-node gate | 9 | 37.30 +- 3.10 | 74.21 +- 3.21 |
| gnnv3b3_fg2 | learned gated adjacency | 9 | 38.15 +- 3.19 | 74.18 +- 3.34 |
| gnnv3c_ngh5 | per-node gate + 5 harmonics | 9 | 37.25 +- 3.27 | 74.01 +- 3.25 |
| gnnv3d_fgu | uniform adjacency + gate | 9 | 38.06 +- 3.19 | 73.93 +- 3.35 |
| gnnv3b3_fg3 | template front end, no graph | 9 | 34.69 +- 3.18 | 71.52 +- 3.27 |
| gnnv3e_cm | fixed channel-mean contrast | 9 | 32.08 +- 2.96 | 68.50 +- 3.56 |
| gnnv3a_pb | per-band + layer-wise adjacency (conv front end) | 9 | 33.29 +- 2.98 | 67.33 +- 3.53 |

### Table 3: GNN family, 19 / 64 channels (separate table)
| method | description | channels | 0.5 s | 1.0 s |
|---|---|---|---|---|
| gnnv3l_b5d19 | dense, 5-band bank C, 19 ch | 19 | 42.95 +- 3.39 | 78.24 +- 3.12 |
| gnnv3m_s42b5d19 | b5d19 seed 42 | 19 | 42.52 +- 3.39 | 77.95 +- 3.03 |
| gnnv3m_s43b5d19 | b5d19 seed 43 | 19 | 43.20 +- 3.35 | 77.88 +- 3.02 |
| gnnv3k_s42d19 | dense19 seed 42 | 19 | 41.21 +- 3.24 | 77.71 +- 3.09 |
| gnnv3j_aw19 | trial-adaptive node weights, 19 ch | 19 | 41.62 +- 3.27 | 77.62 +- 3.01 |
| gnnv3g_dense19 | free signed map, 19 ch | 19 | 41.92 +- 3.30 | 77.54 +- 3.01 |
| gnnv3k_s43d19 | dense19 seed 43 | 19 | 40.96 +- 3.32 | 77.26 +- 3.10 |
| gnnv3i_wide19 | dense19, 4x wider | 19 | 41.55 +- 3.21 | 77.17 +- 3.09 |
| gnnv3g_knn19 | kNN signed map, 19 ch | 19 | 39.89 +- 3.32 | 76.96 +- 3.07 |
| gnnv3g_dense64 | free signed map, 64 ch | 64 | 40.11 +- 3.14 | - |
| gnnv3g_knn64 | kNN signed map, 64 ch | 64 | 39.69 +- 3.28 | - |

### Table 4: earlier GNN generations (v1/v2)
| method | description | channels | 0.5 s | 1.0 s |
|---|---|---|---|---|
| gnnv2d_g2_uniform | graph ablation: uniform | 9 | - | 67.25 +- 3.50 |
| gnnv2c_g2 | v2c reference, learned graph | 9 | 32.56 +- 3.00 | 66.98 +- 3.50 |
| gnnv2b_g2 | v2 + GroupNorm, graph | 9 | 33.06 +- 2.97 | 66.05 +- 3.56 |
| gnnv2_g2 | conv+GCN, learned graph | 9 | 32.13 +- 2.97 | 65.65 +- 3.54 |
| gnnv2d_g2_anatomical | INVALID: polar-coordinate bug in anatomical adjacency | 9 | - | 62.77 +- 3.48 |
| gnnv2d_g2_identity | graph ablation: identity | 9 | - | 59.87 +- 3.39 |
| gnnv2c_g3 | v2c, no graph | 9 | 29.14 +- 2.65 | 54.73 +- 3.48 |
| gnnv2_g3 | conv, no graph | 9 | 29.93 +- 2.70 | 52.98 +- 3.46 |
| gnnv2b_g3 | v2 + GroupNorm, no graph | 9 | 29.87 +- 2.69 | 52.98 +- 3.46 |
| gnn | first conv+GCN | 9 | 13.38 +- 1.18 | 22.76 +- 1.94 |

### Table 5: seeds (b5d19 vs dense19)
| window | group | seed_means | mean | sd_of_seed_means |
|---|---|---|---|---|
| 0p5 | dense19 | 41.92 / 41.21 / 40.96 | 41.365 | 0.494 |
| 0p5 | b5d19 | 42.95 / 42.52 / 43.20 | 42.893 | 0.343 |
| 0p5 | b5d19 - dense19 | CI [0.83, 2.29]  wins 27/35  Wilcoxon p=0.000  Welch p=0.015 | 1.528 | nan |
| 1p0 | dense19 | 77.54 / 77.71 / 77.26 | 77.504 | 0.228 |
| 1p0 | b5d19 | 78.24 / 77.95 / 77.88 | 78.024 | 0.189 |
| 1p0 | b5d19 - dense19 | CI [0.00, 1.04]  wins 24/35  Wilcoxon p=0.045  Welch p=0.040 | 0.52 | nan |

### Table 6: hybrid GNN + CCA fusion (fuse4 = net+(f2+f3)/2, official rule; full adds FBCCA sin/cos)
| net | window | network_alone | CCA_only_f2f3 | fuse4 | fuse_full | fuse4_ITR | fuse4_minus_cnn | fuse4_p | fuse_full_minus_cnn | fuse_full_p |
|---|---|---|---|---|---|---|---|---|---|---|
| cnnref | 0p5 | 41.536 | 43.964 | 51.655 | 51.857 | 114.594 | nan | nan | nan | nan |
| gnnv2c_g2 | 0p5 | 32.56 | 43.964 | 48.774 | 48.952 | 104.375 | -2.881 | 0.0 | -2.905 | 0.0 |
| dense19 | 0p5 | 41.917 | 43.964 | 52.571 | 52.095 | 117.123 | 0.917 | 0.79 | 0.238 | 0.213 |
| b5d19 | 0p5 | 42.952 | 43.964 | 52.655 | 51.94 | 117.65 | 1.0 | 0.782 | 0.083 | 0.217 |
| b5d9 (9 ch) | 0p5 | 40.988 | 43.964 | 51.738 | 50.75 | 114.743 | 0.083 | 0.433 | -1.107 | 0.035 |
| lr (9 ch) | 0p5 | 39.833 | 43.964 | 51.048 | 50.464 | 112.545 | -0.607 | 0.204 | -1.393 | 0.01 |
| dense19 x3 seeds | 0p5 | 42.131 | 43.964 | 52.655 | 51.976 | 117.595 | 1.0 | 0.993 | 0.119 | 0.151 |
| b5d19 x3 seeds | 0p5 | 44.024 | 43.964 | 53.167 | 52.381 | 119.278 | 1.512 | 0.602 | 0.524 | 0.266 |
| cnnref | 1p0 | 68.321 | 80.774 | 82.31 | 83.393 | 154.622 | nan | nan | nan | nan |
| gnnv2c_g2 | 1p0 | 66.976 | 80.774 | 83.19 | 84.06 | 156.566 | 0.881 | 0.787 | 0.667 | 0.593 |
| dense19 | 1p0 | 77.536 | 80.774 | 85.512 | 85.417 | 163.563 | 3.202 | 0.003 | 2.024 | 0.23 |
| b5d19 | 1p0 | 78.238 | 80.774 | 85.619 | 85.393 | 164.138 | 3.31 | 0.009 | 2.0 | 0.469 |
| b5d9 (9 ch) | 1p0 | 76.917 | 80.774 | 85.119 | 84.917 | 162.19 | 2.81 | 0.056 | 1.524 | 0.9 |
| lr (9 ch) | 1p0 | 76.095 | 80.774 | 84.667 | 84.631 | 160.944 | 2.357 | 0.218 | 1.238 | 0.837 |
| dense19 x3 seeds | 1p0 | 78.19 | 80.774 | 85.667 | 85.488 | 164.228 | 3.357 | 0.006 | 2.095 | 0.453 |
| b5d19 x3 seeds | 1p0 | 79.036 | 80.774 | 85.917 | 85.798 | 164.891 | 3.607 | 0.001 | 2.405 | 0.245 |

### Table 7: subject groups (1.0 s / 0.5 s)
| method | window | all | experienced_S01-08 | naive_S09-35 | hard4_S11_19_29_33 | other31 |
|---|---|---|---|---|---|---|
| cca_net_off | 0p5 | 51.95 | 61.09 | 49.24 | 21.04 | 55.94 |
| cca_net_off | 1p0 | 82.48 | 91.3 | 79.86 | 45.73 | 87.22 |
| fbttcca_yf | 0p5 | 39.15 | 43.91 | 37.75 | 13.44 | 42.47 |
| fbttcca_yf | 1p0 | 77.71 | 85.83 | 75.31 | 39.17 | 82.69 |
| cnn_ref | 0p5 | 41.54 | 49.01 | 39.32 | 15.73 | 44.87 |
| cnn_ref | 1p0 | 68.32 | 79.84 | 64.91 | 31.35 | 73.09 |
| gnnv3e_lr | 0p5 | 39.83 | 43.8 | 38.66 | 16.15 | 42.89 |
| gnnv3e_lr | 1p0 | 76.1 | 83.39 | 73.94 | 36.77 | 81.17 |
| gnnv3g_dense19 | 0p5 | 41.92 | 46.3 | 40.62 | 15.83 | 45.28 |
| gnnv3g_dense19 | 1p0 | 77.54 | 84.22 | 75.56 | 37.81 | 82.66 |
| gnnv3l_b5d19 | 0p5 | 42.95 | 45.78 | 42.11 | 16.88 | 46.32 |
| gnnv3l_b5d19 | 1p0 | 78.24 | 84.58 | 76.36 | 36.46 | 83.63 |

### Table 8: paired tests (nominal p)
| A | B | window | diff_A_minus_B | ci_lo | ci_hi | A_ahead | p_wilcoxon |
|---|---|---|---|---|---|---|---|
| gnnv3b3_fg2 | gnnv3b3_fg3 | 0p5 | 3.464 | 2.393 | 4.607 | 29 | 0.0 |
| gnnv3b3_fg2 | gnnv3b3_fg3 | 1p0 | 2.655 | 1.607 | 3.738 | 28 | 0.0 |
| gnnv3d_fgu | gnnv3b3_fg3 | 0p5 | 3.369 | 2.262 | 4.488 | 29 | 0.0 |
| gnnv3d_fgu | gnnv3b3_fg3 | 1p0 | 2.405 | 1.429 | 3.393 | 27 | 0.0 |
| gnnv3e_lr | gnnv3b3_fg2 | 0p5 | 1.679 | 0.583 | 2.786 | 26 | 0.01 |
| gnnv3e_lr | gnnv3b3_fg2 | 1p0 | 1.917 | 0.928 | 3.024 | 25 | 0.001 |
| gnnv3e_cm | gnnv3b3_fg3 | 0p5 | -2.607 | -5.643 | 0.333 | 12 | 0.084 |
| gnnv3e_cm | gnnv3b3_fg3 | 1p0 | -3.024 | -6.786 | 0.417 | 11 | 0.144 |
| gnnv3h_fga | gnnv3d_fgu | 0p5 | 0.476 | 0.095 | 0.881 | 17 | 0.029 |
| gnnv3h_fga | gnnv3d_fgu | 1p0 | 0.929 | 0.536 | 1.381 | 23 | 0.0 |
| gnnv3g_dense19 | gnnv3e_lr | 0p5 | 2.083 | 0.94 | 3.238 | 24 | 0.001 |
| gnnv3g_dense19 | gnnv3e_lr | 1p0 | 1.44 | -0.131 | 3.155 | 19 | 0.081 |
| gnnv3g_dense19 | gnnv3g_knn19 | 0p5 | 2.024 | 1.024 | 3.012 | 24 | 0.001 |
| gnnv3g_dense19 | gnnv3g_knn19 | 1p0 | 0.571 | -0.107 | 1.286 | 22 | 0.201 |
| gnnv3i_wide19 | gnnv3g_dense19 | 0p5 | -0.369 | -1.095 | 0.357 | 14 | 0.312 |
| gnnv3i_wide19 | gnnv3g_dense19 | 1p0 | -0.369 | -1.214 | 0.488 | 15 | 0.421 |
| gnnv3j_aw19 | gnnv3g_dense19 | 0p5 | -0.298 | -0.714 | 0.119 | 10 | 0.101 |
| gnnv3j_aw19 | gnnv3g_dense19 | 1p0 | 0.083 | -0.238 | 0.393 | 16 | 0.493 |
| gnnv3j_aw9 | gnnv3e_lr | 0p5 | 0.929 | 0.06 | 1.774 | 25 | 0.03 |
| gnnv3j_aw9 | gnnv3e_lr | 1p0 | 0.083 | -0.643 | 0.786 | 18 | 0.531 |
| gnnv3l_b5d19 | gnnv3g_dense19 | 0p5 | 1.036 | 0.06 | 2.036 | 20 | 0.093 |
| gnnv3l_b5d19 | gnnv3g_dense19 | 1p0 | 0.702 | -0.143 | 1.524 | 22 | 0.059 |
| gnnv3l_b5d9 | gnnv3e_lr | 0p5 | 1.155 | 0.024 | 2.262 | 22 | 0.037 |
| gnnv3l_b5d9 | gnnv3e_lr | 1p0 | 0.821 | 0.048 | 1.667 | 22 | 0.062 |
| gnnv3k_s42d19 | gnnv3g_dense19 | 0p5 | -0.702 | -1.488 | 0.048 | 12 | 0.108 |
| gnnv3k_s42d19 | gnnv3g_dense19 | 1p0 | 0.179 | -0.548 | 0.905 | 16 | 0.64 |
| gnnv3k_s43d19 | gnnv3g_dense19 | 0p5 | -0.952 | -1.726 | -0.143 | 10 | 0.02 |
| gnnv3k_s43d19 | gnnv3g_dense19 | 1p0 | -0.274 | -1.143 | 0.536 | 16 | 0.937 |
| cca_net_off | gnnv3g_dense19 | 0p5 | 10.036 | 6.821 | 13.345 | 33 | 0.0 |
| cca_net_off | gnnv3g_dense19 | 1p0 | 4.94 | 2.333 | 7.465 | 29 | 0.0 |
| cca_net_off | gnnv3l_b5d19 | 0p5 | 9.0 | 5.821 | 12.369 | 30 | 0.0 |
| cca_net_off | gnnv3l_b5d19 | 1p0 | 4.238 | 1.5 | 6.917 | 26 | 0.001 |
| gnnv3g_dense19 | cnn_ref | 0p5 | 0.381 | -2.738 | 3.81 | 15 | 0.533 |
| gnnv3g_dense19 | cnn_ref | 1p0 | 9.214 | 5.357 | 13.94 | 30 | 0.0 |
| gnnv3l_b5d19 | cnn_ref | 0p5 | 1.417 | -1.869 | 4.964 | 18 | 0.804 |
| gnnv3l_b5d19 | cnn_ref | 1p0 | 9.917 | 5.797 | 15.179 | 29 | 0.0 |
| fbttcca_yf | gnnv3g_dense19 | 0p5 | -2.762 | -6.857 | 1.548 | 10 | 0.07 |
| fbttcca_yf | gnnv3g_dense19 | 1p0 | 0.179 | -3.382 | 3.917 | 17 | 0.993 |
