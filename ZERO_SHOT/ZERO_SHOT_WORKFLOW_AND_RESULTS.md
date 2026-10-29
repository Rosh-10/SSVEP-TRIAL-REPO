# Zero-shot SSVEP classification with a template-based graph network
# Complete workflow, from raw data to final results (for the report / poster)

State as of 2026-10-25. All accuracies are mean % over 35 leave-one-subject-out folds
(Benchmark dataset), at 0.5 s / 1.0 s windows, unless stated otherwise.
Every number below comes from your own runs (logs, compile_*.txt, per_subject_acc.csv,
all_models.csv, zs_fuse outputs). Items marked [NOT RE-CHECKED] were carried over from
earlier notes and should be verified against the per-fold files before you cite them.

HOW TO USE THIS FILE
- Each STEP has: purpose, exact settings, output, RESULT, and what to put in the report.
- Section 19 is the final result set. Section 23 is the exact command order to reproduce.
- p-values are NOMINAL. About 85 paired tests were run on the same 35 subjects.


======================================================================
0. HEADLINE
======================================================================
Task: classify which of 40 flickering targets (8.0-15.8 Hz) a NEW subject looks at, using
only data from the other 34 subjects (zero-shot), in 0.5 s or 1.0 s of EEG.

| system                                         | 0.5 s | 1.0 s |
|------------------------------------------------|-------|-------|
| FBCCA (classical)                              | 20.64 | 63.49 |
| FBtt-CCA (classical)                           | 39.15 | 77.71 |
| CCA-Net replication (CNN + CCA)                | 51.95 | 82.48 |   (paper: 54.27 / 84.02)
| OUR GNN only, no CCA: best single run (b5d19)  | 42.95 | 78.24 |
| OUR GNN only, 3-seed mean (b5d19)              | 42.89 | 78.02 |
| OUR GNN + CCA hybrid, 3-seed ensemble (b5d19)  | 53.17 | 85.92 |   (fuse4 rule)

Three findings:
 1. A template-based GNN (no CCA inside) reaches about 78% at 1.0 s, 8-10 points above the
    CNN branch of CCA-Net, and it is level with the best classical CCA method.
 2. Its gain over a graph-free model comes from a LEARNED SIGNED SPATIAL MIXING of electrodes,
    not from electrode-position priors, deeper graphs, more capacity or adaptive weights.
 3. Replacing the CNN by this GNN inside the CCA-Net fusion adds about +3 points at 1.0 s
    (85.5-85.9 vs 82.3), but the fused number is mostly carried by the CCA features.
86% was NOT reached by the GNN alone (about 78-79%) nor by the hybrid (85.92).


======================================================================
1. STEP 0 - PROBLEM, DATASET, PROTOCOL, METRICS
======================================================================
Purpose: fix the question and the rules BEFORE any modelling, so every later number is comparable.

1.1 Zero-shot SSVEP classification
  SSVEP = the EEG response of the visual cortex that follows a flickering stimulus. Each target
  flickers at its own frequency/phase, so the correct class is "which frequency is the subject
  looking at". Zero-shot = no data from the test subject is used for training or fitting.
  We follow the CCA-Net protocol (Deng et al., IEEE TIM 2025): train on 34 subjects, test on 1,
  repeated for all 35 subjects.

1.2 Dataset (Benchmark, Wang et al. 2016; readme.txt, Sub_info.txt)
  - 35 healthy subjects (17 female, age 17-34, mean 22). Groups: experienced S01-S08 (8),
    naive S09-S35 (27). All right-handed.
  - 40 targets, 8.0-15.8 Hz in steps of 0.2 Hz, joint frequency-phase coding, adjacent
    frequencies differ by 0.5 pi in phase. Frequencies/phases are in Freq_Phase.mat.
  - 6 blocks x 40 trials = 240 trials per subject, random target order inside a block.
  - Trial timeline: 0.5 s cue (red square) -> 5 s flicker -> 0.5 s blank (6 s per trial).
  - EEG: Synamps2, 64 channels (10-20), 1000 Hz, 0.15-200 Hz passband, 50 Hz notch, reference at
    vertex. Stored epochs: 0.5 s pre-stimulus + 5.5 s post, downsampled to 250 Hz.
  - File layout: S01.mat ... S35.mat, variable 'data' = [64 electrodes, 1500 samples, 40 targets, 6 blocks].
  NOT the BETA dataset.

1.3 Evaluation protocol
  - Leave-one-subject-out (LOSO): 35 folds. Test = 240 trials of one subject. Train = the 34 others
    (34 x 240 = 8160 windows).
  - Windows: 0.5 s (125 samples) and 1.0 s (250 samples), starting 0.64 s after the epoch start
    (= 0.5 s pre-stimulus + 0.14 s visual delay = sample 160).
  - Main channel set: the 9 CCA-Net channels (Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2).
    19 channels (P/PO/O region) and all 64 are reported in SEPARATE tables.

1.4 Metrics and statistics
  - Accuracy per subject; mean over 35 subjects; SE = std / sqrt(35).
  - ITR (bits/min) = [log2 M + P log2 P + (1-P) log2((1-P)/(M-1))] * 60 / T, M = 40,
    T = window length + 0.5 s gaze shift (same as CCA-Net). NOTE: the DDGCNN paper divides by the
    window length only, so their ITR values are NOT comparable.
  - Paired comparisons: Wilcoxon signed-rank on per-subject accuracies (35 pairs), plus a
    bootstrap 95% CI (5000 resamples of subjects, seed 0). A single-subject accuracy has SE of
    about 3 points, so only paired 35-subject tests count.
  - Seeds: all runs use seed 41 unless stated. Seeds 42/43 exist for dense19 and b5d19.
RESULT of step 0: protocol only. For the report: dataset table + protocol diagram (section 20).


======================================================================
2. STEP 1 - LOADING AND CHANNEL SELECTION            (zs_data.py, zs_data64.py, zs_elec.py)
======================================================================
Settings
  - 9 channels, 0-based indices [47,53,54,55,56,57,60,61,62] = Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
    (verified against 64-channels.loc).
  - 19 channels = P/PO/O region (zs_elec.region()).  64 channels = all.
  - 64-channels.loc is EEGLAB POLAR format (index, theta in degrees, radius, label). zs_elec converts
    it to x,y correctly. An early script treated theta/radius as x/y (a bug, see section 22).
Output: array (channels, 1500, 40 targets, 6 blocks) per subject.
RESULT: no accuracy yet. Verification: the classical baselines built on this loader reproduce the
        paper exactly (section 7), which validates loading, indexing and windowing.


======================================================================
3. STEP 2 - WINDOWING AND TRIAL ORDER
======================================================================
Settings: window start sample 160, length T = 125 or 250 samples. Trial index = target*6 + block
(label-major, block-minor) -> 240 trials per subject, labels 0-39.
Why 160: the stimulus starts after the 0.5 s cue (sample 125) and the visual system needs about 0.14 s
(35 samples) to respond, as in CCA-Net.
Output: X (240, N_channels, T), y (240,).
RESULT: none (preprocessing).


======================================================================
4. STEP 3 - FILTER BANKS AND STANDARDISATION
======================================================================
Three filter banks, all zero-phase (filtfilt / sosfiltfilt), Chebyshev type I:
  Bank A (the GNN input, 3 bands): pass-band edges start at 6, 14, 22 Hz, all up to 90 Hz;
        stop 4/10/16 Hz and 100 Hz; passband ripple 0.5 dB, gpass 3, gstop 40.
  Bank B (FBtt-CCA): 5 bands (5-90, 14-90, 22-90, 30-90, 38-90 Hz), order 15, SOS.
  Bank C (FBCCA, and the 'b5' GNN variant): 5 bands starting at 6, 14, 22, 30, 38 Hz up to 90 Hz.
        Bands 1-3 of bank C are identical to bank A (checked: max difference 0.0).
Standardisation (FB_stand): per trial, per band, per channel, z-score over time.
  -> This uses ONLY the trial itself. No statistics from the test subject's other trials.
Output for the GNN: tensor (240 trials, 3 bands, N channels, T).
RESULT: none directly; the classical baselines use banks B and C (section 7).


======================================================================
5. STEP 4 - FOLDS AND TRAINING RECIPE (shared by all GNN runs)
======================================================================
  - 35 folds, train on all 34 source subjects, NO validation set, NO early stopping, 100 epochs,
    Adam lr 2e-4, batch 128, cross-entropy, last-epoch model, seed 41, deterministic.
  - One JSON per fold in results/ (accuracy, ITR, predictions, labels, learned graph); trained
    weights in results/models/.
  - Why no validation: a validation set taken from source subjects would add tuning freedom; the
    recipe is copied from the CCA-Net CNN reference so results stay comparable.
RESULT: none. For the report: state this recipe once; every GNN row uses it.


======================================================================
6. STEP 5 - CLASSICAL BASELINES                       (zs_runner.py, zs_baselines.py, zs_cca_official.py)
======================================================================
CCA finds linear combinations of electrodes that maximise correlation with sin/cos templates of a
candidate frequency. FBCCA does this per filter band. FBtt-CCA adds transferred templates from
source subjects.
RESULT:
| method   | 0.5 s | 1.0 s | paper (Table III) |
| FBCCA    | 20.64 | 63.49 | 20.64 / 63.49 (exact match)
| FBtt-CCA | 39.15 | 77.71 | 39.15 / 77.71 (exact match)
Exact agreement with the published numbers validates data loading, windowing, filters and folds.


======================================================================
7. STEP 6 - CCA-NET REPLICATION AND THE CCA FEATURE CACHE   (zs_cnn_ref.py, zs_cca_net.py, zs_cca_feats.py)
======================================================================
CCA-Net (paper) = CSDuDoFN (a CNN fed raw + GMLST-transformed data) + Modified FBtt-CCA, fused by
summing min-max-normalised scores.
  - GMLST coefficient and the subject-specific template (SST) are fitted on SOURCE subjects only;
    the target enters as a single trial. So CCA-Net is strictly zero-shot (confirmed from the paper
    text and from the code).
Our pieces:
  - cnn_ref : the released CNN branch alone.
  - cca_net_off : the full replication (CNN + CCA features).
  - zs_cca_feats.py : saves the CCA features once per fold: f1 = FBCCA (sin/cos), f2 = Modified
    FBtt-CCA (SST), f3 = FBCCA (SST), each (240, 40).
RESULT:
| method                                  | 0.5 s | 1.0 s | paper        |
| CNN branch alone (cnn_ref)              | 41.54 | 68.32 | CSDuDoFN 42.82 / 70.12
| CCA only (f2+f3) = Modified FBtt-CCA    | 43.96 | 80.77 | 43.96 / 80.77 (exact match)
| CCA-Net replication (cca_net_off)       | 51.95 | 82.48 | 54.27 / 84.02
| same decision rule, cnnref logits       | 51.65 | 82.31 |  (zs_fuse.py cross-check)
| + FBCCA sin/cos term ('full' rule)      | 51.86 | 83.39 |  (cca_net_off full: 83.15 at 1.0 s)
Replication gap to the paper: -2.3 points at 0.5 s, -1.5 at 1.0 s. State it openly.
Cross-check: fusing the saved cnn_ref logits with the cached CCA features reproduces cca_net_off
within 0.3 points (82.31 vs 82.48 at 1.0 s; 51.65 vs 51.95 at 0.5 s), which validates the whole fusion tool.


======================================================================
8. STEP 7 - FIRST GNN GENERATIONS (data-driven)       (zs_gnn_model/train.py, v2, v2b, v2c, v2d)
======================================================================
Idea: electrodes are nodes of a graph; a graph convolution with a learned adjacency mixes them.
Front end: temporal convolutions on the filtered signal; GCN layers; fully connected head.
RESULT:
| model                                 | params | 0.5 s | 1.0 s |
| v1 conv + GCN                         | 18k    | 13.38 | 22.76 |
| v2 with graph (g2) / without (g3)     |        | 32.13 / 29.93 | 65.65 / 52.98 |
| v2b (GroupNorm) g2 / g3               |        | 33.06 / 29.87 | 66.05 / 52.98 |
| v2c g2 / g3                           | ~336k  | 32.56 / 29.14 | 66.98 / 54.73 |
| v2d (1.0 s): identity / uniform       |        | -     | 59.87 / 67.25 |
| v2d anatomical                        |        | -     | 62.77  INVALID (bug, section 22) |
Lessons: a graph clearly helps this model (v2c g2 - g3 = +3.4 / +12.3), but a UNIFORM fixed graph
(67.25) matches the learned one (66.98): what matters is mixing electrodes, not which electrodes.
These models are far below the classical methods and have 15x more parameters than the later ones.


======================================================================
9. STEP 8 - THE KEY IDEA: TEMPLATE-BASED CANDIDATE SCORING GNN      (v3 family, zs_gnn_v3b_freq2.py ...)
======================================================================
Insight: the label IS the stimulus frequency/phase, so build the knowledge in.
Front end (fixed, not trained):
  1. For each (band, channel) signal project onto sin/cos templates at the 40 known stimulus
     frequencies x 4 harmonics with the known phases -> a matrix product (T x 320).
     These are fixed constants, NOT canonical correlation. There is NO CCA inside the network
     (state this explicitly in the paper; it is "template-based").
  2. For each candidate class k, the N electrode nodes get a feature vector (bands x harmonics x {cos,sin}).
  3. z-score each feature ACROSS the 40 candidates (the true class is a positive outlier).
     (A LayerNorm version removed the match magnitude and failed.)
  4. Shared graph layers (weights shared over candidates), ELU.
  5. Flatten nodes (keeps electrode identity) + learned class embedding -> shared MLP -> one score
     per candidate -> 40 logits.
About 22-23k parameters (dense19: 23,299; b5d19: 23,811).

   x (3 bands, N ch, T) --fixed sin/cos projection--> (40 cand, N nodes, 24 features)
        --z-score across candidates--> graph layer x2 --flatten + class embedding--> MLP --> 40 scores

RESULT (9 channels): no-graph model fg3 = 34.69 / 71.52. This is the reference for all ablations.
The model is 15x smaller than v2c yet far better at 1.0 s (71.5 vs 55-67).


======================================================================
10. STEP 9 - WHAT MAKES THE GRAPH WORK (ablations, 9 channels, 35 folds)
======================================================================
| variant | what it is                                       | 0.5 s | 1.0 s |
| fg3     | no graph                                         | 34.69 | 71.52 |
| fg2     | learned row-normalised adjacency, gated residual | 38.15 | 74.18 |
| fgu     | FIXED uniform adjacency + learned gate           | 38.06 | 73.93 |
| fga     | fixed anatomical inverse-distance adj. + gate    | 38.54 | 74.86 |
| cm      | fixed channel-mean subtraction (full strength)   | 32.08 | 68.50 |
| lr      | learned signed 9x9 map before each layer (init I)| 39.83 | 76.10 |
| knn9    | signed map restricted to 4 nearest electrodes    | 39.08 | 75.93 |
| v3c ng / h5 / ngh5 | per-node gate / 5 harmonics / both    | 37.30/38.02/37.25 | 74.21/74.69/74.01 |
| v3a pb  | per-band + layer-wise adjacency (conv front end) | 33.29 | 67.33 |
Key paired tests (nominal): fg2 vs fg3 p<0.001 (both windows); fgu vs fg3 p<0.001;
 lr vs fg2 p=0.009 / 0.001; fga vs fgu p=0.038 / <0.001 but fga is behind lr (p=0.013);
 v3a pb vs v2c-G2 p=0.127 / 0.491.
Mechanism: the fg2 gates are strongly negative (about -0.63 layer 1, -0.18 layer 2) -> a PARTIAL
spatial contrast. fgu recovers the gain of fg2, so the learned adjacency adds almost nothing
beyond mixing. A fixed full contrast (cm) HURTS; a learned signed map (lr) is best.
CONCLUSIONS you may write:
  - Learned signed spatial mixing before the graph layers is what helps; a free map is best.
  - kNN / anatomical priors, per-band graphs, per-node gates, 5 harmonics add nothing measurable.
  - The best model's "graph" is effectively a learned signed spatial filter (lr = one-layer GCN with
    a signed, unnormalised adjacency). Do NOT claim it learned electrode topology.


======================================================================
11. STEP 10 - MORE CHANNELS AND MORE CAPACITY
======================================================================
Input = bank A, same recipe. 19-channel (P/PO/O) and 64-channel runs use a free signed map
('dense') or a kNN map. Layer-2 width per node is 32/16/8 for 9/19/64 channels.
| model     | channels | params | 0.5 s | 1.0 s |
| lr        | 9        |        | 39.83 | 76.10 |
| dense19   | 19       | 23,299 | 41.92 | 77.54 |
| knn19     | 19       |        | 39.89 | 76.96 |
| wide19    | 19       | ~85k   | 41.55 | 77.17 |
| dense64   | 64       |        | 40.11 | not run (RAM) |
| knn64     | 64       |        | 39.69 | not run (RAM) |
Paired: dense19 vs lr p=0.001 (0.5 s), 0.081 (1.0 s); dense19 vs knn19 p=0.001 / 0.201; wide19 vs
dense19 p=0.340 / 0.389.
Conclusions: 19 channels give at most a small gain (+2.1 at 0.5 s; +1.4 at 1.0 s, per-seed p 0.04-0.22,
direction consistent over 3 seeds); 64 channels none; 4x more parameters nothing (so the model is not
under-fitting - an earlier claim to the contrary was wrong).
Report: 9-channel results are protocol-comparable with CCA-Net; 19/64-channel results go in a
separate table with same-channel baselines (CNN_ref at 19/64 channels is still to be run).


======================================================================
12. STEP 11 - TRIAL-ADAPTIVE CHANNEL WEIGHTS (negative result)       (zs_gnn_v3j_adapt.py)
======================================================================
Idea: weight nodes per trial: w = N * softmax(beta * s + bias), s = standardised per-trial in-band energy
per electrode (starts uniform).
RESULT:
| model | 0.5 s | 1.0 s | vs dense19/lr (paired p) |
| aw19  | 41.62 | 77.62 | vs dense19 seed 41: 0.118 / 0.483 (vs 3-seed mean: 0.308 / 0.808) |
| aw9   | 40.76 | 76.18 | vs lr: 0.030 (0.5 s, single-seed baseline) / 0.587 (1.0 s) -> one window only, treat as noise |
Diagnostic: beta stayed at 0.04-0.06 (aw19 mean 0.046, aw9 0.042) and |bias| about 0.06-0.09 (max 0.26),
i.e. the weights stayed within about +-5-10% of uniform. beta used only ~4% of the range Adam could
reach in 6,400 steps at lr 2e-4. So this feature gave no useful signal; do NOT write "adaptive weighting
cannot help". Write: "a simple energy-based weighting stayed near uniform and gave no gain".


======================================================================
13. STEP 12 - 5-BAND FILTER BANK AND THE SEED EXPERIMENT       (zs_gnn_v3l_bank5.py, zs_gnn_v3m_bank5seed.py)
======================================================================
Idea: use bank C (5 bands, up to 38 Hz lower edge) instead of bank A (3 bands) to see higher
harmonics separately (hypothesis, NOT tested as a mechanism).
Single seed: b5d9 = 40.99 / 76.92 (vs lr 39.83 / 76.10); b5d19 = 42.95 / 78.24 (vs dense19 41.92 / 77.54).
Pre-registered replication (fixed BEFORE running): 3 seeds of b5d19 vs 3 seeds of dense19, per-subject
3-seed means, accepted only if the bootstrap CI excludes 0 and the 1.0 s gain >= 0.5.
| seed        | dense19 0.5 s | b5d19 0.5 s | dense19 1.0 s | b5d19 1.0 s |
| 41          | 41.92 | 42.95 | 77.54 | 78.24 |
| 42          | 41.21 | 42.52 | 77.71 | 77.95 |
| 43          | 40.96 | 43.20 | 77.26 | 77.88 |
| mean (SD)   | 41.37 (0.49) | 42.89 (0.34) | 77.50 (0.23) | 78.02 (0.19) |
| difference  | +1.53, CI [0.83, 2.29], Wilcoxon p<0.001, Welch p=0.015 | | +0.52, CI [0.02, 1.05], p=0.045, Welch p=0.040 | |
Rule: PASS in both windows (1.0 s only just: 0.52 vs threshold 0.5). Every b5d19 seed beats every
dense19 seed in both windows.
Wording: "the 5-band bank gives a consistent small gain, about +1.5 at 0.5 s and +0.5 at 1.0 s; the
mechanism is untested". The single-seed gain at 1.0 s (+0.70) shrank to +0.52 when seeds were averaged.
Parameter counts: dense19 23,299; b5d19 23,811 (report them beside accuracy).


======================================================================
14. STEP 13 - NOISE FLOOR AND STATISTICAL RULES (important for honest claims)
======================================================================
  - Seed spread of the 35-subject mean: SD about 0.23 (1.0 s) and about 0.5 (0.5 s).
  - Per-subject SD of a seed-vs-seed difference: 2.2-2.6 points -> SE of a 35-subject mean difference
    between two runs of the SAME model is about 0.4. This is the right yardstick, not the range of seed means.
  - Two identical models (dense19 seeds 41 vs 43) give a paired p of 0.016 at 0.5 s. So a single-seed
    p between 0.001 and 0.05 is NOT evidence of a real difference.
  - Seed 41 was the highest of the 3 dense19 seeds at 0.5 s (winner's curse): compare against 3-seed means.
  - At 0.5 s, models within about 1.5-2 points cannot be ranked (lr, knn9, knn19, fga, fg2, fgu, v3c...).
    Large gaps survive: fg3 -> fg2 (+3.5) and the harm of the fixed contrast cm.
  - Bonferroni-style caution: about 85 nominal tests; borderline p (0.001-0.05) would not survive correction.
  - Same-seed pairing does NOT remove noise between different architectures (retracted earlier claim).


======================================================================
15. STEP 14 - SUBJECT-LEVEL ANALYSIS (1.0 s)
======================================================================
| method       | all   | experienced S01-08 | naive S09-35 | hard 4 (S11,S19,S29,S33) | other 31 |
| CCA-Net      | 82.48 | 91.3 | 79.9 | 25/52/59/47 (mean 45.7) | 87.22 |
| dense19      | 77.54 | 84.2 | 75.6 | 18/45/40/48 (mean 37.8) | 82.66 |
| b5d19        | 78.24 | 84.6 | 76.4 | 15/40/41/50 (mean 36.5) | 83.63 |
 - The four hard subjects are weak for EVERY method and their predictions are scattered, not biased to one class.
 - CCA-Net's lead over dense19 (4.94 points) is mostly in the OTHER 31 subjects (4.0 points; the hard four
   contribute 0.9). So the earlier claim "the gap is in the central hard subjects" was wrong.
 - To reach 86% overall with the hard four at CCA-Net level (about 46%), the other 31 would need about 91%.
 - [NOT RE-CHECKED] Spearman |latency - median| vs accuracy: dense19 -0.01, CCA-Net -0.14, FBtt-CCA -0.28
   (latency not explanatory); block phase-consistency vs accuracy +0.49 / +0.57 / +0.52; dense19 errors on
   typical subjects: 31% within 0.4 Hz (about 3x chance), 46% more than 1 Hz away.


======================================================================
16. STEP 15 - HYBRID: GNN LOGITS + CCA FEATURES           (zs_export_logits.py, zs_fuse.py, zs_fuse_ens.py)
======================================================================
Procedure
  - The v3 runner saves weights but not logits; zs_export_logits.py regenerates logits per fold from
    results/models/ on the same test data. Check: argmax(logits) equals the saved predictions in 100% of
    all folds for dense19, b5d19, b5d9 and lr, at both windows, and for the seed models.
  - Fusion rule (exactly CCA-Net's, nothing tuned): min-max normalise each score row across the 40 classes,
    then  fuse4 = net + (f2+f3)/2   and   full = net + (f1+f2+f3)/3.  Ensembles average the normalised logits
    of several networks first.
RESULT, 1.0 s (mean %):
| network (alone)               | + CCA fuse4 | + CCA full | fuse4 - (CNN+CCA), paired p |
| CCA only (f2+f3)              | 80.77       |            |                |
| CNN_ref (68.32)               | 82.31       | 83.39      | reference      |
| GNN v2c-G2 (66.98) [derived]  | about 83.2  | about 84.1 | +0.88 (p=0.787)|
| lr, 9 ch (76.10)              | 84.67       | 84.63      | +2.36 (p=0.218)|
| b5d9, 9 ch (76.92)            | 85.12       | 84.92      | +2.81 (p=0.056)|
| dense19 (77.54)               | 85.51       | 85.42      | +3.20 (p=0.003)|
| b5d19 (78.24)                 | 85.62       | 85.39      | +3.31 (p=0.009)|
| dense19 x 3 seeds (78.19)     | 85.67       | 85.49      | +3.36 (p=0.006)|
| b5d19 x 3 seeds (79.04)       | 85.92       | 85.80      | +3.61 (p=0.001)|
With the 'full' rule the differences vs CNN+CCA are NOT significant (p 0.23-0.90).
RESULT, 0.5 s (fuse4 / full): CCA only 43.96; CNN 51.65 / 51.86; lr 51.05 / 50.46; b5d9 51.74 / 50.75;
 dense19 52.57 / 52.10; b5d19 52.65 / 51.94; dense19 x3 52.65 / 51.98; b5d19 x3 53.17 / 52.38.
 No significant difference to CNN+CCA under fuse4 (p 0.20-0.99); under the 'full' rule the 9-channel models are slightly WORSE
 (b5d9 -1.1, p=0.035; lr -1.4, p=0.010).
Interpretation
  - At 1.0 s the network adds about +3.9 to +5.2 points over CCA alone with a GNN, versus +1.5 with the CNN.
    At 0.5 s every network adds 7-9 points and the GNN has no edge.
  - The fused number is stable at about 85.5-85.9 (ensembling adds only 0.2-0.3).
  - The hybrid is above the paper's CCA-Net (84.02) but 0.08 below 86%.
  - Hybrid caveat: the 19-channel rows pair a 19-channel GNN with 9-channel CCA features -> separate table.
    The 9-channel hybrids (lr, b5d9) are the protocol-comparable ones but have a single seed.
  - This is a HYBRID result and must be labelled so; the GNN-only result stays about 78-79%.


======================================================================
17. STEP 16 - RELATED WORK CHECKED: DDGCNN (Zhang et al., Neural Networks 172, 2024)
======================================================================
Status: reviewed; not run. What we verified (paper + the authors' code, commit 172c434; your DDGCNN-main
copy has identical checksums for model.py, layers.py, config.py, utils.py, train_DDGCNN.py, data_loaders.py):
  - It is an INTRA-INDIVIDUAL (within-subject) method. Cross-subject evaluation is listed as future work.
    Their 83.2% at 1.0 s on Benchmark is within-subject and NOT comparable to zero-shot numbers.
  - The released training script keeps the best validation accuracy over up to 8000 epochs (validation
    windows are random crops), and the paper mentions per-subject hyperparameter tuning -> optimistic.
  - Table 4 reports SEM values of +-19-22% for TRCA/ECCA/FBCCA at 1.0 s, which are impossible for
    accuracies of 35 subjects (maximum possible SEM is about 8.5); their ITR omits the 0.5 s gaze shift.
  - Ideas: layer-wise adjacency (covered by our v3a / fg2 tests: no gain), dense connections (our model
    is shallow and 4x capacity gave nothing), GDCF (acts on the feature axis, not across electrodes).
  - Open choices (ask before doing): (A) run DDGCNN as a baseline under our zero-shot protocol with our
    recipe; (B) test a trial-conditioned low-rank spatial mixing layer inspired by GDCF.


======================================================================
18. WHAT THE PAPER CAN CLAIM (and what it cannot)
======================================================================
CAN claim
  1. A template-based GNN with no CCA reaches 77-79% at 1.0 s zero-shot on Benchmark (9-19 channels),
     beating the CNN branch of CCA-Net by 8-10 points; it is not better than CCA-Net's fusion.
  2. Learned signed spatial mixing is the active ingredient; priors, depth, capacity and simple adaptivity
     add nothing measurable.
  3. 5-band bank: small consistent gain (+1.5 / +0.5), mechanism untested.
  4. GNN replacing CNN in the CCA fusion: +2.4 to +3.6 at 1.0 s (nominal p 0.001-0.2 depending on model/rule).
  5. Honest statistics: seed protocol, noise floor, nominal p-values and test count.
CANNOT claim
  - 86% for the GNN alone (about 78-79%); the hybrid reaches 85.92.
  - That the graph learned electrode topology; that anatomical priors help (the old anatomical result is invalid).
  - That adaptive weighting does not work in general; that DDGCNN works zero-shot (never tested).
  - Fine rankings among models within 1.5-2 points at 0.5 s.


======================================================================
19. FINAL RESULT SET FOR THE PAPER
======================================================================
Table A - protocol-comparable (9 channels), mean % +- SE (35 subjects)
| method                              | 0.5 s        | 1.0 s        |
| FBCCA                               | 20.64 +- 1.80 | 63.49 +- 3.53 |
| FBtt-CCA                            | 39.15 +- 3.21 | 77.71 +- 3.28 |
| CNN_ref (CNN branch)                | 41.54 +- 3.66 | 68.32 +- 3.96 |
| CCA-Net replication                 | 51.95 +- 3.54 | 82.48 +- 3.03 |
| GNN no graph (fg3)                  | 34.69 +- 3.18 | 71.52 +- 3.27 |
| GNN lr (learned signed map)         | 39.83 +- 3.35 | 76.10 +- 3.18 |
| GNN b5d9 (5-band bank)              | 40.99 +- 3.39 | 76.92 +- 3.08 |
Table B - 19 channels (separate)
| dense19 seed 41 / 42 / 43           | 41.92 / 41.21 / 40.96 | 77.54 / 77.71 / 77.26 |
| b5d19 seed 41 / 42 / 43             | 42.95 / 42.52 / 43.20 | 78.24 / 77.95 / 77.88 |
| 3-seed means (dense19 / b5d19)      | 41.37 / 42.89 | 77.50 / 78.02 |
| 3-seed network ensembles            | 42.13 / 44.02 | 78.19 / 79.04 |
Table C - hybrid GNN + CCA (label as hybrid): section 16 table.
Table D - ablation table: section 10 table (with paired p).
Generate all of these automatically:  python3 zs_consolidate.py  -> results/consolidated/report_tables.md,
all_models.csv, per_subject_acc.csv, seeds.csv, fusion.csv, subjects.csv, tests.csv, all_results.xlsx.


======================================================================
20. FIGURES AND POSTER LAYOUT
======================================================================
Figures to draw (source in brackets)
  F1 pipeline diagram: raw -> channels -> window -> filter bank -> z-score -> sin/cos projection -> graph layers -> scores
  F2 trial timeline and window position (readme.txt: cue 0.5 s, flicker 5 s)
  F3 bar chart accuracy at 0.5/1.0 s for baselines and best GNNs with SE bars (all_models.csv)
  F4 ablation bars with 95% CI of the paired difference (tests.csv)
  F5 seed scatter: 3 seeds of dense19 and b5d19 per window (seeds.csv)
  F6 per-subject scatter GNN vs CCA-Net, hard subjects highlighted (per_subject_acc.csv; heat_subjects.py)
  F7 hybrid bars: CCA only / CNN+CCA / GNN+CCA / ensemble+CCA (fusion.csv)
  F8 learned 9x9 map of lr as a heatmap with electrode names (results JSON 'graph'); caption: "a signed spatial filter"
Poster (landscape, 4 columns): 1 Problem + dataset + protocol | 2 Method (F1, F2) | 3 Results (F3, F4, F5) |
4 Hybrid + subject analysis (F6, F7) + take-aways + limitations + QR to code.
Take-aways box: "GNN without CCA = 78% (1.0 s); signed spatial mixing is what matters; GNN beats CCA-Net's CNN
branch by 8-10 points; hybrid 85.9%".


======================================================================
21. LIMITATIONS AND THREATS TO VALIDITY (write these in the paper)
======================================================================
  - One dataset (35 subjects); all model selection and about 85 nominal tests used the same subjects ->
    results are optimistic; a confirmatory run on new subjects (BETA) with the configuration frozen is the
    strongest next evidence (BETA access: unanswered).
  - Single seed for most rows; only dense19 and b5d19 have 3 seeds.
  - 19/64-channel rows are not protocol-comparable; CNN_ref at 19/64 channels is still missing.
  - Hybrid rows: fixed fusion rule but the choice of which networks/rules to report was made after seeing results.
  - The replication of CCA-Net is 1.5-2.3 points below the paper.
  - dense64/knn64 at 1.0 s not run (WSL RAM).
  - The mechanism behind the 5-band gain and the CCA-complementarity is not tested.


======================================================================
22. RETRACTIONS AND CORRECTIONS (keep this list; do not repeat these claims)
======================================================================
  1. v2d "anatomical" result (62.77 at 1.0 s) is INVALID: the loader treated EEGLAB polar (theta, radius) as x,y.
     Do not use it as evidence that anatomical graphs are worse. The correct anatomical test is fga (74.86 / 38.54).
  2. "Underfitting, so capacity helps": wrong (wide19 no gain).
  3. "The 4 hard subjects are central to the gap": wrong (about 82% of CCA-Net's lead is in the other 31).
  4. "Latency/phase explains the errors": not supported.
  5. "Noise floor = range of seed means (0.45)": too lenient; use the SE of a paired difference (about 0.4).
  6. "Same-seed pairing removes noise": wrong for different architectures; use 3-vs-3 seed means.
  7. "No GNN+CCA fusion exists": the old v2c-G2 logits could be fused (about 83.2 / 84.1); the strong v3
     models had no logits until zs_export_logits.py.
  8. Fine rankings at 0.5 s are not reliable (identical-model p = 0.016).


======================================================================
23. REPRODUCIBILITY - ORDER OF COMMANDS (run from ZERO_SHOT/, venv active, long jobs in screen)
======================================================================
  python3 test_stage1.py                                   # stage-1 checks (see its docstring for options)
  python3 zs_runner.py --sl 0.5 1.0 --methods fbcca fbttcca_yf
  python3 zs_cnn_ref.py --sl 0.5 1.0 --resume
  python3 zs_cca_net.py --sl 0.5 1.0 --resume --device cuda
  python3 zs_cca_feats.py --sl 0.5 1.0 --resume            # CCA features (CPU)
  python3 zs_gnn_v2c.py --sl 0.5 1.0 --variant g2 g3 --resume
  python3 zs_gnn_v3b_freq2.py --sl 0.5 1.0 --variant fg2 fg3 --resume
  (then v3c, v3d, v3e, v3g, v3h, v3i, v3j, v3k, v3l: each takes --sl 0.5 1.0 --variant <names> --resume)
  bash run_fill.sh                                         # fills the 0.5 s column of the current family
  bash run_b5seed.sh                                       # b5d19 seeds 42/43 at both windows
  python3 zs_export_logits.py zs_gnn_v3g_graph dense19 --sl 0.5 1.0   # and the other variants/seeds
  python3 zs_fuse.py --tag cnnref --sl 0.5 1.0 --vs gnnv3g_dense19
  python3 zs_fuse_ens.py --nets gnnv3l_b5d19 gnnv3m_s42b5d19 gnnv3m_s43b5d19 --ref cnnref
  python3 zs_b5verdict.py                                  # 3-vs-3 seed test
  python3 compile_results.py --ref gnnv3g_dense19          # per-window tables with paired Wilcoxon
  python3 zs_consolidate.py                                # everything into results/consolidated/
Rules: never edit a bash script while it runs; one GPU job at a time; Windows sleep off, one WSL window open;
the channel set is read from the TRAILING DIGITS of v3 variant names (9, 19 or 64); never overwrite old results.


======================================================================
24. PENDING WORK (ask before doing any of it)
======================================================================
  1. Run zs_consolidate.py and check the generated tables against this file.
  2. DDGCNN: option A (baseline under our protocol) / option B (trial-conditioned mixing) / neither.
  3. BETA access for a confirmatory run (frozen configuration).
  4. Seeds for the 9-channel hybrids (b5d9, lr) so the protocol-comparable hybrid row has a noise floor.
  5. CNN_ref at 19 (and 64) channels for the 19-channel table.
  6. 64 channels at 1.0 s (needs more WSL RAM or a lighter loader).
  7. Write the paper: methods from sections 1-5 and 9, results from sections 6-16, limitations from 21.
