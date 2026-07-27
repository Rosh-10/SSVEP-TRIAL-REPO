# SSVEP-TRIAL-REPO

A GNN-based classifier for 40-target SSVEP EEG frequency classification, built on the Benchmark SSVEP Dataset (Wang, Chen, Gao, Gao, 2016, IEEE TNSRE).

**Research goal:** Replace the standard DNN classifier in an SSVEP pipeline with a Graph Neural Network that explicitly models spatial relationships between EEG electrodes, and test whether this improves 40-target classification accuracy (8.0–15.8 Hz targets).

This is a two-phase project:
- **Phase 1:** Build a working GNN baseline for standard SSVEP classification, benchmarked against literature (CCA, FBCCA, DNN, Ensemble DNN, FBtt-CCA, SSVEPformer, CCA-Net).
- **Phase 2:** Identify genuine novelty from Phase 1 results and failure modes (direction not yet decided — intentionally deferred until Phase 1 results are in).

---

## 1. Dataset setup (required before running anything)

The raw dataset is **not included in this repo** (excluded via `.gitignore` — 35 subject files, too large to version-control, and redistribution terms are unclear).

You need to download the **Benchmark SSVEP Dataset** yourself and place the files as follows:

```
SSVEP-TRIAL-REPO/
└── data/
    └── raw/
        └── benchmark/
            ├── S01.mat
            ├── S02.mat
            ├── ...
            ├── S35.mat
            ├── Freq_Phase.mat
            └── 64-channels.loc
```

- `S01.mat` – `S35.mat`: one file per subject, each a `[64 channels, 1500 time points, 40 targets, 6 blocks]` array.
- `Freq_Phase.mat`: stimulation frequency (Hz) and phase (radians) for each of the 40 targets.
- `64-channels.loc`: electrode position/layout file.

If your `DATA_ROOT` differs from the default, update it in `config.py` — this is the single place path configuration lives, so the rest of the codebase works unmodified across machines (e.g. your Windows setup vs a teammate's).

> **Note:** the BETA dataset (Liu et al., 2020, 70 subjects) is a planned future addition — not used in the current pipeline.

---

## 2. Environment setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

*(If `requirements.txt` doesn't exist yet, generate one with `pip freeze > requirements.txt` once your environment is finalized.)*

---

## 3. Verifying your setup

Before running anything else, confirm all data files are present and readable:

```bash
python preprocessing/verify_data.py
```

This checks that all 35 subject files plus `Freq_Phase.mat` and `64-channels.loc` are present and loadable via `scipy.io.loadmat`.

---

## 4. Run order (current pipeline)

Scripts should be run **in this order** — later steps depend on earlier ones (either directly via imports, or because they build understanding/parameters used in later design decisions):

| Step | Script | Purpose |
|---|---|---|
| 1 | `preprocessing/verify_data.py` | Confirms all data files present |
| 2 | `preprocessing/data_loading.py` | Defines `load_subject()`, `extract_trial()` — imported by later scripts, not run standalone |
| 3 | `preprocessing/step6a_fft_check.py` | Empirical FFT check (Subject 1, Target 1, Block 1, channel Oz) — confirms signal at 8 Hz + harmonics, confirms notch filter worked |
| 4 | `preprocessing/step6b_load_freqphase.py` | Loads `Freq_Phase.mat`, returns clean `freqs`/`phases` arrays (40,) — imported by later scripts |
| 5 | `preprocessing/step6c_harmonic_check.py` | Computes 2nd-harmonic-to-fundamental ratio across all 40 targets, for 3 subjects (S01, S15, S30) |
| 6 | `preprocessing/step6d_diagnose_outlier.py` | Diagnoses the harmonic-ratio outlier found in Step 6c (Subject 30, Target 9) using raw magnitudes |

**Note on MATLAB scripts:** Steps 1–4 of the original exploration were done in MATLAB (kept in `EXPLORATION-USING-MATLAB/`, untouched, exploration-only — not part of the Python pipeline and not required to run it).

---

## 5. Project status / progress log

### Completed
- **Data infrastructure:** `config.py` (single editable `DATA_ROOT`), `verify_data.py`, `data_loading.py` with explicit MATLAB 1-indexed → Python 0-indexed conversion.
- **Step 6a — FFT sanity check:** Confirmed for Subject 1/Target 1/Block 1/Oz: dominant peak at 8.0 Hz (fundamental), strong 2nd harmonic at 16.0 Hz, weaker 3rd harmonic at 24.0 Hz, negligible 50 Hz powerline noise (hardware notch filter confirmed effective — no software notch needed).
- **Step 6b — Frequency/phase loader:** `load_freq_phase()` loads and flattens `Freq_Phase.mat`'s `(1,40)` MATLAB row-vectors into clean `(40,)` NumPy arrays. Confirmed target indices follow **keyboard layout position, not sorted frequency** — always use `freqs[target_index - 1]`, never assume ordering.
- **Step 6c — Harmonic ratio analysis (all 40 targets, 3 subjects: S01, S15, S30):** Ratios range ~0.14–2.85 with no clean frequency-dependent trend. Key finding: **harmonics are not reliably weaker than the fundamental** — they can rival or exceed it. This directly informs the bandpass filter design (harmonics must be preserved, not filtered out).
- **Step 6d — Outlier diagnosis:** Subject 30's high ratio (2.851, Target 9) confirmed as a genuine strong-harmonic response (both fundamental and harmonic magnitudes were substantial, not a noisy/weak-fundamental artifact) — not a bug, a real inter-subject difference (n=3, not yet generalized to the full 35).

### Key learnings
- Harmonics carry substantial, sometimes dominant, signal power — filter design must preserve them.
- No software notch filter needed; recording-time hardware notch confirmed effective.
- Target indices follow keyboard layout, not frequency order — always look up via `freqs[target_index - 1]`.
- Phase values follow a clean `0, π/2, π, 3π/2` pattern — potentially relevant for future phase-informed decoding (e.g. eCCA).
- Numeric confirmation caught a visual misread of the FFT plot — always verify peak claims numerically, not just by eye.
- Strict one-responsibility-per-file discipline enforced after a circular import bug (Step 6c code accidentally pasted into the Step 6b file).

### Next immediate step
- **Bandpass filter design** (Step 7): cutoff frequency justification (informed by the harmonic findings above), Butterworth vs. FIR trade-offs, zero-phase `filtfilt` filtering.

### On the horizon
- 6-block-averaged FFT magnitude spectra (decided: average magnitudes across blocks, not raw signals, to avoid phase-alignment issues)
- Epoching and normalization
- CCA/FFT feature extraction
- Graph construction for the GNN
- GNN implementation, training, evaluation
- Dedicated GNN + SSVEP literature search before committing to a Phase 2 novelty direction

---

## 6. Repo structure

```
SSVEP-TRIAL-REPO/
├── data/
│   ├── raw/benchmark/       # gitignored — see Section 1 for setup
│   └── processed/
├── preprocessing/
├── models/
├── training/
├── evaluation/
├── notebooks/
├── presentation/
├── report/
├── results/
├── EXPLORATION-USING-MATLAB/ # MATLAB exploration only, not part of the Python pipeline
├── config.py
└── README.md
```

---

## 7. Reference literature

- Wang, Y., Chen, X., Gao, X., & Gao, S. (2016). A benchmark dataset for SSVEP-based brain-computer interfaces. *IEEE TNSRE*.
- Chen, X. et al. (2015). Filter bank canonical correlation analysis (FBCCA) for SSVEP.
- Deng, et al. CCA-Net (zero-shot cross-subject transfer learning). *IEEE TIM*. Used as a reference benchmark only — not the pipeline this project reproduces.