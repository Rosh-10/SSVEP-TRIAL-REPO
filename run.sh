#!/usr/bin/env bash
#
# FIRST_RUNS/run.sh

set -e  # stop immediately if any script fails
set -u  # treat unset variables as errors

# Detect Python binary
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Error: Python is not installed."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Preprocessing
(
    cd "$SCRIPT_DIR/preprocessing"
    "$PYTHON" verify_data.py
    "$PYTHON" test_mat_loader.py
    "$PYTHON" step6a_fft_check.py
    "$PYTHON" step6b_load_freqphase.py
    "$PYTHON" step6b_inspect_freqphase.py
    "$PYTHON" step6c_harmonic_check.py
    "$PYTHON" step6d_diagnose_outlier_ratio.py
    "$PYTHON" step6e_harmonic_check.py
    "$PYTHON" step6f.py
    "$PYTHON" step6g.py
    "$PYTHON" step7a.py
    "$PYTHON" step7b.py
    "$PYTHON" step7c.py
    "$PYTHON" step7d.py
    "$PYTHON" step8a_load_filter_design.py
    "$PYTHON" step8b_preprocess_trial.py
    "$PYTHON" step8c_batch_process.py
    "$PYTHON" step8d_validate.py
)

# Feature Extraction
(
    cd "$SCRIPT_DIR/feature_extraction"
    "$PYTHON" step9a_filter_design.py
    "$PYTHON" step9b_batch_extract.py
    "$PYTHON" step9c_validate.py
    "$PYTHON" step9d_spectrum_features.py
    "$PYTHON" test_accuracy_spectrum.py
    "$PYTHON" test_accuracy_spectrum_v2.py
    "$PYTHON" step9e_spectrum_5s.py
    "$PYTHON" step9e_check.py
)

# Graph Construction & Training
(
    cd "$SCRIPT_DIR"
    mkdir -p "$ROOT_DIR/results"

    "$PYTHON" graph_construction/step10_graph_construction_fixed.py
    "$PYTHON" training1/step11b_baseline.py $(seq 1 35) \vert{} tee "$ROOT_DIR/results/11b_baseline.txt"
    "$PYTHON" training1/step11c_harmonic_rule.py $(seq 1 35) \vert{} tee "$ROOT_DIR/results/11c_harmonic.txtYou ran `bash FIRST_RUNS/run.sh`[cite: 1], which means the script stored inside the **`FIRST_RUNS`** directory still contains the broken block comment syntax (`: << "Skip:"` ... `Skip`) and unescaped parentheses[cite: 1]. 

To fix this, update `FIRST_RUNS/run.sh` with the corrected implementation.

### Updated `FIRST_RUNS/run.sh`

Replace the contents of `FIRST_RUNS/run.sh` with this code:

```bash
#!/usr/bin/env bash
#
# FIRST_RUNS/run.sh
# Full execution script for SSVEP-GNN preprocessing, feature extraction, graph construction, and training.

set -e  # stop immediately if any command fails
set -u  # treat unset variables as errors

# Detect active Python environment
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Error: Python is not installed."
    exit 1
fi

# Define project directories relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PREPROCESSING_DIR="$SCRIPT_DIR/preprocessing"
FEATURE_DIR="$SCRIPT_DIR/feature_extraction"
GRAPH_DIR="$SCRIPT_DIR/graph_construction"
TRAINING_DIR="$SCRIPT_DIR/training1"

echo "Running full pipeline from: $SCRIPT_DIR"

# ==============================================================================
# PHASE 1: PREPROCESSING
# ==============================================================================
(
    cd "$PREPROCESSING_DIR"

    echo "================================================================"
    echo "Step 0a: verify_data.py"
    echo "================================================================"
    "$PYTHON" verify_data.py

    echo "================================================================"
    echo "Step 0b: test_mat_loader.py"
    echo "================================================================"
    "$PYTHON" test_mat_loader.py

    echo "================================================================"
    echo "Step 6A: step6a_fft_check.py"
    echo "================================================================"
    "$PYTHON" step6a_fft_check.py

    echo "================================================================"
    echo "Step 6B load: step6b_load_freqphase.py"
    echo "================================================================"
    "$PYTHON" step6b_load_freqphase.py

    echo "================================================================"
    echo "Step 6B inspect: step6b_inspect_freqphase.py"
    echo "================================================================"
    "$PYTHON" step6b_inspect_freqphase.py

    echo "================================================================"
    echo "Step 6C: step6c_harmonic_check.py"
    echo "================================================================"
    "$PYTHON" step6c_harmonic_check.py

    echo "================================================================"
    echo "Step 6D: step6d_diagnose_outlier_ratio.py"
    echo "================================================================"
    "$PYTHON" step6d_diagnose_outlier_ratio.py

    echo "================================================================"
    echo "Step 6E: step6e_harmonic_check.py"
    echo "================================================================"
    "$PYTHON" step6e_harmonic_check.py

    echo "================================================================"
    echo "Step 6F: step6f.py"
    echo "================================================================"
    "$PYTHON" step6f.py

    echo "================================================================"
    echo "Step 6G: step6g.py"
    echo "================================================================"
    "$PYTHON" step6g.py

    echo "================================================================"
    echo "Step 7A: step7a.py"
    echo "================================================================"
    "$PYTHON" step7a.py

    echo "================================================================"
    echo "Step 7B: step7b.py"
    echo "================================================================"
    "$PYTHON" step7b.py

    echo "================================================================"
    echo "Step 7C: step7c.py"
    echo "================================================================"
    "$PYTHON" step7c.py

    echo "================================================================"
    echo "Step 7D: step7d.py"
    echo "================================================================"
    "$PYTHON" step7d.py

    echo "================================================================"
    echo "Step 8A: Load Freq_Phase, design filter, cache"
    echo "================================================================"
    "$PYTHON" step8a_load_filter_design.py

    echo "================================================================"
    echo "Step 8B: Preprocess trial"
    echo "================================================================"
    "$PYTHON" step8b_preprocess_trial.py

    echo "================================================================"
    echo "Step 8C: Batch preprocess all 35 subjects"
    echo "================================================================"
    "$PYTHON" step8c_batch_process.py

    echo "================================================================"
    echo "Step 8D: Validate preprocessed dataset"
    echo "================================================================"
    "$PYTHON" step8d_validate.py
)

# ==============================================================================
# PHASE 2: FEATURE EXTRACTION
# ==============================================================================
(
    cd "$FEATURE_DIR"

    echo "================================================================"
    echo "STEP 9a: FILTER BANK FEATURE EXTRACTION"
    echo "================================================================"
    "$PYTHON" step9a_filter_design.py

    echo "================================================================"
    echo "STEP 9b: Batch extraction"
    echo "================================================================"
    "$PYTHON" step9b_batch_extract.py

    echo "================================================================"
    echo "STEP 9c: Validation"
    echo "================================================================"
    "$PYTHON" step9c_validate.py

    echo "================================================================"
    echo "STEP 9d: Spectrum features"
    echo "================================================================"
    "$PYTHON" step9d_spectrum_features.py

    echo "================================================================"
    echo "Test accuracy"
    echo "================================================================"
    "$PYTHON" test_accuracy_spectrum.py

    echo "================================================================"
    echo "Test accuracy v2"
    echo "================================================================"
    "$PYTHON" test_accuracy_spectrum_v2.py

    echo "================================================================"
    echo "STEP 9e: 5s spectrum"
    echo "================================================================"
    "$PYTHON" step9e_spectrum_5s.py

    echo "================================================================"
    echo "STEP 9e check"
    echo "================================================================"
    "$PYTHON" step9e_check.py
)

# ==============================================================================
# PHASE 3: GRAPH CONSTRUCTION & TRAINING
# ==============================================================================
mkdir -p "$ROOT_DIR/results"

echo "========================================================================"
echo "Graph construction"
echo "========================================================================"
"$PYTHON" "$GRAPH_DIR/step10_graph_construction_fixed.py"

echo "========================================================================"
echo "STEP 11b: logistic-regression baseline"
echo "========================================================================"
"$PYTHON" "$TRAINING_DIR/step11b_baseline.py" $(seq 1 35) | tee "$ROOT_DIR/results/11b_baseline.txt"

echo "========================================================================"
echo "STEP 11c: harmonic rule"
echo "========================================================================"
"$PYTHON" "$TRAINING_DIR/step11c_harmonic_rule.py" $(seq 1 35) | tee "$ROOT_DIR/results/11c_harmonic.txt"

echo "========================================================================"
echo "STEP 11d: GCN, occipital-10"
echo "========================================================================"
"$PYTHON" "$TRAINING_DIR/step11d_gcn_edited.py" --occ $(seq 1 35) | tee "$ROOT_DIR/results/11d_occ_gcn_all35.txt"

echo "========================================================================"
echo "STEP 11d: no-graph ablation, occipital-10"
echo "========================================================================"
"$PYTHON" "$TRAINING_DIR/step11d_gcn_edited.py" --occ --noadj $(seq 1 35) | tee "$ROOT_DIR/results/11d_occ_nograph_all35.txt"

echo "========================================================================"
echo "FULL PIPELINE COMPLETED SUCCESSFULLY"
echo "========================================================================"