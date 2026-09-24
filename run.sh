#!/usr/bin/env bash
#
# run_pipeline.sh
# Runs the SSVEP-GNN preprocessing pipeline scripts in order, through Step 6D
# (the last step actually completed — 6E and 6F are NOT yet written/run).
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# Run this from the repo root, or edit PREPROCESSING_DIR below if your
# layout differs.

set -e  # stop immediately if any script fails
set -u  # treat unset variables as errors

PREPROCESSING_DIR="./preprocessing"

cd "$PREPROCESSING_DIR/"
echo $(ls)
echo "================================================================"
echo "Step 0a: verify_data.py"
echo "================================================================"
python verify_data.py

echo "================================================================"
echo "Step 0b: test_mat_loader.py"
echo "================================================================"
python test_mat_loader.py

echo "================================================================"
echo "Step 6A: step6a_fft_check.py"
echo "================================================================"
python step6a_fft_check.py

echo "================================================================"
echo "Step 6B (load): step6b_load_freqphase.py"
echo "================================================================"
python step6b_load_freqphase.py

echo "================================================================"
echo "Step 6B (inspect): step6b_inspect_freqphase.py"
echo "================================================================"
python step6b_inspect_freqphase.py

echo "================================================================"
echo "Step 6C: step6c_harmonic_check.py"
echo "================================================================"
python step6c_harmonic_check.py

echo "================================================================"
echo "Step 6D: step6d_diagnose_outlier_ratio.py"
echo "================================================================"
python step6d_diagnose_outlier_ratio.py

echo "================================================================"
echo "PIPELINE COMPLETE — through Step 6D"

echo "================================================================"
echo "Step 6E: step6e_harmonic_check.py"
echo "================================================================"
python step6e_harmonic_check.py

echo "================================================================"
echo "Step 6F: step6f.py"
echo "================================================================"
python step6f.py

echo "================================================================"
echo "Step 6G: step6g.py"
echo "================================================================"
python step6g.py

echo "================================================================"
echo "Step 7A: step7a.py"
echo "================================================================"
python step7a.py

echo "================================================================"
echo "Step 7B: step7b.py"
echo "================================================================"
python step7b.py

echo "================================================================"
echo "Step 7C: step7c.py"
echo "================================================================"
python step7c.py

echo "================================================================"
echo "Step 7D: step7d.py"
echo "================================================================"
python step7d.py





echo "========================================================================"
echo "Step 8A: Load Freq_Phase, design filter, cache"
echo "========================================================================"
python step8a_load_filter_design.py
echo ""
 
echo "========================================================================"
echo "Step 8C: Batch preprocess all 35 subjects"
echo "This may take 2–5 minutes depending on I/O speed."
echo "========================================================================"
python step8c_batch_process.py
echo ""
 
echo "========================================================================"
echo "Step 8D: Validate preprocessed dataset"
echo "========================================================================"
python step8d_validate.py
echo ""
 