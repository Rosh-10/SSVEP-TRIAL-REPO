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
 
 echo "============================================================b============"
echo "Step 8b "
echo "========================================================================"
python step8b_preprocess_trial.py
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

#


cd ..

echo $(ls)
cd feature_extraction

echo "========================================================================"
echo "STEP 9a: FILTER BANK FEATURE EXTRACTION - SINGLE TRIAL TEST"
echo "========================================================================"
python step9a_filter_design.py

echo ""

echo "========================================================================"
echo "STEP 9b:Batch_extraction"
echo "========================================================================"
python step9b_batch_extract.py


echo ""

echo "========================================================================"
echo "STEP 9c:Validation"
echo "========================================================================"
python step9c_validate.py


echo ""

echo "========================================================================"
echo "STEP 9d: Spectrum features"
echo "========================================================================"
python step9d_spectrum_features.py

echo ""


echo "========================================================================"
echo "Test accuracy"
echo "========================================================================"
python test_accuracy_spectrum.py

echo ""


echo "========================================================================"
echo "Test accuracy v2"
echo "========================================================================"
python test_accuracy_spectrum_v2.py

echo ""

echo "========================================================================"
echo "STEP 9e: exact 5 s spectrum (used by Step 11)"
echo "========================================================================"
python step9e_spectrum_5s.py

echo ""

echo "========================================================================"
echo "STEP 9e check"
echo "========================================================================"
python step9e_check.py

echo ""

cd ..

echo "========================================================================"
echo "Graph construction"
echo "========================================================================"
python graph_construction/step10_graph_construction_fixed.py 

echo ""

echo "========================================================================"
echo "STEP 11b: logistic-regression baseline"
echo "========================================================================"
python training1/step11b_baseline.py $(seq 1 35) | tee results/11b_baseline.txt

echo ""

echo "========================================================================"
echo "STEP 11c: harmonic rule"
echo "========================================================================"
python training1/step11c_harmonic_rule.py $(seq 1 35) | tee results/11c_harmonic.txt

echo ""

echo "========================================================================"
echo "STEP 11d: GCN, occipital-10"
echo "========================================================================"
python training1/step11d_gcn_edited.py --occ $(seq 1 35) | tee results/11d_occ_gcn_all35.txt

echo ""

echo "========================================================================"
echo "STEP 11d: no-graph ablation, occipital-10"
echo "========================================================================"
python training1/step11d_gcn_edited.py --occ --noadj $(seq 1 35) | tee results/11d_occ_nograph_all35.txt

echo ""