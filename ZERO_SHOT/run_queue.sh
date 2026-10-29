#!/usr/bin/env bash
# run_queue.sh -- runs every remaining experiment one after another (never two on the GPU at once).
# Run from ZERO_SHOT/ inside a screen session:   bash run_queue.sh
# Safe to stop and restart: every job uses --resume, so finished folds are skipped.
# Each job's output goes to logs/<name>.log ; a failed job does not stop the queue.
cd "$(dirname "$0")"
mkdir -p logs

python3 zs_gnn_v3c_freq3.py --smoke || { echo "freq3 smoke test FAILED - stopping"; exit 1; }

run () {   # run <logname> <command...>
  name=$1; shift
  echo "[$(date '+%F %T')] START $name"
  "$@" 2>&1 | tee -a "logs/$name.log"
  echo "[$(date '+%F %T')] END   $name (exit ${PIPESTATUS[0]})"
}

# 1. new frequency-aware variants (most likely to improve the best model)      ~3 h
run freq3        python3 zs_gnn_v3c_freq3.py --sl 0.5 1.0 --variant ng h5 ngh5 --resume
# 2. v3a per-band at 0.5 s, then shared-adjacency ablation at both windows     ~3 h + ~9 h
run v3a_pb_05    python3 zs_gnn_v3a_band.py --sl 0.5 --variant pb --resume
run v3a_sh       python3 zs_gnn_v3a_band.py --sl 0.5 1.0 --variant sh --resume
# 3. v2d graph ablation at 0.5 s                                                ~6 h
run v2d_05       python3 zs_gnn_v2d.py --sl 0.5 --variant g2_anatomical g2_identity g2_uniform --resume --device cuda
# 4. v2e per-band (shared weights), all 35 subjects, both windows               ~11 h
run v2e          python3 zs_gnn_v2e.py --sl 1.0 0.5 --device cuda --resume

echo "[$(date '+%F %T')] ALL DONE"
