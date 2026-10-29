#!/bin/bash
# run_fill.sh -- fill the gaps in the results table. Every job uses --resume, so finished folds are skipped.
# Usage (inside screen, from ZERO_SHOT/):  bash run_fill.sh [--64] [--legacy]
#   default   : 0.5 s column for the current family (seeds, bank-5, adaptive weights, wide19)
#   --64      : dense64/knn64 at 1.0 s (needs >= 10 GB WSL RAM, skipped otherwise)
#   --legacy  : v2d identity/uniform at 0.5 s (anatomical excluded: its adjacency code was wrong)
# Never edit this file while it runs.
cd "$(dirname "$0")"; mkdir -p logs
exec > >(tee -a logs/run_fill.log) 2>&1
if pgrep -f "python3 zs_gnn" > /dev/null; then echo "a zs_gnn job is already running; stop it first"; exit 1; fi
W64=0; LEG=0; for a in "$@"; do [ "$a" = "--64" ] && W64=1; [ "$a" = "--legacy" ] && LEG=1; done

job() {   # job <script> <log> <args...>: smoke-test first, skip the job if the test fails
  s=$1; log=$2; shift 2
  if python3 "$s" --smoke > "logs/smoke_${s%.py}.log" 2>&1; then
    echo "$(date +%T) start $s $*"; python3 "$s" "$@" > "$log" 2>&1; echo "$(date +%T) end   $s"
  else echo "$(date +%T) SMOKE FAILED: $s skipped (see logs/smoke_${s%.py}.log)"; fi
}

# A. 0.5 s column of the current family (same order of value: noise floor first)
job zs_gnn_v3k_seed.py  logs/f_v3k.log  --sl 0.5 --variant s42d19 s43d19 --resume
job zs_gnn_v3l_bank5.py logs/f_v3l.log  --sl 0.5 --variant b5d19 b5d9 --resume
job zs_gnn_v3j_adapt.py logs/f_v3j.log  --sl 0.5 --variant aw19 aw9 --resume
job zs_gnn_v3i_cap.py   logs/f_v3i.log  --sl 0.5 --variant wide19 --resume

# B. 64 channels at 1.0 s (RAM-limited, see header)
if [ $W64 = 1 ]; then
  mem=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
  if [ "$mem" -ge 10 ]; then job zs_gnn_v3g_graph.py logs/f_v3g64.log --sl 1.0 --variant knn64 dense64 --resume
  else echo "$(date +%T) only ${mem} GB RAM visible: 64-ch 1.0 s skipped (set memory=11GB in .wslconfig, wsl --shutdown)"; fi
fi

# C. legacy completeness
[ $LEG = 1 ] && job zs_gnn_v2d.py logs/f_v2d.log --sl 0.5 --variant g2_identity g2_uniform --resume --device cuda

echo "$(date +%T) ALL DONE"
python3 zs_table.py > logs/all_models.txt 2>&1
python3 compile_results.py --ref gnnv3g_dense19 > logs/compile_dense19.txt 2>&1
python3 compile_results.py --ref gnnv3e_lr > logs/compile_lr.txt 2>&1
