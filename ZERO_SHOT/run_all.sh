#!/bin/bash
cd "$(dirname "$0")"; mkdir -p logs
exec > >(tee -a logs/run_all.log) 2>&1
if pgrep -f "python3 zs_gnn" > /dev/null; then echo "a zs_gnn job is already running; stop it first"; exit 1; fi

job() {   # job <script> <log> <args...>: smoke-test first, skip the job if the test fails
  s=$1; log=$2; shift 2
  if python3 "$s" --smoke > "logs/smoke_${s%.py}.log" 2>&1; then
    echo "$(date +%T) start $s $*"; python3 "$s" "$@" > "$log" 2>&1; echo "$(date +%T) end   $s"
  else echo "$(date +%T) SMOKE FAILED: $s skipped (see logs/smoke_${s%.py}.log)"; fi
}

python3 zs_gnn_v3l_bank5.py --cache > logs/cache_bank5.log 2>&1 &    # CPU only, runs next to the first jobs
CP=$!

job zs_gnn_v3k_seed.py  logs/v3k_42.log --sl 1.0 --variant s42d19 --resume
job zs_gnn_v3j_adapt.py logs/v3j.log    --sl 1.0 --variant aw19 aw9 --resume

wait $CP
if python3 -c "import numpy as np, zs_gnn_v3l_bank5 as L, zs_gnn_v3g_graph as G, zs_data64 as D64; a=L.prepC(1,1.0,G.chans(19))['A'][:,:3]; b=D64.prep64(1,1.0,G.chans(19))['A']; d=np.abs(a-b).max(); print('bank C bands 1-3 vs bank A, max diff', d); assert d < 1e-4"; then
  job zs_gnn_v3l_bank5.py logs/v3l.log --sl 1.0 --variant b5d19 b5d9 --resume
else echo "$(date +%T) bank C bands 1-3 differ from bank A: v3l skipped"; fi

job zs_gnn_v3k_seed.py  logs/v3k_43.log --sl 1.0 --variant s43d19 --resume
echo "$(date +%T) ALL DONE"
python3 compile_results.py --ref gnnv3g_dense19 > logs/compile_dense19.txt 2>&1
python3 compile_results.py --ref gnnv3e_lr > logs/compile_lr.txt 2>&1
