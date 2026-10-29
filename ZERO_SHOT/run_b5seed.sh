#!/bin/bash
# run_b5seed.sh -- b5d19 seeds 42 and 43 (1.0 s first = decision window, then 0.5 s). ~2.5 h. Never edit while running.
cd "$(dirname "$0")"; mkdir -p logs
exec > >(tee -a logs/run_b5seed.log) 2>&1
if pgrep -f "python3 zs_gnn" > /dev/null; then echo "a zs_gnn job is already running"; exit 1; fi
if python3 zs_gnn_v3m_bank5seed.py --smoke > logs/smoke_v3m.log 2>&1; then
  echo "$(date +%T) start 1.0 s"; python3 zs_gnn_v3m_bank5seed.py --sl 1.0 --variant s42b5d19 s43b5d19 --resume > logs/f_v3m_10.log 2>&1
  echo "$(date +%T) start 0.5 s"; python3 zs_gnn_v3m_bank5seed.py --sl 0.5 --variant s42b5d19 s43b5d19 --resume > logs/f_v3m_05.log 2>&1
else echo "$(date +%T) SMOKE FAILED (logs/smoke_v3m.log)"; exit 1; fi
echo "$(date +%T) ALL DONE"; python3 zs_table.py > logs/all_models.txt 2>&1
