#!/bin/bash
mkdir -p logs
python3 zs_gnn_v3g_graph.py --sl 1.0 --variant knn9 knn19 dense19 --resume > logs/v3g_10.log 2>&1
python3 zs_gnn_v3h_anat.py --sl 1.0 0.5 --variant fga --resume > logs/v3h.log 2>&1
python3 zs_gnn_v3g_graph.py --sl 0.5 --variant knn9 knn19 dense19 --resume > logs/v3g_05.log 2>&1
python3 zs_gnn_v3g_graph.py --sl 0.5 --variant knn64 dense64 --resume > logs/v3g_64.log 2>&1
