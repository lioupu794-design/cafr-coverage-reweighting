#!/usr/bin/env bash
set -e

DATA_FILE=${DATA_FILE:-/root/autodl-tmp/PEMS08/PEMS08.npz}
DATA_KEY=${DATA_KEY:-data}
OUT_DIR=${OUT_DIR:-outputs/pems08_topology_official_stpgnn_quick}

python src/run_official_stpgnn_cafr.py \
  --data_file "$DATA_FILE" \
  --data_key "$DATA_KEY" \
  --target_dim 0 \
  --output_dir "$OUT_DIR" \
  --model official_stpgnn \
  --coverage_type spa \
  --rho 0.1 \
  --graph_k 8 \
  --topology_stress_mode coverage \
  --topology_weak_ratio 0.1 \
  --weak_keep_degree 0 \
  --weak_edge_scale 0.0 \
  --use_support_mask \
  --strategies erm cafr random_cafr \
  --lambda_weights 4 \
  --random_cafr_lambda 4 \
  --enable_vg_cafr \
  --vg_beta 1 \
  --vg_gap_mode positive \
  --seeds 2026 \
  --epochs 2 \
  --patience 2 \
  --batch_size 64 \
  --stpgnn_topk 35 \
  --stpgnn_nhid 32 \
  --stpgnn_dims 32 \
  --num_workers 2
