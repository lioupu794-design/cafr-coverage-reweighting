#!/usr/bin/env bash
set -e

DATA_FILE=${DATA_FILE:-/root/autodl-tmp/PEMS08/PEMS08.npz}
DATA_KEY=${DATA_KEY:-data}
OUT_DIR=${OUT_DIR:-outputs/pems08_obs_sparse_official_stpgnn_full}

python src/run_official_stpgnn_cafr.py \
  --data_file "$DATA_FILE" \
  --data_key "$DATA_KEY" \
  --target_dim 0 \
  --output_dir "$OUT_DIR" \
  --model official_stpgnn \
  --coverage_type obs \
  --rho 0.2 \
  --sparse_ratio 0.2 \
  --low_keep_prob 0.2 \
  --dense_keep_prob 1.0 \
  --graph_k 8 \
  --topology_stress_mode none \
  --use_support_mask \
  --strategies erm cafr random_cafr \
  --lambda_weights 1 2 4 8 \
  --random_cafr_lambda 4 \
  --enable_vg_cafr \
  --vg_beta 1 \
  --vg_gap_mode positive \
  --seeds 2026 2027 2028 \
  --epochs 60 \
  --patience 10 \
  --batch_size 64 \
  --stpgnn_topk 35 \
  --stpgnn_nhid 32 \
  --stpgnn_dims 32 \
  --num_workers 2
