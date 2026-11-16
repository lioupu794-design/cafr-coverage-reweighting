#!/usr/bin/env bash
set -e

python src/run_official_stpgnn_cafr.py \
  --demo \
  --demo_steps 600 \
  --demo_nodes 40 \
  --output_dir outputs/demo_official_stpgnn \
  --model official_stpgnn \
  --coverage_type spa \
  --rho 0.2 \
  --graph_k 8 \
  --topology_stress_mode coverage \
  --topology_weak_ratio 0.2 \
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
  --batch_size 16 \
  --stpgnn_topk 8 \
  --stpgnn_nhid 16 \
  --stpgnn_dims 16 \
  --num_workers 0
