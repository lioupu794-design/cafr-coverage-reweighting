# Coverage-Aware Fair Reweighting (CAFR)

Code, scripts and figure sources for the paper *Fair Traffic Flow Forecasting
under Uneven Coverage Conditions*.

The method reweights a node-level forecasting loss by **historical observation
coverage** and **spatial neighbourhood support**, both computed from the
training period before any model is fit. The reweighting strength is selected
on validation data (VG-CAFR). The forecasting backbone and the inference path
are unchanged.

## Environment

```
python 3.10
torch 2.1.2+cu118      torchvision 0.16.2
numpy  pandas  scipy  matplotlib
GPU with ~8 GB memory (CPU also works, roughly 20x slower)
```

```bash
pip install -r requirements.txt
```

## Data

PEMS04 and PEMS08 are public. Download and place them under `data/`:

| file | size | source |
|---|---|---|
| `data/PEMS08.npz` | 18,525,419 B | https://github.com/guoshnBJTU/ASTGNN/tree/master/data/PEMS08 |
| `data/PEMS04.npz` | 32,956,284 B | https://github.com/guoshnBJTU/ASTGNN/tree/master/data/PEMS04 |

Each file stores a `(num_steps, num_nodes, 3)` array under the key `data`.

## Quick start

Topology-support scenario on PEMS08, ERM plus four matched-protocol baselines:

```bash
bash scripts/run_pems08_topology_official_stpgnn_full.sh
```

Observation-sparsity scenario on PEMS08:

```bash
bash scripts/run_pems08_obs_sparse_official_stpgnn_full.sh
```

CAFR with the four candidate strengths and validation-based selection:

```bash
python src/run_official_stpgnn_cafr.py \
  --data_file data/PEMS08.npz --data_key data --target_dim 0 \
  --output_dir outputs/cafr --model official_stpgnn \
  --coverage_type spa --rho 0.1 --graph_k 8 \
  --topology_stress_mode coverage --topology_weak_ratio 0.1 \
  --weak_keep_degree 0 --weak_edge_scale 0.0 --use_support_mask \
  --strategies cafr --lambda_weights 1 2 4 8 --enable_vg_cafr \
  --seeds 2026 2027 2028 --epochs 60 --patience 10 --batch_size 64
```

Every run writes `summary_by_run.csv` and `strategy_comparison.csv` under
`--output_dir`.

## Repository layout

```
src/          training script (models, coverage scores, reweighting, selection)
scripts/      one shell entry per reported scenario
third_party/  STPGNN reference implementation, kept verbatim with its licence
figures/      scripts that redraw Fig. 1-3 and the resulting PDFs
results/      result tables quoted in the paper (see below)
```

## Results

| file | what it holds |
|---|---|
| `results/strategy_comparison_baselines.csv` | ERM and the four baselines re-run on the STPGNN backbone (PEMS08 topology setting), 63 columns including per-horizon errors; this is the backbone check, not the Graph WaveNet table of the paper |
| `results/summary_by_run.csv` | the same runs before averaging, one row per seed |
| `results/cafr_lambda_sweep.csv` | Section 4.6: CAFR at lambda in {1, 2, 4, 8} and the validation-selected VG-CAFR |
| `results/training_log.txt` | stdout of the baseline run |

All entries are means over seeds 2026, 2027 and 2028. The ERM and CAFR rows of
Tables 2 and 3 are reproduced by the commands above; the per-run logs of those
runs were not archived, so only the aggregate sweep file is shipped.

## Figures

```bash
python figures/make_fig1_v3.py   # problem illustration
python figures/make_fig2_v3.py   # pipeline
python figures/make_fig3.py      # sparsity sweep
```

## Notes

* `--stpgnn_topk 35` is correct for PEMS08; use `70` for PEMS04.
* Runs are independent and can be launched in parallel; each seed writes to its
  own sub-directory.
* The coverage scores are computed from training-period observations only, so
  no validation or test error enters the group definition.
