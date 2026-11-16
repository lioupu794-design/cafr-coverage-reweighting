# Results

Drop the CSV files exported by the training script here.

| file to add | produced by |
|---|---|
| `strategy_comparison_erm.csv` | ERM-only runs, all backbones and scenarios |
| `strategy_comparison_cafr.csv` | CAFR with lambda in {1, 2, 4, 8} plus VG-CAFR |
| `strategy_comparison_baselines.csv` | ERM, Inverse Freq, Gap Reg, Smooth GDRO, Random-CAFR |
| `summary_by_run.csv` | per-seed rows for all of the above |

Each `strategy_comparison.csv` written by `src/run_official_stpgnn_cafr.py`
has the columns `strategy, lambda, overall_mae, low_mae, high_mae, gap` and is
already averaged over the three seeds. Rename the copies as in the table above
so that the README mapping stays valid.
