# H4 Stage Work Order: Horse Race Harness (Stage V2)

## Context
Single evaluation harness that runs every H2 model + H3 LSTM on identical data and emits the paper's main table. V2 uses V1B data when available; runs on V1A meanwhile (flag-switched).

## Deliverables
1. `run_all.py` orchestration: pull(optional) -> build -> train all -> evaluate -> figures. Flags: `--source`, `--stage`, `--skip-pull`, `--models subset,list`, `--quick` (reduced epochs/trees for smoke test).
2. Metrics per model on val AND test: RMSE, MAE, directional accuracy (sign match, zero-pred excluded from naive-zero row), daily cross-sectional Spearman IC (mean and t-stat across days), IC hit-rate.
3. `results/horse_race_results.csv` + figures: (a) test IC bar chart, (b) cumulative IC-weighted spread proxy, (c) LSTM training curve.
4. Sanity battery run automatically: shuffled-target run must produce ~0 IC (leakage alarm if |IC| > 0.02); date-boundary assertion; feature/target timestamp audit.

## Acceptance criteria
- One command reproduces the full table.
- All models score on the exact same (date,ticker) test rows (inner join, count printed).
- Results CSV includes: model, split, n_obs, rmse, mae, dir_acc, ic_mean, ic_tstat, train_seconds.

## Reporting rule
Naive-zero row always printed first. Any model not beating it on test RMSE gets no excuses in the report.
