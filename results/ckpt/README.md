# `results/ckpt/` — superseded checkpointed development run

**These files are not the reported results.** The canonical horse race is
[`../horse_race_results.csv`](../horse_race_results.csv).

## What this directory is

`run_zoo_ckpt.py` and `run_lstm_ckpt.py` are resume-safe runners: they fit one model at a time and
write a per-model JSON here so an interrupted session can restart without refitting everything. The
JSONs committed here come from a development run executed **before** the final panel was built.

| | `results/ckpt/*.json` (this directory) | `results/horse_race_results.csv` (canonical) |
|---|---|---|
| Test observations | 269,831 (LSTM 269,309) | **466,426** (LSTM 465,904) |
| Validation observations | 123,302 (LSTM 123,187) | 129,795 (LSTM 129,680) |
| Panel | earlier, smaller build | full 518-ticker Bloomberg panel |
| Naive-zero test RMSE | 0.019589 | 0.020411 |
| Produced | 2026-07-07 00:46–01:01 | 2026-07-07 14:34 |

The two runs are on different panels, so their numbers are **not comparable** and are not expected
to agree. Every downstream result in this repository — `xs_regime_results.csv`,
`xs_weekly_results.csv`, `regime_results.csv`, `risk_adjusted_results.csv` — was produced after
14:34 on the canonical panel. The paper and the top-level README quote the canonical panel
throughout (518 tickers, 2,448,678 stock-days, 466,426 test observations).

## The LSTM specifically

`lstm.json` reports test IC 0.0095 (t = 1.32) on 269,309 observations. The canonical LSTM row is
IC 0.0042 (t = 0.76) on 465,904 observations. Neither is statistically distinguishable from zero,
and neither beats the naive-zero baseline on RMSE, so the level-prediction conclusion is the same on
both panels. `run_lstm_ckpt.py` also hardcodes its own `CFG` (1 layer, hidden 32, dropout 0.1, batch
2048, max 8 epochs, patience 2) rather than importing `config.LSTM_CFG`; that is deliberate for a
fast resumable run, but it means this directory's LSTM is not the full-capacity architecture either.

`lstm_state.pt` is the checkpoint state dict for that development run and is kept only so the
resumable path can be demonstrated.

## Re-running

`run_zoo_ckpt.py` skips any model whose JSON already exists, so re-running it against a new panel
without clearing this directory will reuse these stale metrics. It writes its aggregate to
`results/horse_race_results_ckpt.csv`, deliberately **not** to the canonical
`results/horse_race_results.csv`, so a checkpointed re-run cannot silently overwrite the reported
table. To regenerate this directory from scratch, delete its `*.json` and `lstm_state.pt` first.

The canonical table is produced by `run_all.py` (see the top-level README).
