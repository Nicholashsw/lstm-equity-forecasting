# Master Plan: Deep Learning Based Stock Price Forecasting Using LSTM Networks

**Owner:** Nicholas Hong | **Type:** Finance UROP | **Status:** Data infra complete, modeling in progress

## One-paragraph summary
Build a reproducible panel-forecasting system that predicts next-day equity returns for US large caps, benchmarks an LSTM against classical ML (the "horse race"), and extends to shock/regime-conditional prediction using market context series. Data comes from Bloomberg BDH exports (via `mod_BBG_UROP_Export.bas`) with a yfinance mirror for development off-terminal. Everything runs from `run_all.py`.

## Stage map
| Stage | Purpose | Data | Status |
|---|---|---|---|
| V1A_US_CORE | Standalone LSTM proof, fast iteration | ~60 US large caps + ETFs + indices, OHLCV, 2005-present | **Runs now on yfinance mirror** |
| V1B_SP500 | Serious panel dataset | Full S&P 500, OHLCV | Awaiting Bloomberg CSVs |
| V2_HORSE_RACE | Model comparison on identical V1B data | Reuses V1B | Framework built, runs on V1A now |
| V3_SHOCK_REGIME | Market context features + regime prediction | 12 macro series (VIX, MOVE, DXY, UST curve, commodities) | **Runs now on yfinance mirror** |
| V4_GLOBAL_ETF | Robustness across geographies | 17 country/region ETFs | Optional, parser ready |

## Pipeline (what run_all.py does)
1. **Pull** — `pull_yfinance.py` downloads the dev mirror (or `parse_bbg_csv.py` ingests Bloomberg CSVs when available; both emit the identical long-format parquet schema).
2. **Build** — `build_dataset.py` merges equity OHLCV with context series, computes features, forms the supervised panel with strict no-lookahead alignment.
3. **Train** — `models_zoo.py` (Naive, Linear, Ridge, ElasticNet, Tree, RF, XGBoost, LightGBM, SVR-linear, Poly-Ridge) + `lstm_model.py` (PyTorch LSTM, pooled panel training).
4. **Evaluate** — chronological train/val/test split, metrics: RMSE, MAE, directional accuracy, daily cross-sectional IC. Outputs `results/horse_race_results.csv` + figures.

## Data contract (Bloomberg <-> yfinance parity)
Both sources normalize to: `date, ticker, field, value` long format, then pivot to `date, ticker, open, high, low, close, volume`. Context series: `date, series, value`. See `02_DATA_SPEC.md` for exact Bloomberg block-layout parsing rules and yfinance ticker mapping. **When Bloomberg CSVs arrive, only step 1 changes; steps 2-4 are source-agnostic.**

## Splits and leakage rules (non-negotiable)
- Chronological split only. Default: train <= 2021-12-31, val 2022, test 2023+.
- All features use information available at close of day t; target is return t -> t+1.
- Scalers/encoders fit on train only.
- No survivorship correction in V1A/V1B-current-members (documented limitation; extension E4 addresses it).

## Extensions roadmap (post core deliverable)
- **E1 Attention/Transformer:** replace LSTM encoder with temporal attention; compare param-matched.
- **E2 Probabilistic forecasts:** quantile loss heads, CRPS evaluation, VaR-style shock warnings.
- **E3 Regime-conditional models:** HMM/threshold-VIX regimes; per-regime model gating; does context (V3) help most in stress regimes?
- **E4 Survivorship-corrected panel:** point-in-time S&P membership (Bloomberg INDX_MWEIGHT_HIST), re-run horse race.
- **E5 Economic evaluation:** long-short decile portfolio from predictions, net-of-cost Sharpe, turnover.
- **E6 Multi-horizon:** 1d/5d/21d targets, horizon-dependent model ranking.

## Handoff index (each is a self-contained stage work order)
- H1 `03_HANDOFF_H1_DATA.md` — data ingestion + Bloomberg parser
- H2 `04_HANDOFF_H2_BASELINES.md` — classical model zoo
- H3 `05_HANDOFF_H3_LSTM.md` — LSTM architecture + training
- H4 `06_HANDOFF_H4_HORSERACE.md` — V2 evaluation harness
- H5 `07_HANDOFF_H5_REGIME.md` — V3 context/regime extension
- H6 `08_HANDOFF_H6_EVAL_REPORT.md` — final report + figures
- H7 `09_HANDOFF_H7_WRDS.md` — WRDS extension: survivorship fix + lending signal

## Definition of done (core)
1. `python run_all.py` completes from a clean clone with only data files present.
2. Horse race table over >= 10 models on identical test window.
3. LSTM beats naive baseline on directional accuracy OR the report honestly documents that it does not (negative results are acceptable; leakage is not).
4. Report reproduces every figure from committed code.
