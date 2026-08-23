# H5 Stage Work Order: Shock/Regime Context (Stage V3)

## Context
Tests the thesis that market context helps most in stress. Consumes `context.parquet` (12 series) + the V1A/V1B panel.

## Deliverables
1. Context features in `build_dataset.py` (behind `use_context` flag): for each series — 1d/5d/21d change, 63d z-score; term-spread (10Y-2Y proxy), VIX level + 5d change, MOVE z, DXY 21d change, oil/copper/gold 21d momentum. All lagged to close(t).
2. Regime labeler: `regime_t = high_vol if VIX_z(63d) > 1 else normal` (simple, defensible; HMM is extension E3).
3. Analyses: (a) horse race metrics recomputed per regime; (b) LSTM with vs without context, per regime; (c) shock event study: model errors around the 10 largest |SPX| move days in test.
4. `results/regime_results.csv` + 2 figures (per-regime IC bars; context vs no-context LSTM comparison).

## Acceptance criteria
- Context features never use information after close(t). Regime label at t uses data through t (it conditions evaluation, not features — both uses documented separately).
- Clear stated answer in the run log: "context helps / doesn't help, and where."

## Do NOT
Do not tune the regime threshold on test performance.
