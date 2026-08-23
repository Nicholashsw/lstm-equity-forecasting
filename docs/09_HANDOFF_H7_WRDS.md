# H7 Stage Work Order: WRDS Integration — Survivorship Fix + Lending Signal

## Context
NTU has WRDS access. Two accessible datasets close the two biggest gaps in the
current result: CRSP (survivorship-free prices) and Markit Securities Finance
Americas (equity lending / short-side signal). Read `00_MASTER_PLAN.md` (E4),
`02_DATA_SPEC.md`, and the risk-adjustment finding in `results/risk_adjusted_results.csv`.

Confirmed entitlements (from the NTU WRDS product list): crsp_a_stock,
crsp_a_indexes, crsp_a_ccm, comp_na_daily_all, tr_ibes, tr_13f, tr_insiders,
markit_msf_analytics_eqty_amer (Americas, subscribed), markit_msf_analytics_eqty_asia,
ff_all, contrib_global_factor.

## Task A: survivorship-free rebuild (highest rigor value)
1. `pip install wrds`; run `python src/pull_wrds.py crsp --start 2005-01-01`.
   Produces `data/processed/panel_ohlcv_crsp.parquet`, permno-keyed, with
   delisting returns attached (includes delisted names).
2. Add a `--source crsp` path to `build_dataset.py`: entity id = permno, not
   ticker. Everything else (features, splits, leakage rules) is unchanged.
3. Re-run the daily horse race and the cross-sectional regime analysis on the
   CRSP panel. Compare high-vol IC vs the Bloomberg current-members panel.
   Expected: effect persists but is weaker (survivorship inflates the old number).
   Report the delta explicitly; this is the E4 deliverable.

## Task B: Markit lending signal (highest upside)
1. `python src/pull_wrds.py describe markit_msf_analytics_eqty_amer` FIRST to
   confirm column names for your vintage, then `python src/pull_wrds.py markit`.
2. In `build_dataset.py` (flag `use_lending`), merge Markit on (date,ticker),
   lag all fields to close(t), and add: `util_z` (63d z-score of utilisation),
   `d_dcbs` (change in daily cost-of-borrow score), `log_fee` (log indicative
   fee), `demand_supply_ratio`. These are known cross-sectional return predictors.
3. Add these to the cross-sectional model and re-run the regime split. Test the
   specific hypothesis: does a lending-based signal produce alpha in the high-vol
   regime where the price-only signal did not survive risk adjustment?
4. Risk-adjust the resulting long-short with `src/risk_adjust.py` (FF3+Mom).
   The bar is a significant alpha t-stat, not just gross IC.

## Task C: proper alpha reporting
Replace raw long-short Sharpe in the paper/report with FF3+Momentum alpha and
factor betas (`src/risk_adjust.py` already does this; extend to the CRSP and
lending variants). A significant alpha after four factors is the difference
between "loads on known factors" and "new signal".

## Hard rules
- CRSP/Markit are licensed; never commit `data/`. Keep it in `.gitignore`.
- permno is the entity key for CRSP; do not silently fall back to ticker.
- Lending fields are end-of-day: they MUST be lagged to close(t). A same-day
  merge without lag is lookahead.
- Report survivorship delta honestly even if it kills the effect.

## Definition of done
`panel_ohlcv_crsp.parquet`, `markit_msf.parquet`, regime results on both,
`risk_adjusted_results.csv` extended to CRSP + lending variants, and one
paragraph: does any variant deliver significant four-factor alpha in any regime?
