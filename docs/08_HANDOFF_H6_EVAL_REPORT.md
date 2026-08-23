# H6 Stage Work Order: Final Evaluation + Report

## Context
Converts results into the UROP report. All numbers must come from committed CSVs; all figures from committed code.

## Deliverables
1. `docs/REPORT.md` (source for the final PDF): Abstract, Data (Bloomberg pipeline + yfinance mirror + parity), Methodology (features, splits, leakage protocol, sanity battery results), Models, Results (horse race table, regime analysis), Limitations (survivorship, current-members universe, single-market), Extensions (E1-E6 with 1-para each), Reproducibility appendix (exact commands, seeds, package versions via `pip freeze` snapshot).
2. Figure pass: consistent styling, labeled axes, test-window shading on any time-series plot.
3. Honest-results audit: every claim cross-checked against `horse_race_results.csv`; negative findings stated plainly.
4. `README.md` at repo root: 10-line quickstart.

## Acceptance criteria
- A reader with the repo + data can rebuild every number.
- No figure or table exists that the code cannot regenerate.
- Limitations section >= as long as the "wins" paragraph. This is what makes the project credible to a supervisor.
