# H1 Stage Work Order: Data Ingestion Layer

## Context
You are implementing the ingestion layer for an LSTM stock forecasting UROP. Read `00_MASTER_PLAN.md` and `02_DATA_SPEC.md` first. The Bloomberg VBA exporter already exists (`mod_BBG_UROP_Export.bas`); your job is the Python side.

## Inputs available to you
- `configs/universes` defined in `src/config.py` (V1A/V3/V4 lists + Yahoo mapping)
- Sample Bloomberg CSVs will be provided under `data/raw/bloomberg/` (block layout per 02_DATA_SPEC section 2). Until then, generate a synthetic fixture CSV in tests.

## Deliverables
1. `src/pull_yfinance.py` — CLI: `--stage v1a|v3|v4|all --start 2005-01-01`. Batched download (<= 20 tickers/request), retry w/ backoff, writes canonical parquets.
2. `src/parse_bbg_csv.py` — CLI: `--indir data/raw/bloomberg --stage v1a|v1b|v3|v4`. Implements parser rules 1-6 exactly. `--check-parity` mode per spec section 5.
3. `tests/test_parser.py` — synthetic block-layout fixture (3 tickers x 2 fields, ragged lengths, one `#N/A Requesting Data` cell) must round-trip to correct long format.

## Acceptance criteria
- Both sources emit byte-identical schema (columns, dtypes, sort order).
- Parser handles: ragged blocks, empty trailing columns, N/A strings, duplicate keys (keep-last + logged warning).
- Zero silent drops: every dropped row category is counted and printed.
- `pytest tests/` green.

## Step-by-step
1. Read config universes; build Yahoo symbol map (strip suffixes, BRK/B rule, index carets).
2. Implement yfinance pull with `auto_adjust=False`, per-batch try/except, missing-ticker report.
3. Implement block parser: read `header=None`, stride 3 columns, validate rows 5/6, coerce, melt, pivot.
4. Write parity check.
5. Write tests, run, fix.

## Do NOT
- Do not forward-fill prices. Do not use adjusted close as `close`. Do not reorder the canonical column order.
