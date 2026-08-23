# Data Specification

## 1. Canonical schemas (source-agnostic)
**Equity panel** `data/processed/panel_ohlcv.parquet`:
`date (datetime64), ticker (str), open, high, low, close, volume (float)` — one row per ticker-date, sorted, no duplicate keys.

**Context series** `data/processed/context.parquet`:
`date, series (str), value (float)` — series names use Bloomberg convention (e.g. `VIX Index`).

## 2. Bloomberg source (production)
Input: CSVs exported by `mod_BBG_UROP_Export.bas` from sheets `BBG_V1A_001...` etc.

**Block layout per CSV (must parse exactly):**
- Rows 1-4: metadata (skip; row 3 contains date range for validation)
- Row 5: ticker string at the first column of each 3-column block
- Row 6: field string (PX_OPEN / PX_HIGH / PX_LOW / PX_LAST / VOLUME)
- Row 8+: two columns per block: Excel date, value; third column blank spacer
- Blocks tile horizontally every 3 columns; blocks may have different lengths (Fill=P, Days=W)

**Parser rules (`src/parse_bbg_csv.py`):**
1. Read raw with `header=None`. Iterate columns 0, 3, 6, ...
2. Block valid iff row-5 cell non-empty and row-6 cell in known field set.
3. Dates: accept `M/D/YYYY`, `YYYY-MM-DD`, or Excel serial; coerce with `pd.to_datetime(errors='coerce')`, drop NaT.
4. Values: numeric coercion; drop strings like `#N/A N/A`, `#N/A Requesting Data` (also log their count — nonzero means the freeze ran too early).
5. Field mapping: PX_OPEN->open, PX_HIGH->high, PX_LOW->low, PX_LAST->close, VOLUME->volume, TOT_RETURN_INDEX_GROSS_DVDS->tri.
6. Concatenate all blocks -> long format -> pivot -> canonical schema. Duplicate (date,ticker) keys: keep last, log count.

## 3. yfinance mirror (development)
`auto_adjust=False`; use unadjusted OHLC + Volume to match Bloomberg PX_* fields. (Adjusted close kept as `adj_close` for optional total-return work.)

**Ticker mapping (Bloomberg -> Yahoo):**
| Bloomberg | Yahoo | Note |
|---|---|---|
| AAPL US Equity | AAPL | strip " US Equity" |
| BRK/B US Equity | BRK-B | slash -> dash |
| VWRA LN Equity | VWRA.L | LSE suffix |
| SPX Index | ^GSPC | |
| NDX Index | ^NDX | |
| RTY Index | ^RUT | |
| INDU Index | ^DJI | |
| VIX Index | ^VIX | |
| MOVE Index | ^MOVE | Yahoo has it from ~2002 |
| DXY Curncy | DX-Y.NYB | ICE dollar index |
| USGG2YR Index | (proxy) ^IRX | Yahoo lacks 2Y; 13-week bill yield used as short-rate proxy. Flagged; Bloomberg is authoritative |
| USGG10YR Index | ^TNX | CBOE 10Y yield index |
| USGG30YR Index | ^TYX | |
| CL1 Comdty | CL=F | front WTI |
| GC1 Comdty | GC=F | front gold |
| HG1 Comdty | HG=F | front copper |

## 4. Cleaning rules (both sources)
- Drop ticker-dates with close <= 0 or missing close.
- Volume NaN -> 0 for indices/yields (they legitimately have none), else leave NaN and forward-fill nothing (prices are never forward-filled at panel level; Fill=P already handled non-trading days at source).
- Winsorize daily returns at |r| > 100% -> drop row and log (data error guard).
- Calendar: inner alignment happens at feature stage, not by reindexing prices.

## 5. Parity check (when Bloomberg data arrives)
`python src/parse_bbg_csv.py --check-parity` compares close prices for overlapping tickers/dates vs the yfinance parquet; report mean/max abs pct diff per ticker. Tolerance: <= 0.5% median diff (splits/timing explain most residual). Output: `results/parity_report.csv`.
