"""Ticker and day counts quoted in the README and paper -> results/data_counts.csv.

Counts only, no prices. Reads the Bloomberg sheet CSVs and the saved LSTM test
predictions, both read-only and not in this repository.
usage: python compute_data_counts.py SHEET_DIR PREDICTIONS_CSV [SUPERVISED_PANEL_PARQUET]
The optional third argument is the supervised panel rebuilt from the current extract.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
import config as C
from parse_bbg_csv import parse_one_csv

sheet_dir, pred_csv = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(sheet_dir.glob("bloomberg_sp500_ohlcv_d_2005_2026_sheet[0-9][0-9].csv"))
long_df = pd.concat([parse_one_csv(f) for f in files], ignore_index=True)
long_df = long_df.drop_duplicates(["date", "ticker", "field"], keep="last")
eq = long_df[~long_df.ticker.isin(C.CONTEXT_MAP)]
eq = eq[(eq.field == "close") & (eq.value > 0)].copy()
eq["ticker"] = eq["ticker"].str.replace(r"\s+[A-Z]{1,2}\s+Equity$", "", regex=True)

etfs = set(parse_one_csv(files[-1]).ticker.str.replace(r"\s+[A-Z]{1,2}\s+Equity$", "", regex=True))
pred = pd.read_csv(pred_csv, usecols=["date", "ticker", "y_true"])
days = eq.groupby("ticker").size()
zero = pred.groupby("date").y_true.apply(lambda x: (x == 0).all())

rows = {
    "workbook_tickers": eq.ticker.nunique(),
    "etf_tickers": len(etfs),
    "used_tickers": pred.ticker.nunique(),
    "stock_tickers": pred.ticker.nunique() - len(etfs),
    "days_FDXF": days["FDXF"],
    "days_HONA": days["HONA"],
    "test_dates": pred.date.nunique(),
    "zero_target_test_dates": int(zero.sum()),
    "lstm_test_rows": len(pred),
    "prediction_files_MB": round(sum(f.stat().st_size for f in pred_csv.parent.rglob("*") if f.is_file()) / 1e6),
    "lstm_test_rows_FISV": int((pred.ticker == "FISV").sum()),
}
if len(sys.argv) > 3:
    sp = pd.read_parquet(sys.argv[3], columns=["ticker", "split"])
    sp = sp[sp.split == "test"]
    rows["rebuild_test_rows"] = len(sp)
    rows["rebuild_test_rows_FISV"] = int((sp.ticker == "FISV").sum())
pd.Series(rows, name="value").rename_axis("key").to_csv(C.RESULTS / "data_counts.csv")
print(pd.Series(rows))
