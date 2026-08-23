"""Parse Bloomberg CSVs exported by mod_BBG_UROP_Export.bas into canonical parquet.

Block layout (0-indexed rows after read with header=None):
  rows 0-3  metadata
  row  4    ticker (first column of each 3-col block)
  row  5    field
  rows 7+   [date, value, spacer] per block

Usage:
  python src/parse_bbg_csv.py --indir data/raw/bloomberg           # build panel+context
  python src/parse_bbg_csv.py --indir data/raw/bloomberg --check-parity
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

FIELD_MAP = {
    "PX_OPEN": "open", "PX_HIGH": "high", "PX_LOW": "low",
    "PX_LAST": "close", "VOLUME": "volume",
    "TOT_RETURN_INDEX_GROSS_DVDS": "tri",
}
NA_STRINGS = {"#N/A N/A", "#N/A Requesting Data", "#N/A Invalid Security",
              "#NAME?", "N/A", ""}


def parse_one_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False,
                      low_memory=False)
    if raw.shape[0] < 8:
        print(f"  skip {path.name}: too few rows")
        return pd.DataFrame()
    recs, pending_na = [], 0
    for c0 in range(0, raw.shape[1], 3):
        ticker = str(raw.iat[4, c0]).strip() if raw.shape[0] > 4 else ""
        field = str(raw.iat[5, c0]).strip() if raw.shape[0] > 5 else ""
        if not ticker or field not in FIELD_MAP:
            continue
        dates = raw.iloc[7:, c0]
        vals = raw.iloc[7:, c0 + 1] if c0 + 1 < raw.shape[1] else pd.Series(dtype=str)
        blk = pd.DataFrame({"d": dates.values, "v": vals.reindex(dates.index).values})
        pending_na += blk["v"].isin(NA_STRINGS).sum() - (blk["v"] == "").sum()
        blk["date"] = pd.to_datetime(blk["d"], errors="coerce")
        blk["value"] = pd.to_numeric(blk["v"], errors="coerce")
        blk = blk.dropna(subset=["date", "value"])
        if blk.empty:
            continue
        blk["ticker"] = ticker
        blk["field"] = FIELD_MAP[field]
        recs.append(blk[["date", "ticker", "field", "value"]])
    if pending_na > 0:
        print(f"  WARNING {path.name}: {pending_na} #N/A cells "
              f"(freeze may have run before Bloomberg finished)")
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def build(indir: Path):
    files = sorted(indir.glob("BBG_*.csv"))
    if not files:
        raise SystemExit(f"No BBG_*.csv files in {indir}")
    long_frames = []
    for f in files:
        print(f"parsing {f.name}")
        df = parse_one_csv(f)
        if len(df):
            long_frames.append(df)
    long_df = pd.concat(long_frames, ignore_index=True)
    dup = long_df.duplicated(["date", "ticker", "field"]).sum()
    if dup:
        print(f"WARNING: {dup} duplicate (date,ticker,field) rows; keeping last")
        long_df = long_df.drop_duplicates(["date", "ticker", "field"], keep="last")

    # Context tickers (single-field PX_LAST universe) go to context.parquet
    ctx_names = set(C.CONTEXT_MAP.keys())
    is_ctx = long_df["ticker"].isin(ctx_names)
    ctx = (long_df[is_ctx & (long_df["field"] == "close")]
           .rename(columns={"ticker": "series"})[["date", "series", "value"]]
           .sort_values(["series", "date"]))
    eq = long_df[~is_ctx]

    panel = (eq.pivot_table(index=["date", "ticker"], columns="field",
                            values="value", aggfunc="last")
             .reset_index())
    panel.columns.name = None
    # normalize ticker: strip any Bloomberg equity yellow-key suffix
    # ("AAPL UW Equity" / "A UN Equity" / "VWRA LN Equity" -> short ticker)
    panel["ticker"] = panel["ticker"].str.replace(
        r"\s+[A-Z]{1,2}\s+Equity$", "", regex=True)
    panel = panel.dropna(subset=["close"])
    panel = panel[panel["close"] > 0].sort_values(["ticker", "date"])

    if len(panel):
        panel.to_parquet(C.PROC / "panel_ohlcv.parquet", index=False)
        print(f"panel: {len(panel):,} rows, {panel['ticker'].nunique()} tickers")
    if len(ctx):
        ctx.to_parquet(C.PROC / "context.parquet", index=False)
        print(f"context: {ctx['series'].nunique()} series")


def check_parity(indir: Path):
    """Compare Bloomberg close vs existing yfinance parquet."""
    yf_panel = pd.read_parquet(C.PROC / "panel_ohlcv.parquet")[
        ["date", "ticker", "close"]].rename(columns={"close": "close_yf"})
    frames = [parse_one_csv(f) for f in sorted(indir.glob("BBG_*.csv"))]
    bbg = pd.concat(frames, ignore_index=True)
    bbg = bbg[bbg["field"] == "close"].copy()
    bbg["ticker"] = bbg["ticker"].str.replace(r"\s+[A-Z]{1,2}\s+Equity$", "", regex=True)
    m = bbg.merge(yf_panel, on=["date", "ticker"])
    m["abs_pct_diff"] = (m["value"] / m["close_yf"] - 1).abs()
    rep = (m.groupby("ticker")["abs_pct_diff"]
           .agg(["median", "mean", "max", "count"]).sort_values("median"))
    rep.to_csv(C.RESULTS / "parity_report.csv")
    print(rep.tail(15))
    print(f"parity report -> {C.RESULTS / 'parity_report.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=str(C.RAW / "bloomberg"))
    ap.add_argument("--check-parity", action="store_true")
    a = ap.parse_args()
    if a.check_parity:
        check_parity(Path(a.indir))
    else:
        build(Path(a.indir))
