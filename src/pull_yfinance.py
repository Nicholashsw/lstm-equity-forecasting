"""Pull the yfinance development mirror.

Emits the same canonical schema the Bloomberg parser emits:
  panel_ohlcv.parquet : date, ticker, open, high, low, close, volume, adj_close
  context.parquet     : date, series, value   (series in Bloomberg naming)

Usage: python src/pull_yfinance.py [--start 2005-01-01] [--include-v4]
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def _download(symbols, start, tries=3):
    for attempt in range(tries):
        try:
            df = yf.download(symbols, start=start, progress=False,
                             auto_adjust=False, group_by="ticker", threads=True)
            if df is not None and len(df):
                return df
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1}: {e}")
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed for batch starting {symbols[0]}")


def pull_equity_panel(start, include_v4=False):
    shorts = C.V1A_EQUITIES + C.V1A_ETFS + (C.V4_ETFS if include_v4 else [])
    ymap = {C.to_yahoo(s): s for s in shorts}
    ysyms = list(ymap)
    frames, missing = [], []
    for i in range(0, len(ysyms), 20):
        batch = ysyms[i:i + 20]
        print(f"equities batch {i // 20 + 1}: {batch[0]}..{batch[-1]}")
        raw = _download(batch, start)
        for ysym in batch:
            try:
                sub = raw[ysym].dropna(how="all")
            except KeyError:
                missing.append(ysym)
                continue
            if sub.empty:
                missing.append(ysym)
                continue
            out = pd.DataFrame({
                "date": sub.index,
                "ticker": ymap[ysym],
                "open": sub["Open"].values,
                "high": sub["High"].values,
                "low": sub["Low"].values,
                "close": sub["Close"].values,
                "volume": sub["Volume"].values,
                "adj_close": sub["Adj Close"].values,
            })
            frames.append(out)
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    panel = panel.dropna(subset=["close"])
    panel = panel[panel["close"] > 0]
    dup = panel.duplicated(["date", "ticker"]).sum()
    if dup:
        print(f"WARNING: {dup} duplicate keys dropped (keep last)")
        panel = panel.drop_duplicates(["date", "ticker"], keep="last")
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    if missing:
        print(f"MISSING tickers: {missing}")
    panel.to_parquet(C.PROC / "panel_ohlcv.parquet", index=False)
    print(f"panel: {panel.shape[0]:,} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    return panel


FRED_YIELDS = {  # Yahoo no longer serves history for ^TNX/^IRX/^TYX/^MOVE
    "USGG2YR Index": "DGS2",
    "USGG10YR Index": "DGS10",
    "USGG30YR Index": "DGS30",
}


def _pull_fred(series_id, start):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().query("date >= @start")


def pull_context(start):
    frames = []
    for bbg_name, ysym in C.CONTEXT_MAP.items():
        if bbg_name in FRED_YIELDS:
            print(f"context: {bbg_name} <- FRED:{FRED_YIELDS[bbg_name]}")
            f = _pull_fred(FRED_YIELDS[bbg_name], pd.Timestamp(start))
            f["series"] = bbg_name
            frames.append(f[["date", "series", "value"]])
            continue
        if bbg_name == "MOVE Index":
            print("context: MOVE Index unavailable in mirror "
                  "(Bloomberg-only; feature auto-skips)")
            continue
        print(f"context: {bbg_name} <- {ysym}")
        try:
            raw = yf.download(ysym, start=start, progress=False, auto_adjust=False)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {ysym}: {e}")
            continue
        if raw is None or raw.empty:
            print(f"  EMPTY {ysym}")
            continue
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        s = close.dropna()
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(s.index).tz_localize(None),
            "series": bbg_name,
            "value": s.values,
        }))
        time.sleep(0.3)
    ctx = pd.concat(frames, ignore_index=True).sort_values(["series", "date"])
    ctx.to_parquet(C.PROC / "context.parquet", index=False)
    print(f"context: {ctx['series'].nunique()} series, {len(ctx):,} rows")
    return ctx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=C.START_DATE)
    ap.add_argument("--include-v4", action="store_true")
    a = ap.parse_args()
    pull_equity_panel(a.start, a.include_v4)
    pull_context(a.start)
    print("DONE pull_yfinance")
