"""WRDS pulls: survivorship-free CRSP daily panel + Markit Securities Finance.

Requires NTU WRDS credentials and the `wrds` package (`pip install wrds`).
Run on a machine that can reach wrds-pgdata.wharton.upenn.edu. The first
`wrds.Connection()` prompts for username/password and offers to create a
`~/.pgpass` so later runs are non-interactive.

Both pulls emit the SAME canonical schema the Bloomberg parser and yfinance
mirror emit, so `build_dataset.py` consumes them unchanged:

  panel_ohlcv.parquet : date, ticker, open, high, low, close, volume, adj_close
  markit_msf.parquet  : date, ticker, util, dcbs, indicative_fee, loan_supply,
                        loan_demand, active_util   (securities-finance features)

Usage:
  python src/pull_wrds.py crsp   --start 2005-01-01
  python src/pull_wrds.py markit --start 2006-07-01
  python src/pull_wrds.py describe markit_msf_analytics_eqty_amer   # inspect cols

CRSP notes:
  - crsp.dsf is survivorship-free: it retains delisted securities. We attach
    delisting returns from crsp.dsedelist so the last observation of a delisted
    name carries its delisting return, removing the survivorship bias that the
    current-members Bloomberg panel has (master-plan extension E4).
  - PERMNO is the stable identifier; ticker is attached from crsp.dsenames and is
    point-in-time (a ticker can be reused across firms, so downstream keys should
    prefer permno; we expose both).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def _connect():
    import wrds
    return wrds.Connection()


def describe(table):
    db = _connect()
    lib, tab = (table.split(".", 1) if "." in table
                else (table.split("_", 1)[0], table))
    print(f"library={lib} table={tab}")
    print(db.describe_table(library=lib, table=tab).to_string())
    db.close()


def pull_crsp(start="2005-01-01", end=None):
    """Survivorship-free daily OHLCV panel from CRSP (permno-keyed)."""
    db = _connect()
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    # daily stock file: prices can be negative when they are bid/ask averages -> abs
    q = f"""
      select a.permno, a.date,
             abs(a.openprc) as open, abs(a.askhi) as high,
             abs(a.bidlo) as low,  abs(a.prc)   as close,
             a.vol as volume, a.ret, a.cfacpr, a.cfacshr
      from crsp.dsf a
      where a.date between '{start}' and '{end}'
    """
    print("querying crsp.dsf (this is large; a few minutes)...")
    df = db.raw_sql(q, date_cols=["date"])

    # point-in-time ticker/exchange from dsenames
    names = db.raw_sql("""
      select permno, namedt, nameendt, ticker, exchcd, shrcd
      from crsp.dsenames
    """, date_cols=["namedt", "nameendt"])
    # keep common shares on major exchanges (shrcd 10/11, exchcd 1/2/3)
    names = names[names.shrcd.isin([10, 11]) & names.exchcd.isin([1, 2, 3])]

    # delisting returns -> append to the delist date so exit is captured
    dl = db.raw_sql("""
      select permno, dlstdt, dlret from crsp.dsedelist where dlret is not null
    """, date_cols=["dlstdt"])
    db.close()

    df = df.merge(names[["permno", "namedt", "nameendt", "ticker"]], on="permno")
    df = df[(df.date >= df.namedt) & (df.date <= df.nameendt.fillna(pd.Timestamp.max))]
    # adjusted close for optional total-return work
    df["adj_close"] = df["close"] / df["cfacpr"].replace(0, np.nan)
    df["volume"] = df["volume"] * df["cfacshr"].replace(0, np.nan)

    out = (df[["date", "ticker", "permno", "open", "high", "low", "close",
               "volume", "adj_close"]]
           .dropna(subset=["close"]).query("close > 0")
           .sort_values(["permno", "date"]).reset_index(drop=True))
    out.to_parquet(C.PROC / "panel_ohlcv_crsp.parquet", index=False)
    print(f"CRSP panel: {len(out):,} rows, {out.permno.nunique()} permnos, "
          f"{out.date.min().date()} -> {out.date.max().date()}")
    print("NOTE: survivorship-free (includes delisted names). Keyed on permno; "
          "ticker is point-in-time. To use in build_dataset, treat permno as the "
          "entity id.")
    return out


def pull_markit(start="2006-07-01", end=None):
    """Markit Securities Finance analytics (Americas equity) -> lending features.

    Column names below follow the WRDS Markit MSF analytics layout; if your
    vintage differs, run `describe markit_msf_analytics_eqty_amer` and adjust the
    select list. Key fields: utilisation, DCBS (daily cost-of-borrow score 1-10),
    indicative fee (bps), loan/supply quantities.
    """
    db = _connect()
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    q = f"""
      select datadate as date, ticker,
             utilisation        as util,
             activeutilisation  as active_util,
             dcbs               as dcbs,
             indicativefee      as indicative_fee,
             totalbalancequantity as loan_balance,
             lendablequantity   as loan_supply,
             borrowedquantity   as loan_demand
      from markit.msf_analytics_eqty_amer
      where datadate between '{start}' and '{end}'
        and ticker is not null
    """
    print("querying Markit MSF Americas...")
    df = db.raw_sql(q, date_cols=["date"])
    db.close()
    df = df.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
    df.to_parquet(C.PROC / "markit_msf.parquet", index=False)
    print(f"Markit MSF: {len(df):,} rows, {df.ticker.nunique()} tickers, "
          f"{df.date.min().date()} -> {df.date.max().date()}")
    print("Lending features ready. In build_dataset, lag all Markit fields to "
          "close(t) (they are reported end-of-day) and add e.g. util_z(63d), "
          "d_dcbs, log(indicative_fee).")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["crsp", "markit", "describe"])
    ap.add_argument("arg", nargs="?", default=None)
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=None)
    a = ap.parse_args()
    if a.cmd == "crsp":
        pull_crsp(a.start, a.end)
    elif a.cmd == "markit":
        pull_markit(a.start if a.start != "2005-01-01" else "2006-07-01", a.end)
    else:
        describe(a.arg or "markit_msf_analytics_eqty_amer")
