"""Build the supervised panel: features at close(t), target = log ret t->t+1.

Leakage rules: every feature uses data through close(t) only. Context merged
on date t (same-day close, known before t+1 open). Splits assigned by the
LABEL date (t+1 return is labeled with date t here; membership uses date t).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


def _equity_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c = g["close"]
    logc = np.log(c)
    g["f_ret_1d"] = logc.diff()
    g["f_ret_5d"] = logc.diff(5)
    g["f_ret_21d"] = logc.diff(21)
    g["f_vol_21d"] = g["f_ret_1d"].rolling(21).std()
    g["f_hl_range"] = (g["high"] - g["low"]) / c
    g["f_px_sma20"] = c / c.rolling(20).mean() - 1
    g["f_px_sma50"] = c / c.rolling(50).mean() - 1
    vol = g["volume"].replace(0, np.nan)
    g["f_volu_z"] = ((np.log(vol) - np.log(vol).rolling(21).mean())
                     / np.log(vol).rolling(21).std())
    # targets: next-day log return (t->t+1) aligned to feature date t
    g["target_ret_1d"] = logc.diff().shift(-1)
    # 5-day forward log return (for cross-sectional 5d target; NaN last 5 rows)
    g["fwd_ret_5d"] = (logc.shift(-5) - logc)
    # simple next-day return for realized portfolio P&L
    g["next_simple_1d"] = c.shift(-1) / c - 1.0
    return g


def _context_features(ctx: pd.DataFrame, calendar=None) -> pd.DataFrame:
    wide = ctx.pivot(index="date", columns="series", values="value").sort_index()
    # Align every series to one shared calendar (the equity trading days) so
    # rolling windows don't hit NaN gaps from mismatched source calendars
    # (e.g. MOVE is daily while Fill=P series are weekly-ish). Forward-fill
    # within-series, capped at 5 days so stale values can't leak far.
    if calendar is not None:
        cal = pd.DatetimeIndex(sorted(pd.DatetimeIndex(calendar).unique()))
        full = wide.index.union(cal)
        wide = wide.reindex(full).ffill(limit=5).reindex(cal)
    f = pd.DataFrame(index=wide.index)

    def z(s, w=63):
        return (s - s.rolling(w).mean()) / s.rolling(w).std()

    def chg(s, n):
        return np.log(s).diff(n)

    if "VIX Index" in wide:
        f["c_vix_lvl"] = wide["VIX Index"] / 100.0
        f["c_vix_chg5"] = wide["VIX Index"].diff(5) / 100.0
        f["c_vix_z"] = z(wide["VIX Index"])
    if "MOVE Index" in wide:
        f["c_move_z"] = z(wide["MOVE Index"])
    if "DXY Curncy" in wide:
        f["c_dxy_chg21"] = chg(wide["DXY Curncy"], 21)
    if "USGG10YR Index" in wide and "USGG2YR Index" in wide:
        f["c_term_spread"] = (wide["USGG10YR Index"] - wide["USGG2YR Index"]) / 10.0
    if "USGG10YR Index" in wide:
        f["c_y10_chg5"] = wide["USGG10YR Index"].diff(5) / 10.0
    for name, col in [("oil", "CL1 Comdty"), ("gold", "GC1 Comdty"),
                      ("copper", "HG1 Comdty")]:
        if col in wide:
            f[f"c_{name}_mom21"] = chg(wide[col], 21)
    if "SPX Index" in wide:
        f["c_spx_ret5"] = chg(wide["SPX Index"], 5)
    # regime label: VIX 63d z-score > 1 => high_vol (uses data through t only)
    if "VIX Index" in wide:
        f["regime_highvol"] = (z(wide["VIX Index"]) > 1).astype(int)
    return f.reset_index().rename(columns={"index": "date"})


def build(use_context: bool = True) -> pd.DataFrame:
    panel = pd.read_parquet(C.PROC / "panel_ohlcv.parquet")
    # indices/ETF universe includes index tickers only if pulled; equities only here
    parts = []
    for tkr, g in panel.groupby("ticker", sort=True):
        gg = _equity_features(g)
        gg["ticker"] = tkr
        parts.append(gg)
    feat = pd.concat(parts, ignore_index=True)

    if use_context and (C.PROC / "context.parquet").exists():
        ctx = pd.read_parquet(C.PROC / "context.parquet")
        cf = _context_features(ctx, calendar=feat["date"].unique())
        cf["date"] = pd.to_datetime(cf["date"])
        feat = feat.merge(cf, on="date", how="left")

    fcols = [c for c in feat.columns if c.startswith(("f_", "c_"))]
    out = feat.dropna(subset=fcols + ["target_ret_1d"]).copy()
    # data-error guard
    bad = out["target_ret_1d"].abs() > 1.0
    if bad.sum():
        print(f"dropping {bad.sum()} rows with |ret|>100%")
        out = out[~bad]

    out["split"] = np.where(out["date"] <= C.TRAIN_END, "train",
                    np.where(out["date"] <= C.VAL_END, "val", "test"))

    # ---- cross-sectional targets: demean by day (dollar-neutral relative return)
    out["target_xs_1d"] = (out["target_ret_1d"]
                           - out.groupby("date")["target_ret_1d"].transform("mean"))
    out["target_xs_5d"] = (out["fwd_ret_5d"]
                           - out.groupby("date")["fwd_ret_5d"].transform("mean"))
    # ---- cross-sectional rank features (daily percentile rank of each base feat)
    rank_bases = ["f_ret_1d", "f_ret_5d", "f_ret_21d", "f_vol_21d", "f_volu_z"]
    for b in rank_bases:
        out["xr_" + b] = out.groupby("date")[b].rank(pct=True)
    xrcols = ["xr_" + b for b in rank_bases]

    keep = (["date", "ticker", "split", "target_ret_1d", "target_xs_1d",
             "target_xs_5d", "next_simple_1d"] + fcols + xrcols
            + (["regime_highvol"] if "regime_highvol" in out else []))
    keep = list(dict.fromkeys([c for c in keep if c in out.columns]))
    out = out[keep].sort_values(["ticker", "date"]).reset_index(drop=True)
    out.to_parquet(C.PROC / "supervised_panel.parquet", index=False)
    print(f"supervised panel: {len(out):,} rows | f/c/xr features: "
          f"{len([c for c in keep if c.startswith(('f_','c_','xr_'))])} | "
          f"splits: {out['split'].value_counts().to_dict()}")
    return out


if __name__ == "__main__":
    build(use_context=True)
