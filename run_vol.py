"""Realised-volatility extension: where the horse race has something to forecast.

The level race found nothing beats zero on next-day returns. Volatility is
persistent, so this asks the same question on a target that is forecastable:
log realised variance over the next 21 trading days,

    y(t) = log( mean of r^2 over t+1 .. t+21 ),   r = daily log return.

Same panel, same splits (train <= 2021, val 2022, test 2023+). Training and
validation rows are purged so that no label window reaches past its split end.
Test origins are every 5th test day for every ticker.

Models
  naive         trailing 21-day log RV (random walk)
  HAR, HAR-X    pooled panel OLS on daily / weekly / monthly / quarterly log RV
                (HAR-X adds log VIX^2), fit on train + val
  LightGBM      pooled, all panel features + HAR features, val early stopping
  LSTM          src/lstm_model.py unchanged, compact config, vol target
  TimesFM 2.5 / TimesFM 3 / Chronos-2   zero-shot on each ticker's trailing log RV
  "+ vix"       the same with log VIX^2 as a past-only covariate
  averages      equal-weight combinations of the above

Scoring: RMSE and QLIKE on log RV, Diebold-Mariano vs HAR on the daily
cross-sectional mean loss (Newey-West, 5 lags for the overlapping windows),
daily cross-sectional rank IC of predicted vs realised vol, and the same
VIX regime flag as the level race.

  python run_vol.py            -> results/foundation/vol_results.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as C  # noqa: E402
from foundation import TimesFM25, TimesFM3, Chronos2  # noqa: E402

OUT = C.RESULTS / "foundation"
OUT.mkdir(parents=True, exist_ok=True)
H = 21
STEP = 5
FLOOR = 1e-8
HAR = ["lrv_d", "lrv_w", "lrv_m", "lrv_q"]


def vol_panel():
    from run_foundation import load_panel
    p = load_panel()
    p["date"] = pd.to_datetime(p["date"])
    parts = []
    for tkr, g in p.groupby("ticker", sort=True):
        g = g.sort_values("date").copy()
        r2 = g["f_ret_1d"] ** 2
        rv21 = r2.rolling(21).mean()
        g["lrv_d"] = np.log(r2.clip(lower=FLOOR))
        g["lrv_w"] = np.log(r2.rolling(5).mean().clip(lower=FLOOR))
        g["lrv_m"] = np.log(rv21.clip(lower=FLOOR))
        g["lrv_q"] = np.log(r2.rolling(63).mean().clip(lower=FLOOR))
        g["lvix2"] = np.log(g["c_vix_lvl"] ** 2 / 252)
        g["y_vol"] = np.log(rv21.shift(-H).clip(lower=FLOOR))
        g["label_end"] = g["date"].shift(-H)
        parts.append(g)
    p = pd.concat(parts, ignore_index=True)
    p = p.dropna(subset=HAR + ["lvix2"]).reset_index(drop=True)

    # purge: a train/val label may not end after its own split boundary
    end = p["label_end"]
    p["vsplit"] = np.where((p["split"] == "train") & (end <= C.TRAIN_END), "train",
                  np.where((p["split"] == "val") & (end <= C.VAL_END), "val",
                  np.where((p["split"] == "test") & p["y_vol"].notna(), "test", "purge")))
    return p


def ols(X, y):
    Xc = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    return lambda Z: b[0] + Z @ b[1:]


def qlike(y, f):
    r = np.exp(y - f)
    return r - np.log(r) - 1


def dm_dates(loss_a, loss_b, dates, lags):
    d = pd.Series(loss_a - loss_b).groupby(dates).mean().to_numpy()
    n = len(d)
    dc = d - d.mean()
    v = dc @ dc / n + sum(2 * (1 - l / (lags + 1)) * (dc[l:] @ dc[:-l]) / n for l in range(1, lags + 1))
    s = d.mean() / np.sqrt(v / n)
    return float(s), float(2 * stats.norm.sf(abs(s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--skip-lstm", action="store_true")
    a = ap.parse_args()
    t0 = time.time()

    p = vol_panel()
    fcols = [c for c in p.columns if c.startswith(("f_", "c_"))]
    xcols = list(dict.fromkeys(fcols + HAR + ["lvix2"]))
    tr, va = p[p.vsplit == "train"], p[p.vsplit == "val"]
    test_days = np.sort(p.loc[p.vsplit == "test", "date"].unique())[::STEP]
    te = p[(p.vsplit == "test") & p["date"].isin(test_days)].sort_values(["ticker", "date"])
    print(f"vol panel: train {len(tr):,} | val {len(va):,} | test origins {len(te):,} "
          f"({te['date'].nunique()} days x {te['ticker'].nunique()} tickers)")

    preds = {"naive": te["lrv_m"].to_numpy()}
    trva = pd.concat([tr, va])
    preds["har"] = ols(trva[HAR].to_numpy(), trva["y_vol"].to_numpy())(te[HAR].to_numpy())
    hx = HAR + ["lvix2"]
    preds["har_x"] = ols(trva[hx].to_numpy(), trva["y_vol"].to_numpy())(te[hx].to_numpy())

    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=31, min_child_samples=200,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=C.SEED, verbose=-1)
    m.fit(tr[xcols], tr["y_vol"], eval_set=[(va[xcols], va["y_vol"])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    preds["lightgbm"] = m.predict(te[xcols])
    print(f"  har / lightgbm        {time.time() - t0:6.0f}s", flush=True)

    if not a.skip_lstm:
        from lstm_model import train_lstm
        lp = p.copy()
        lp["target_ret_1d"] = lp["y_vol"]
        lp["split"] = lp["vsplit"]
        out, _ = train_lstm(lp, xcols, C.LSTM_CFG_QUICK, tag="foundation/lstm_vol")
        lt = out["test"]
        lt["date"] = pd.to_datetime(lt["date"])
        preds["lstm"] = te[["date", "ticker"]].merge(lt, on=["date", "ticker"], how="left")["y_pred"].to_numpy()
        print(f"  lstm                  {time.time() - t0:6.0f}s", flush=True)

    # zero-shot: each ticker's trailing log RV, ending at the origin
    ctx, cov = [], []
    for tkr, g in p.groupby("ticker", sort=True):
        pos = pd.Series(np.arange(len(g)), index=g["date"].values)
        s, v = g["lrv_m"].to_numpy(np.float32), g["lvix2"].to_numpy(np.float32)
        for d in te.loc[te["ticker"] == tkr, "date"]:
            j = pos[d]
            lo = max(0, j - a.ctx + 1)
            ctx.append(s[lo:j + 1])
            cov.append(v[None, lo:j + 1])
    for model, variants in [(TimesFM25(max_context=a.ctx, max_horizon=32), [False]),
                            (Chronos2(), [False, True]),
                            (TimesFM3(), [False, True])]:
        for use in variants:
            name = model.name.lower().replace("-", "_").replace(".", "p") + ("_vix" if use else "")
            preds[name] = model.predict(ctx, H, covariates=cov if use else None)
            print(f"  {name:20s}  {time.time() - t0:6.0f}s", flush=True)
        del model

    fms = [k for k in preds if k.startswith(("timesfm", "chronos"))]
    preds["avg_harx_lgbm_fm_vix"] = np.mean([preds["har_x"], preds["lightgbm"],
                                             preds["timesfm_3_vix"]], axis=0)
    preds["avg_foundation"] = np.mean([preds[k] for k in fms], axis=0)
    preds["avg_all"] = np.mean([preds[k] for k in preds if k not in ("naive",) and not k.startswith("avg")], axis=0)

    y = te["y_vol"].to_numpy()
    dates = te["date"].to_numpy()
    regime = te["regime_highvol"].to_numpy()
    lag = int(np.ceil(H / STEP))
    se_har = (preds["har"] - y) ** 2
    rows, long = [], []
    for name, f in preds.items():
        ok = np.isfinite(f)
        e2 = (f - y) ** 2
        dm, pv = dm_dates(e2[ok], se_har[ok], dates[ok], lag) if name != "har" else (np.nan, np.nan)
        df = pd.DataFrame({"date": dates, "y": y, "f": f})[ok]
        ic = df.groupby("date").apply(lambda g: stats.spearmanr(g["f"], g["y"])[0], include_groups=False)
        row = dict(model=name, n_obs=int(ok.sum()), rmse=float(np.sqrt(e2[ok].mean())),
                   rel_rmse_har=float(np.sqrt(e2[ok].mean() / se_har[ok].mean())),
                   qlike=float(qlike(y[ok], f[ok]).mean()), dm_vs_har=dm, dm_p=pv,
                   xs_ic=float(ic.mean()), xs_ic_t=float(ic.mean() / ic.std() * np.sqrt(len(ic))))
        for lab, mask in [("highvol", regime == 1), ("normal", regime == 0)]:
            mk = ok & mask
            row[f"rel_rmse_har_{lab}"] = float(np.sqrt(e2[mk].mean() / se_har[mk].mean()))
        rows.append(row)
        long.append(pd.DataFrame({"date": dates, "ticker": te["ticker"].to_numpy(),
                                  "model": name, "y_true": y, "y_pred": f}))

    res = pd.DataFrame(rows).sort_values("rmse")
    res.to_csv(OUT / "vol_results.csv", index=False)
    pd.concat(long).to_parquet(OUT / "vol_predictions.parquet", index=False)
    print("\n==== REALISED VOL, TEST (21-day log RV) ====")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nhigh-vol share of origins: {regime.mean():.2%}")
    print(f"results -> {OUT / 'vol_results.csv'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
