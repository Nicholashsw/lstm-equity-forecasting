"""Foundation-model extension of the level horse race.

Adds zero-shot TimesFM 2.5, TimesFM 3 and Chronos-2, plus Prophet, to the
next-day return race and scores them with the same metrics() as run_all.py
on the same test rows.

Each forecast for (ticker, t) sees only that ticker's daily log returns
through close(t), the same information set as f_ret_1d. The "+ ctx"
variants also see VIX level and SPX 5-day return through t as past-only
covariates. Prophet is refit per ticker at each month start and forecasts
every day of that month (refitting daily is 56k Stan fits).

Run after the panel exists (run_all.py, or src/build_dataset.py):
  python run_foundation.py                  # all test days
  python run_foundation.py --stride 5       # every 5th test day, ~5x faster
Results -> results/foundation/level_results.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as C  # noqa: E402
from foundation import TimesFM25, TimesFM3, Chronos2, ProphetModel  # noqa: E402
from run_all import metrics  # noqa: E402

OUT = C.RESULTS / "foundation"
OUT.mkdir(parents=True, exist_ok=True)
COVS = ["c_vix_lvl", "c_spx_ret5"]


def load_panel():
    path = C.PROC / "supervised_panel.parquet"
    if not path.exists():
        from build_dataset import build
        return build(use_context=True)
    return pd.read_parquet(path)


def test_origins(panel, stride):
    test = panel[panel["split"] == "test"]
    days = np.sort(test["date"].unique())[::stride]
    return test[test["date"].isin(days)].sort_values(["ticker", "date"])


def contexts(panel, rows, ctx_len, with_cov):
    """One return window per test row, ending at the row's own date."""
    ctx, cov = [], []
    for tkr, g in panel.groupby("ticker", sort=True):
        g = g.sort_values("date")
        pos = pd.Series(np.arange(len(g)), index=g["date"].values)
        r = g["f_ret_1d"].to_numpy(np.float32)
        x = g[COVS].to_numpy(np.float32).T
        for d in rows.loc[rows["ticker"] == tkr, "date"]:
            j = pos[d]
            lo = max(0, j - ctx_len + 1)
            ctx.append(r[lo:j + 1])
            if with_cov:
                cov.append(x[:, lo:j + 1])
    return ctx, (cov if with_cov else None)


def prophet_preds(panel, rows):
    pr = ProphetModel("B")
    out = []
    for tkr, g in rows.groupby("ticker", sort=True):
        hist = panel.loc[panel["ticker"] == tkr, ["date", "f_ret_1d"]].sort_values("date")
        for _, block in g.groupby(g["date"].dt.to_period("M")):
            first = block["date"].min()
            s = hist[hist["date"] < first].set_index("date")["f_ret_1d"].iloc[-750:]
            fc = pr.forecast_path(s, len(block) + 25)
            # trading-day steps ahead of the refit date for each row in the month
            steps = hist["date"].searchsorted(block["date"].values) - hist["date"].searchsorted(first) + 1
            out.append(pd.Series(fc[steps - 1], index=block.index))
    return pd.concat(out).reindex(rows.index).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--skip-prophet", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    panel = load_panel()
    panel["date"] = pd.to_datetime(panel["date"])
    rows = test_origins(panel, a.stride)
    print(f"test rows: {len(rows):,} | tickers {rows['ticker'].nunique()} | "
          f"days {rows['date'].nunique()} | ctx {a.ctx}")

    base = rows[["date", "ticker"]].copy()
    base["y_true"] = rows["target_ret_1d"].to_numpy()
    preds = {}

    ctx, cov = contexts(panel, rows, a.ctx, with_cov=True)
    for model, variants in [(TimesFM25(max_context=a.ctx, max_horizon=16), [False]),
                            (Chronos2(), [False, True]),
                            (TimesFM3(), [False, True])]:
        for use in variants:
            name = model.name.lower().replace("-", "_").replace(".", "p") + ("_ctx" if use else "")
            preds[name] = model.predict(ctx, 1, covariates=cov if use else None)
            print(f"  {name:16s} {time.time() - t0:6.0f}s", flush=True)
        del model

    if not a.skip_prophet:
        preds["prophet"] = prophet_preds(panel, rows)
        print(f"  {'prophet':16s} {time.time() - t0:6.0f}s", flush=True)

    preds["naive_zero"] = np.zeros(len(rows))
    res, long = [], []
    for name, p in preds.items():
        df = base.copy()
        df["y_pred"] = p
        res.append(dict(model=name, **metrics(df)))
        long.append(df.assign(model=name))

    res = pd.DataFrame(res).sort_values("ic_mean", ascending=False)
    res.to_csv(OUT / "level_results.csv", index=False)
    pd.concat(long).to_parquet(OUT / "level_predictions.parquet", index=False)
    print("\n==== FOUNDATION MODELS, TEST ====")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nresults -> {OUT / 'level_results.csv'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
