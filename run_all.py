"""End-to-end runner. Usage:
  python run_all.py --source yfinance --quick          # smoke/demo run
  python run_all.py --source bloomberg --skip-pull     # after Bloomberg CSVs parsed
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve() .parent / "src"))
import config as C  # noqa: E402
from build_dataset import build  # noqa: E402
from models_zoo import get_models  # noqa: E402
from lstm_model import train_lstm  # noqa: E402

np.random.seed(C.SEED)


def metrics(df):
    e = df["y_pred"] - df["y_true"]
    dir_acc = float((np.sign(df["y_pred"]) == np.sign(df["y_true"]))
                    [df["y_pred"] != 0].mean()) if (df["y_pred"] != 0).any() else np.nan
    daily_ic = (df.groupby("date")
                .apply(lambda g: stats.spearmanr(g["y_pred"], g["y_true"])[0]
                       if g["y_pred"].nunique() > 1 and len(g) > 4 else np.nan,
                       include_groups=False)
                .dropna())
    ic_t = (daily_ic.mean() / (daily_ic.std() / np.sqrt(len(daily_ic)))
            if len(daily_ic) > 2 else np.nan)
    return dict(n_obs=len(df),
                rmse=float(np.sqrt((e ** 2).mean())),
                mae=float(e.abs().mean()),
                dir_acc=dir_acc,
                ic_mean=float(daily_ic.mean()) if len(daily_ic) else np.nan,
                ic_tstat=float(ic_t) if ic_t == ic_t else np.nan)


def run_zoo(panel, fcols, shuffle_target=False):
    d = {s: panel[panel["split"] == s] for s in ("train", "val", "test")}
    Xtr, ytr = d["train"][fcols].to_numpy(), d["train"]["target_ret_1d"].to_numpy()
    Xv, yv = d["val"][fcols].to_numpy(), d["val"]["target_ret_1d"].to_numpy()
    if shuffle_target:
        ytr = np.random.default_rng(C.SEED).permutation(ytr)
    rows, preds_store = [], {}
    for name, model in get_models(fcols).items():
        t0 = time.time()
        model.fit(Xtr, ytr, Xv, yv)
        for s in ("val", "test"):
            pdf = d[s][["date", "ticker"]].copy()
            pdf["y_true"] = d[s]["target_ret_1d"].to_numpy()
            pdf["y_pred"] = model.predict(d[s][fcols].to_numpy())
            m = metrics(pdf)
            rows.append(dict(model=name, split=s,
                             train_seconds=round(time.time() - t0, 1), **m))
            if s == "test":
                preds_store[name] = pdf
        print(f"  {name:15s} done ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows), preds_store


def figures(res, lstm_hist, preds_store):
    test = res[res.split == "test"].set_index("model")
    order = test["ic_mean"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    order.plot.barh(ax=ax, color=["#c0392b" if v < 0 else "#2c6e49" for v in order])
    ax.set_xlabel("Mean daily cross-sectional IC (test)")
    ax.set_title("Horse race: test-set information coefficient")
    ax.axvline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(C.FIGS / "test_ic_by_model.png", dpi=150)
    plt.close(fig)

    if lstm_hist:
        h = pd.DataFrame(lstm_hist, columns=["epoch", "train", "val"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(h.epoch, h.train, label="train MSE (std target)")
        ax.plot(h.epoch, h.val, label="val MSE")
        ax.set_xlabel("epoch")
        ax.legend()
        ax.set_title("LSTM training curve")
        fig.tight_layout()
        fig.savefig(C.FIGS / "lstm_training_curve.png", dpi=150)
        plt.close(fig)

    # cumulative long-short decile spread proxy for the top-IC model
    best = test["ic_mean"].idxmax()
    pdf = preds_store.get(best)
    if pdf is not None:
        def spread(g):
            q = g["y_pred"].rank(pct=True)
            return g.loc[q >= 0.8, "y_true"].mean() - g.loc[q <= 0.2, "y_true"].mean()
        s = pdf.groupby("date").apply(spread, include_groups=False).dropna()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(pd.to_datetime(s.index), s.cumsum())
        ax.set_title(f"Cumulative top-bottom quintile spread ({best}, test, pre-cost)")
        ax.set_ylabel("cum log return")
        fig.tight_layout()
        fig.savefig(C.FIGS / "ls_spread_best_model.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yfinance", "bloomberg"], default="yfinance")
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-context", action="store_true")
    a = ap.parse_args()

    if not a.skip_pull:
        if a.source == "yfinance":
            from pull_yfinance import pull_equity_panel, pull_context
            pull_equity_panel(C.START_DATE)
            pull_context(C.START_DATE)
        else:
            from parse_bbg_csv import build as bbg_build
            bbg_build(C.RAW / "bloomberg")

    panel = build(use_context=not a.no_context)
    fcols = [c for c in panel.columns if c.startswith(("f_", "c_"))]

    print("== classical zoo ==")
    res, preds_store = run_zoo(panel, fcols)

    print("== sanity: shuffled-target leakage test (ridge only) ==")
    sh_panel = panel.copy()
    d = {s: sh_panel[sh_panel["split"] == s] for s in ("train", "val", "test")}
    from models_zoo import get_models as gm
    ridge = gm(fcols)["ridge"]
    ytr_sh = np.random.default_rng(C.SEED).permutation(
        d["train"]["target_ret_1d"].to_numpy())
    ridge.fit(d["train"][fcols].to_numpy(), ytr_sh,
              d["val"][fcols].to_numpy(), d["val"]["target_ret_1d"].to_numpy())
    sdf = d["test"][["date", "ticker"]].copy()
    sdf["y_true"] = d["test"]["target_ret_1d"].to_numpy()
    sdf["y_pred"] = ridge.predict(d["test"][fcols].to_numpy())
    sm = metrics(sdf)
    print(f"shuffled-target test IC = {sm['ic_mean']:+.4f} "
          f"({'OK' if abs(sm['ic_mean']) < 0.02 else 'LEAKAGE ALARM'})")
    res = pd.concat([res, pd.DataFrame([dict(model="SANITY_shuffled_ridge",
                                             split="test", train_seconds=0, **sm)])])

    print("== LSTM ==")
    cfg = C.LSTM_CFG_QUICK if a.quick else C.LSTM_CFG
    lstm_preds, hist = train_lstm(panel, fcols, cfg)
    for s in ("val", "test"):
        m = metrics(lstm_preds[s])
        res = pd.concat([res, pd.DataFrame([dict(model="lstm", split=s,
                                                 train_seconds=np.nan, **m)])])
    preds_store["lstm"] = lstm_preds["test"]
    lstm_preds["test"].to_csv(C.RESULTS / "lstm_test_predictions.csv", index=False)

    res = res.reset_index(drop=True)
    res.to_csv(C.RESULTS / "horse_race_results.csv", index=False)
    figures(res, hist, preds_store)

    show = res[res.split == "test"].sort_values("ic_mean", ascending=False)
    cols = ["model", "n_obs", "rmse", "mae", "dir_acc", "ic_mean", "ic_tstat"]
    print("\n==== TEST RESULTS ====")
    print(show[cols].to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))
    print(f"\nresults -> {C.RESULTS / 'horse_race_results.csv'}")


if __name__ == "__main__":
    main()
