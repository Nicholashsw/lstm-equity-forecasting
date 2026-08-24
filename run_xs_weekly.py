"""Task 3b-weekly: 5-day-horizon cross-sectional long-short, weekly rebalance.

Cuts turnover ~5x vs daily. Target = target_xs_5d (5d forward XS-demeaned).
Rebalance every 5th trading day; hold the decile book for the week; realized
weekly return = compounded simple daily returns of held names over the holding
window. Costs 10 bps/side on one-way turnover at each rebalance.

Overlap note: we sample NON-OVERLAPPING weeks for both fitting-signal eval and
the portfolio, so weekly returns are independent (no overlapping-window t-stat
inflation). Stated explicitly in output.
"""
import sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, "src")
import config as C
from models_zoo import get_models

warnings.filterwarnings("ignore")
COST_PER_SIDE = 0.0010
WK = 52  # weekly periods per year for annualization

p = pd.read_parquet(C.PROC / "supervised_panel.parquet")
feats = [c for c in p.columns if c.startswith(("f_", "c_", "xr_"))]
regime_by_date = (p[["date", "regime_highvol"]].drop_duplicates("date")
                  .set_index("date")["regime_highvol"])

# per-ticker simple daily returns indexed by date, for compounding holding-period P&L
ret_wide = (p.pivot_table(index="date", columns="ticker", values="next_simple_1d")
            .sort_index())
all_dates = ret_wide.index.to_numpy()
# rebalance dates: every 5th trading day within the test window
test_dates = np.sort(p.loc[p.split == "test", "date"].unique())
rebal_dates = test_dates[::5]


def fwd_week_return(entry_date, tickers):
    """Compounded simple return over the 5 trading days AFTER entry_date."""
    pos = np.searchsorted(all_dates, entry_date)
    window = all_dates[pos + 1: pos + 6]
    if len(window) == 0:
        return pd.Series(dtype=float)
    sub = ret_wide.loc[window, [t for t in tickers if t in ret_wide.columns]]
    return (1 + sub.fillna(0)).prod() - 1


def run_model(name):
    d = {s: p[p.split == s] for s in ("train", "val", "test")}
    m = get_models(feats)[name]
    t0 = time.time()
    m.fit(d["train"][feats].to_numpy(), d["train"]["target_xs_5d"].fillna(0).to_numpy(),
          d["val"][feats].to_numpy(), d["val"]["target_xs_5d"].fillna(0).to_numpy())
    test = d["test"][["date", "ticker"]].copy()
    test["pred"] = m.predict(d["test"][feats].to_numpy())
    test["y"] = d["test"]["target_xs_5d"].to_numpy()
    print(f"{name}: fit {time.time()-t0:.0f}s")

    prev_w = {}
    recs = []
    for rd in rebal_dates:
        g = test[test.date == rd]
        if len(g) < 20 or g.pred.nunique() < 10:
            continue
        n_side = max(1, int(len(g) * 0.1))
        s = g.sort_values("pred")
        short, long = s.iloc[:n_side], s.iloc[-n_side:]
        wk = fwd_week_return(rd, list(long.ticker) + list(short.ticker))
        if wk.empty:
            continue
        gross = wk.reindex(long.ticker).mean() - wk.reindex(short.ticker).mean()
        w = {t: 1.0 / n_side for t in long.ticker}
        for t in short.ticker:
            w[t] = w.get(t, 0.0) - 1.0 / n_side
        names = set(w) | set(prev_w)
        turn = sum(abs(w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in names)
        cost = turn * COST_PER_SIDE
        # XS IC at this rebalance (independent weekly obs)
        ic = stats.spearmanr(g.pred, g.y, nan_policy="omit")[0]
        recs.append((rd, gross, gross - cost, turn, ic,
                     int(regime_by_date.get(rd, 0))))
        prev_w = w
    r = pd.DataFrame(recs, columns=["date", "gross", "net", "turnover",
                                    "ic", "regime"]).set_index("date")
    return r


def perf(x):
    if len(x) < 4:
        return dict(ann_ret=np.nan, sharpe=np.nan, maxdd=np.nan)
    mu, sd = x.mean(), x.std()
    sharpe = (mu / sd) * np.sqrt(WK) if sd > 0 else np.nan
    eq = x.cumsum()
    return dict(ann_ret=mu * WK, sharpe=sharpe, maxdd=(eq - eq.cummax()).min())


rows, curves = [], {}
for nm in ("ridge", "lightgbm"):
    r = run_model(nm)
    curves[nm] = r
    for reg, lab in [(None, "all"), (1, "highvol"), (0, "normal")]:
        sub = r if reg is None else r[r.regime == reg]
        if len(sub) < 4:
            rows.append(dict(model=nm, regime=lab, n_weeks=len(sub)))
            continue
        icm = sub["ic"].mean()
        ict = icm / (sub["ic"].std() / np.sqrt(len(sub)))
        g, n = perf(sub["gross"]), perf(sub["net"])
        rows.append(dict(model=nm, regime=lab, n_weeks=len(sub),
                         ic_mean=round(icm, 4), ic_t=round(ict, 2),
                         ls_ann_gross=round(g["ann_ret"], 3),
                         sharpe_gross=round(g["sharpe"], 2),
                         sharpe_net=round(n["sharpe"], 2),
                         maxdd_net=round(n["maxdd"], 3),
                         avg_turnover=round(sub["turnover"].mean(), 2)))

res = pd.DataFrame(rows)
res.to_csv(C.RESULTS / "xs_weekly_results.csv", index=False)
print("\n==== 5D WEEKLY-REBALANCE CROSS-SECTIONAL L/S (non-overlapping weeks) ====")
print(res.to_string(index=False))

r = curves["lightgbm"]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(r.index, r["gross"].cumsum(), label="gross", lw=1.4)
ax.plot(r.index, r["net"].cumsum(), label="net of 10bps/side", lw=1.4)
d = r.index.to_numpy()
hv = r["regime"].to_numpy() == 1
i = 0
while i < len(hv):
    if hv[i]:
        j = i
        while j < len(hv) and hv[j]:
            j += 1
        ax.axvspan(d[i], d[min(j, len(d) - 1)], color="#c0392b", alpha=0.10)
        i = j
    else:
        i += 1
ax.axhline(0, color="k", lw=0.7)
ax.set_title("LightGBM 5d weekly-rebalance cross-sectional L/S (test 2023+, high-vol shaded)")
ax.set_ylabel("cumulative weekly return")
ax.legend()
fig.tight_layout()
fig.savefig(C.FIGS / "xs_weekly_equity_curve.png", dpi=150)
print("figure saved | rebalances:", len(rebal_dates))
