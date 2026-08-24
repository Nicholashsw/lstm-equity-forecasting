"""Task 3b: regime-conditional cross-sectional long-short.

Target: next-day cross-sectionally demeaned return (dollar-neutral relative).
Models: ridge, lightgbm (with degenerate-prediction guard).
Eval by regime (all/highvol/normal): daily XS IC + decile long-short portfolio
with annualized return, Sharpe, max drawdown, turnover, and cost-adjusted Sharpe.
Costs: 10 bps per side, charged on one-way turnover (sum|dw|) each day.
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
COST_PER_SIDE = 0.0010  # 10 bps
ANN = 252

p = pd.read_parquet(C.PROC / "supervised_panel.parquet")
feats = [c for c in p.columns if c.startswith(("f_", "c_", "xr_"))]
regime_by_date = p[["date", "regime_highvol"]].drop_duplicates("date").set_index("date")["regime_highvol"]


def daily_ic(df):
    ic = df.groupby("date").apply(
        lambda g: stats.spearmanr(g.pred, g.y)[0]
        if g.pred.nunique() > 1 and len(g) > 4 else np.nan,
        include_groups=False).dropna()
    t = ic.mean() / (ic.std() / np.sqrt(len(ic))) if len(ic) > 2 else np.nan
    return ic, ic.mean(), t


def long_short(df, decile=0.1):
    """Daily equal-weight top-vs-bottom decile L/S. Returns per-day gross ret,
    net ret, and turnover. Realized ret uses simple next-day returns."""
    days = sorted(df.date.unique())
    prev_w = {}
    recs = []
    for d in days:
        g = df[df.date == d]
        if len(g) < 20 or g.pred.nunique() < 10:
            continue
        n_side = max(1, int(len(g) * decile))
        s = g.sort_values("pred")
        short = s.iloc[:n_side]
        long = s.iloc[-n_side:]
        w = {}
        for t in long.ticker:
            w[t] = 1.0 / n_side
        for t in short.ticker:
            w[t] = w.get(t, 0.0) - 1.0 / n_side
        gross = ((long["ret_simple"].mean()) - (short["ret_simple"].mean()))
        # one-way turnover = sum |w_t - w_{t-1}| across union of names
        names = set(w) | set(prev_w)
        turn = sum(abs(w.get(t, 0.0) - prev_w.get(t, 0.0)) for t in names)
        cost = turn * COST_PER_SIDE
        recs.append((d, gross, gross - cost, turn))
        prev_w = w
    r = pd.DataFrame(recs, columns=["date", "gross", "net", "turnover"]).set_index("date")
    return r


def perf(series):
    if len(series) < 5:
        return dict(ann_ret=np.nan, sharpe=np.nan, maxdd=np.nan)
    mu, sd = series.mean(), series.std()
    sharpe = (mu / sd) * np.sqrt(ANN) if sd > 0 else np.nan
    eq = series.cumsum()
    maxdd = (eq - eq.cummax()).min()
    return dict(ann_ret=mu * ANN, sharpe=sharpe, maxdd=maxdd)


def run_model(name):
    d = {s: p[p.split == s] for s in ("train", "val", "test")}
    Xtr = d["train"][feats].to_numpy()
    ytr = d["train"]["target_xs_1d"].to_numpy()
    Xv = d["val"][feats].to_numpy()
    yv = d["val"]["target_xs_1d"].to_numpy()
    m = get_models(feats)[name]
    t0 = time.time()
    m.fit(Xtr, ytr, Xv, yv)
    test = d["test"][["date", "ticker", "next_simple_1d"]].copy()
    test = test.rename(columns={"next_simple_1d": "ret_simple"})
    test["y"] = d["test"]["target_xs_1d"].to_numpy()
    test["pred"] = m.predict(d["test"][feats].to_numpy())
    test["regime"] = test["date"].map(regime_by_date)
    print(f"{name}: fit {time.time()-t0:.0f}s, test rows {len(test):,}")

    rows, curves = [], {}
    for reg, label in [(None, "all"), (1, "highvol"), (0, "normal")]:
        sub = test if reg is None else test[test.regime == reg]
        _, icm, ict = daily_ic(sub)
        ls = long_short(sub)
        g, n = perf(ls["gross"]), perf(ls["net"])
        excl = (sub.groupby("date").pred.nunique() < 10).sum()
        rows.append(dict(model=name, regime=label, n_days=sub.date.nunique(),
                         ic_mean=round(icm, 4), ic_t=round(ict, 2),
                         ls_ann_ret_gross=round(g["ann_ret"], 3),
                         ls_sharpe_gross=round(g["sharpe"], 2),
                         ls_sharpe_net=round(n["sharpe"], 2),
                         ls_maxdd_net=round(n["maxdd"], 3),
                         avg_turnover=round(ls["turnover"].mean(), 2),
                         degenerate_days=int(excl)))
        if reg is None:
            curves[name] = (ls, test)
    return rows, curves


all_rows, all_curves = [], {}
for nm in ("ridge", "lightgbm"):
    r, c = run_model(nm)
    all_rows += r
    all_curves.update(c)

res = pd.DataFrame(all_rows)
res.to_csv(C.RESULTS / "xs_regime_results.csv", index=False)
print("\n==== CROSS-SECTIONAL LONG-SHORT BY REGIME ====")
print(res.to_string(index=False))

# equity curve figure for ridge, high-vol shaded
ls, test = all_curves["ridge"]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ls.index, ls["gross"].cumsum(), label="gross", lw=1.3)
ax.plot(ls.index, ls["net"].cumsum(), label="net of 10bps/side", lw=1.3)
reg = test.drop_duplicates("date").set_index("date")["regime"].reindex(ls.index).fillna(0)
in_hv = reg.values == 1
d = ls.index.values
i = 0
while i < len(in_hv):
    if in_hv[i]:
        j = i
        while j < len(in_hv) and in_hv[j]:
            j += 1
        ax.axvspan(d[i], d[min(j, len(d)-1)], color="#c0392b", alpha=0.08)
        i = j
    else:
        i += 1
ax.axhline(0, color="k", lw=0.7)
ax.set_title("Ridge cross-sectional long-short (test 2023+, high-vol shaded)")
ax.set_ylabel("cumulative daily return")
ax.legend()
fig.tight_layout()
fig.savefig(C.FIGS / "xs_ls_equity_curve.png", dpi=150)
print("figure saved")
