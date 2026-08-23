"""Per-regime horse race: does market context help more in high-vol periods?"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
import config as C
from models_zoo import get_models
from scipy import stats

panel = pd.read_parquet(C.PROC / "supervised_panel.parquet")
fcols = [c for c in panel.columns if c.startswith(("f_", "c_"))]
price_only = [c for c in fcols if c.startswith("f_")]

def ic_stats(df):
    daily = (df.groupby("date").apply(
        lambda g: stats.spearmanr(g.y_pred, g.y_true)[0]
        if g.y_pred.nunique() > 1 and len(g) > 4 else np.nan,
        include_groups=False).dropna())
    t = daily.mean()/(daily.std()/np.sqrt(len(daily))) if len(daily) > 2 else np.nan
    dacc = float((np.sign(df.y_pred)==np.sign(df.y_true))[df.y_pred!=0].mean())
    return dict(n=len(df), dir_acc=dacc, ic=float(daily.mean()), ic_t=float(t))

def fit_predict(feats):
    d = {s: panel[panel.split==s] for s in ("train","val","test")}
    Xtr, ytr = d["train"][feats].to_numpy(), d["train"]["target_ret_1d"].to_numpy()
    Xv, yv = d["val"][feats].to_numpy(), d["val"]["target_ret_1d"].to_numpy()
    out = {}
    for name in ["ridge","lightgbm"]:
        m = get_models(feats)[name]  # feats-consistent index for momentum not used
        m.fit(Xtr, ytr, Xv, yv)
        te = d["test"][["date","ticker","regime_highvol"]].copy()
        te["y_true"] = d["test"]["target_ret_1d"].to_numpy()
        te["y_pred"] = m.predict(d["test"][feats].to_numpy())
        out[name] = te
    return out

rows = []
for label, feats in [("price_only", price_only), ("with_context", fcols)]:
    preds = fit_predict(feats)
    for model, te in preds.items():
        for reg, name in [(None,"all"),(1,"high_vol"),(0,"normal")]:
            sub = te if reg is None else te[te.regime_highvol==reg]
            s = ic_stats(sub)
            rows.append(dict(feature_set=label, model=model, regime=name, **s))
res = pd.DataFrame(rows)
res.to_csv(C.RESULTS / "regime_results.csv", index=False)
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
