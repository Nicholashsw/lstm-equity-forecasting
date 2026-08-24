import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "src")
import config as C
from models_zoo import get_models
from scipy import stats

p = pd.read_parquet(C.PROC / "supervised_panel.parquet")
fcols = [c for c in p.columns if c.startswith(("f_", "c_"))]
fcols_noctx = [c for c in fcols if not c.startswith("c_")]

def ic_by_day(df):
    ic = (df.groupby("date").apply(
        lambda g: stats.spearmanr(g.y_pred, g.y_true)[0]
        if g.y_pred.nunique() > 1 and len(g) > 4 else np.nan,
        include_groups=False).dropna())
    t = ic.mean()/(ic.std()/np.sqrt(len(ic))) if len(ic) > 2 else np.nan
    return ic.mean(), t, len(ic)

def fit_eval(cols, tag):
    d = {s: p[p.split == s] for s in ("train","val","test")}
    Xtr, ytr = d["train"][cols].to_numpy(), d["train"]["target_ret_1d"].to_numpy()
    Xv, yv = d["val"][cols].to_numpy(), d["val"]["target_ret_1d"].to_numpy()
    # use ridge + lightgbm as representative linear/nonlinear learners
    rows = []
    for name in ("ridge","lightgbm"):
        m = get_models(cols)[name]
        t0=time.time(); m.fit(Xtr,ytr,Xv,yv)
        test = d["test"][["date","ticker"]].copy()
        test["y_true"]=d["test"]["target_ret_1d"].to_numpy()
        test["y_pred"]=m.predict(d["test"][cols].to_numpy())
        test = test.merge(p[["date","regime_highvol"]].drop_duplicates("date"), on="date", how="left")
        for reg,label in [(None,"all"),(1,"highvol"),(0,"normal")]:
            sub = test if reg is None else test[test.regime_highvol==reg]
            icm,ict,nd = ic_by_day(sub)
            rows.append(dict(feats=tag,model=name,regime=label,ic_mean=round(icm,4),
                             ic_t=round(ict,2),n_days=nd))
        print(f"{tag}/{name} done {time.time()-t0:.0f}s")
    return rows

out = []
out += fit_eval(fcols, "with_context")
out += fit_eval(fcols_noctx, "no_context")
res = pd.DataFrame(out)
res.to_csv(C.RESULTS/"regime_results.csv", index=False)
print(res.to_string(index=False))
