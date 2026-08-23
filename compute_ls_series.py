import sys, numpy as np, pandas as pd
sys.path.insert(0,"src"); import config as C
from models_zoo import get_models
p = pd.read_parquet(C.PROC/"supervised_panel.parquet")
feats=[c for c in p.columns if c.startswith(("f_","c_","xr_"))]
d={s:p[p.split==s] for s in ("train","val","test")}
m=get_models(feats)["ridge"]
m.fit(d["train"][feats].to_numpy(), d["train"]["target_xs_1d"].to_numpy(),
      d["val"][feats].to_numpy(), d["val"]["target_xs_1d"].to_numpy())
t=d["test"][["date","ticker","next_simple_1d","regime_highvol"]].copy()
t["pred"]=m.predict(d["test"][feats].to_numpy())
rows=[]
for dt,g in t.groupby("date"):
    if len(g)<20 or g.pred.nunique()<10: continue
    n=max(1,int(len(g)*0.1)); s=g.sort_values("pred")
    ret=s.iloc[-n:]["next_simple_1d"].mean()-s.iloc[:n]["next_simple_1d"].mean()
    rows.append((dt,ret,int(g["regime_highvol"].iloc[0])))
ls=pd.DataFrame(rows,columns=["date","ret","regime"])
ls.to_csv(C.RESULTS/"ridge_ls_daily.csv",index=False)
print("saved",len(ls),"days")
