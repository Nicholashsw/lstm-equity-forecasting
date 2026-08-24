import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "src")
import config as C
from lstm_model import train_lstm
from scipy import stats

panel = pd.read_parquet(C.PROC / "supervised_panel.parquet")
fcols = [c for c in panel.columns if c.startswith(("f_", "c_"))]
cfg = dict(C.LSTM_CFG); cfg["max_epochs"] = 50   # early stopping is the real governor
print("FULL cfg:", cfg, "| features:", len(fcols))
t0 = time.time()
preds, hist = train_lstm(panel, fcols, cfg, tag="lstm_full")
def m(df):
    dacc = float((np.sign(df.y_pred)==np.sign(df.y_true))[df.y_pred!=0].mean())
    ic = df.groupby("date").apply(lambda g: stats.spearmanr(g.y_pred,g.y_true)[0]
        if g.y_pred.nunique()>1 and len(g)>4 else np.nan, include_groups=False).dropna()
    t = ic.mean()/(ic.std()/np.sqrt(len(ic)))
    return dacc, ic.mean(), t
for s in ("val","test"):
    d,i,t = m(preds[s]); print(f"{s}: dir_acc={d:.4f} ic={i:+.4f} ic_t={t:.2f}")
print(f"total {time.time()-t0:.0f}s, epochs run={len(hist)}")
