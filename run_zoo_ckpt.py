"""Checkpointed horse-race runner: one model at a time, resume-safe."""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as C
from models_zoo import get_models

CKPT = C.RESULTS / "ckpt"
CKPT.mkdir(exist_ok=True)


def metric_row(df):
    e = df["y_pred"] - df["y_true"]
    nz = df["y_pred"] != 0
    dir_acc = float((np.sign(df["y_pred"]) == np.sign(df["y_true"]))[nz].mean()) if nz.any() else np.nan
    ic = (df.groupby("date")
          .apply(lambda g: stats.spearmanr(g["y_pred"], g["y_true"])[0]
                 if g["y_pred"].nunique() > 1 and len(g) > 4 else np.nan,
                 include_groups=False).dropna())
    t = float(ic.mean() / (ic.std() / np.sqrt(len(ic)))) if len(ic) > 2 else np.nan
    return dict(n_obs=len(df), rmse=float(np.sqrt((e**2).mean())),
                mae=float(e.abs().mean()), dir_acc=dir_acc,
                ic_mean=float(ic.mean()) if len(ic) else np.nan, ic_tstat=t)


def main():
    panel = pd.read_parquet(C.PROC / "supervised_panel.parquet")
    fcols = [c for c in panel.columns if c.startswith(("f_", "c_"))]
    d = {s: panel[panel["split"] == s] for s in ("train", "val", "test")}
    Xtr = d["train"][fcols].to_numpy(np.float32)
    ytr = d["train"]["target_ret_1d"].to_numpy(np.float32)
    Xv = d["val"][fcols].to_numpy(np.float32)
    yv = d["val"]["target_ret_1d"].to_numpy(np.float32)

    for name, model in get_models(fcols).items():
        out = CKPT / f"{name}.json"
        if out.exists():
            print(f"skip {name} (done)")
            continue
        t0 = time.time()
        print(f"fit {name}...", flush=True)
        model.fit(Xtr, ytr, Xv, yv)
        rows = []
        for s in ("val", "test"):
            pdf = d[s][["date", "ticker"]].copy()
            pdf["y_true"] = d[s]["target_ret_1d"].to_numpy()
            pdf["y_pred"] = model.predict(d[s][fcols].to_numpy(np.float32))
            rows.append(dict(model=name, split=s,
                             train_seconds=round(time.time() - t0, 1),
                             **metric_row(pdf)))
            if s == "test":
                pdf.to_parquet(CKPT / f"preds_{name}.parquet", index=False)
        out.write_text(json.dumps(rows))
        print(f"  {name} done ({time.time()-t0:.0f}s)", flush=True)

    # shuffled-target sanity (ridge)
    out = CKPT / "SANITY_shuffled_ridge.json"
    if not out.exists():
        ridge = get_models(fcols)["ridge"]
        ysh = np.random.default_rng(C.SEED).permutation(ytr)
        ridge.fit(Xtr, ysh, Xv, yv)
        pdf = d["test"][["date", "ticker"]].copy()
        pdf["y_true"] = d["test"]["target_ret_1d"].to_numpy()
        pdf["y_pred"] = ridge.predict(d["test"][fcols].to_numpy(np.float32))
        r = dict(model="SANITY_shuffled_ridge", split="test", train_seconds=0,
                 **metric_row(pdf))
        out.write_text(json.dumps([r]))
        print(f"sanity IC {r['ic_mean']:+.4f} "
              f"({'OK' if abs(r['ic_mean']) < 0.02 else 'LEAKAGE ALARM'})")

    rows = []
    for f in sorted(CKPT.glob("*.json")):
        rows += json.loads(f.read_text())
    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS / "horse_race_results.csv", index=False)
    print("\nALL ZOO DONE")
    print(res[res.split == "test"].sort_values("ic_mean", ascending=False)
          .to_string(index=False))


if __name__ == "__main__":
    main()
