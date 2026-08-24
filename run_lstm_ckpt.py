"""Checkpointed LSTM training: resumes from last completed epoch.

Note: CFG below is a deliberately compact development configuration and is
NOT config.LSTM_CFG (which is 2 layers x 64 hidden, dropout 0.2, patience 5).
Outputs land in results/ckpt/ and are superseded development metrics; see
results/ckpt/README.md for how they relate to the canonical table."""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config as C
from lstm_model import SeqDataset, LSTMForecaster, _epoch
from run_zoo_ckpt import metric_row, CKPT

CFG = dict(seq_len=30, hidden=32, layers=1, dropout=0.1,
           lr=1e-3, batch_size=2048, max_epochs=8, patience=2)
STATE = CKPT / "lstm_state.pt"
torch.manual_seed(C.SEED)
np.random.seed(C.SEED)


def main():
    panel = pd.read_parquet(C.PROC / "supervised_panel.parquet")
    fcols = [c for c in panel.columns if c.startswith(("f_", "c_"))]
    tr = panel["split"] == "train"
    mu = panel.loc[tr, fcols].mean().to_numpy(np.float32)
    sd = panel.loc[tr, fcols].std().replace(0, 1).to_numpy(np.float32)
    y_mu = float(panel.loc[tr, "target_ret_1d"].mean())
    y_sd = float(panel.loc[tr, "target_ret_1d"].std())

    print("building sequence datasets...", flush=True)
    ds = {s: SeqDataset(panel, fcols, CFG["seq_len"], s, mu, sd, y_mu, y_sd)
          for s in ("train", "val", "test")}
    dl = {s: DataLoader(ds[s], batch_size=CFG["batch_size"],
                        shuffle=(s == "train")) for s in ds}
    print({s: len(ds[s]) for s in ds}, "sequences", flush=True)

    model = LSTMForecaster(len(fcols), CFG["hidden"], CFG["layers"], CFG["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    start_ep, best_val, patience, hist, best_state = 1, np.inf, 0, [], None

    if STATE.exists():
        ck = torch.load(STATE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep = ck["epoch"] + 1
        best_val, patience, hist = ck["best_val"], ck["patience"], ck["hist"]
        best_state = ck["best_state"]
        print(f"resumed at epoch {start_ep} (best_val {best_val:.4f})", flush=True)

    stopped = False
    for ep in range(start_ep, CFG["max_epochs"] + 1):
        t0 = time.time()
        tr_loss, _ = _epoch(model, dl["train"], opt)
        val_loss, _ = _epoch(model, dl["val"])
        hist.append((ep, tr_loss, val_loss))
        if val_loss < best_val - 1e-5:
            best_val, patience = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        epoch=ep, best_val=best_val, patience=patience,
                        hist=hist, best_state=best_state), STATE)
        print(f"epoch {ep} train {tr_loss:.4f} val {val_loss:.4f} "
              f"pat {patience} ({time.time()-t0:.0f}s)", flush=True)
        if patience >= CFG["patience"]:
            print("early stop", flush=True)
            stopped = True
            break
    if not stopped and len(hist) < CFG["max_epochs"]:
        return  # interrupted; will resume next call

    if best_state:
        model.load_state_dict(best_state)
    rows = []
    for s in ("val", "test"):
        _, preds = _epoch(model, dl[s])
        pdf = ds[s].frame(preds, y_sd, y_mu)
        rows.append(dict(model="lstm", split=s, train_seconds=np.nan,
                         **metric_row(pdf)))
        if s == "test":
            pdf.to_parquet(CKPT / "preds_lstm.parquet", index=False)
    (CKPT / "lstm.json").write_text(json.dumps(rows))
    pd.DataFrame(hist, columns=["epoch", "train_mse", "val_mse"]).to_csv(
        C.RESULTS / "lstm_bbg_training_curve.csv", index=False)
    print("LSTM DONE")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
