"""Pooled-panel LSTM forecaster (stage V1A/V1B core model).

Sequences built lazily (per-ticker arrays + index math) so the full panel
never materializes as a 3D tensor. Split membership is defined by the label
date; feature windows may reach back into earlier history (standard, and
documented in H3).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

torch.manual_seed(C.SEED)
np.random.seed(C.SEED)


class SeqDataset(Dataset):
    def __init__(self, panel, fcols, seq_len, split, mu, sd, y_mu, y_sd):
        self.seq_len = seq_len
        self.X, self.y, self.meta, self.index = [], [], [], []
        for tid, (tkr, g) in enumerate(panel.groupby("ticker", sort=True)):
            g = g.sort_values("date")
            X = ((g[fcols].to_numpy(np.float32) - mu) / sd).astype(np.float32)
            y = ((g["target_ret_1d"].to_numpy(np.float32) - y_mu) / y_sd)
            splits = g["split"].to_numpy()
            dates = g["date"].to_numpy()
            self.X.append(X)
            self.y.append(y.astype(np.float32))
            self.meta.append((tkr, dates))
            ok = np.where(splits == split)[0]
            for j in ok:
                if j >= seq_len - 1:
                    self.index.append((tid, j))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        tid, j = self.index[k]
        x = self.X[tid][j - self.seq_len + 1: j + 1]
        return torch.from_numpy(x), torch.tensor(self.y[tid][j])

    def frame(self, preds, y_sd, y_mu):
        rows = []
        for (tid, j), p in zip(self.index, preds):
            tkr, dates = self.meta[tid]
            rows.append((dates[j], tkr, float(self.y[tid][j] * y_sd + y_mu),
                         float(p * y_sd + y_mu)))
        return pd.DataFrame(rows, columns=["date", "ticker", "y_true", "y_pred"])


class LSTMForecaster(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _epoch(model, loader, opt=None, loss_fn=nn.MSELoss()):
    training = opt is not None
    model.train(training)
    tot, n, preds = 0.0, 0, []
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            out = model(xb)
            loss = loss_fn(out, yb)
            if training:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            else:
                preds.append(out.numpy())
            tot += loss.item() * len(yb)
            n += len(yb)
    return tot / n, (np.concatenate(preds) if preds else None)


def train_lstm(panel, fcols, cfg, tag="lstm"):
    tr_mask = panel["split"] == "train"
    mu = panel.loc[tr_mask, fcols].mean().to_numpy(np.float32)
    sd = panel.loc[tr_mask, fcols].std().replace(0, 1).to_numpy(np.float32)
    y_mu = float(panel.loc[tr_mask, "target_ret_1d"].mean())
    y_sd = float(panel.loc[tr_mask, "target_ret_1d"].std())

    ds = {s: SeqDataset(panel, fcols, cfg["seq_len"], s, mu, sd, y_mu, y_sd)
          for s in ("train", "val", "test")}
    dl = {s: DataLoader(ds[s], batch_size=cfg["batch_size"],
                        shuffle=(s == "train"), num_workers=0)
          for s in ds}
    print({s: len(ds[s]) for s in ds}, "sequences")

    model = LSTMForecaster(len(fcols), cfg["hidden"], cfg["layers"],
                           cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    hist, best_val, best_state, patience = [], np.inf, None, 0
    for ep in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        tr_loss, _ = _epoch(model, dl["train"], opt)
        val_loss, _ = _epoch(model, dl["val"])
        hist.append((ep, tr_loss, val_loss))
        print(f"epoch {ep:02d} train {tr_loss:.4f} val {val_loss:.4f} "
              f"({time.time() - t0:.0f}s)")
        if val_loss < best_val - 1e-5:
            best_val, patience = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print("early stop")
                break
    if best_state:
        model.load_state_dict(best_state)

    out = {}
    for s in ("val", "test"):
        _, preds = _epoch(model, dl[s])
        out[s] = ds[s].frame(preds, y_sd, y_mu)
    pd.DataFrame(hist, columns=["epoch", "train_mse", "val_mse"]).to_csv(
        C.RESULTS / f"{tag}_training_curve.csv", index=False)
    return out, hist
