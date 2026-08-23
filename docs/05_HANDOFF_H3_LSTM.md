# H3 Stage Work Order: LSTM Model

## Context
Core deliverable of the UROP title. PyTorch CPU. Consumes the same panel as H2 plus sequence construction.

## Design (implement exactly, then iterate)
- **Sequences:** per ticker, sliding window length L=30 trading days over feature vector; label = target at window end. Pooled across tickers (panel model, shared weights) — NOT one model per ticker.
- **Architecture v1:** LSTM(input=F, hidden=64, layers=2, dropout=0.2) -> last hidden -> Linear(64->1). Loss: MSE on standardized target (scaler fit on train only, inverse-transform predictions).
- **Training:** Adam lr 1e-3, batch 512, max 30 epochs, early stopping on val RMSE patience 5, grad clip 1.0, seed 42.
- **Splits:** identical date boundaries as H2 (import from `src/config.py`, never redefine).

## Deliverables
1. `src/lstm_model.py`: `SeqDataset`, `LSTMForecaster(nn.Module)`, `train_lstm(panel, cfg) -> preds_df` returning `date,ticker,y_true,y_pred` on val and test.
2. Ablation hooks (flags, not new files): `--hidden`, `--layers`, `--seq-len`, `--use-context/no-context`.
3. Training curve PNG to `results/figures/`.

## Acceptance criteria
- No sequence crosses a split boundary (window entirely within its split for features; label side defines membership — a test label may use pre-test history for features, that is allowed and standard; document it).
- Same test rows as H2 predictions (inner-join check in harness must be >95% row overlap; report exact).
- Runs < 20 min CPU on V1A panel.

## Known trap
Daily returns are ~unpredictable; expect tiny R^2 (even negative). Do not "fix" this by leaking. Directional accuracy 51-54% and small positive IC is a realistic good outcome.
