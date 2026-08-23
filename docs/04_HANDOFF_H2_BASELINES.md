# H2 Stage Work Order: Classical Model Zoo

## Context
Horse-race baselines for stage V2. Read `00_MASTER_PLAN.md` section "Splits and leakage rules". You consume the feature panel produced by `build_dataset.py` (H1/H3 own that); assume columns: keys `date,ticker`, features `f_*`, target `target_ret_1d`.

## Deliverables
`src/models_zoo.py` exposing `get_models() -> dict[str, sklearn-like estimator]` containing exactly:
1. Naive-zero (predict 0.0; the anchor every table row is judged against)
2. Naive-momentum (predict yesterday's return)
3. LinearRegression
4. Ridge (alpha grid [0.1, 1, 10] chosen on val)
5. ElasticNet (small grid)
6. DecisionTree (max_depth grid [3,5,8])
7. RandomForest (n_estimators 300, max_depth 8, n_jobs=-1)
8. XGBoost (hist, depth 6, lr 0.05, 400 trees, early stop on val)
9. LightGBM (default-ish, early stop on val)
10. SVR (LinearSVR on subsampled train — full SVR is O(n^2), forbidden on panel size)
11. Polynomial(2)-Ridge on a capped feature subset (poly explosion guard: <= 12 base features)

## Acceptance criteria
- Every model wrapped so `fit(X_tr, y_tr, X_val, y_val)` / `predict(X)` works uniformly (thin wrapper class fine).
- Val set used ONLY for hyperparameter pick / early stopping. Never refit on val+train for the headline table (document this choice).
- StandardScaler inside a Pipeline for linear/SVR models; tree models unscaled.
- Deterministic seeds everywhere.

## Do NOT
- No shuffled CV. No target scaling that leaks test statistics. No model that cannot train in < 5 min on 100k rows CPU.
