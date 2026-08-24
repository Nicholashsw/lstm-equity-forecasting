"""Classical model zoo (stage V2 horse race). Uniform fit/predict wrappers.

Val set is used only for hyperparameter selection / early stopping.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR

SEED = 42


class Wrap:
    """sklearn estimator with the uniform 4-arg fit signature."""

    def __init__(self, est):
        self.est = est

    def fit(self, Xtr, ytr, Xval, yval):
        self.est.fit(Xtr, ytr)
        return self

    def predict(self, X):
        return self.est.predict(X)


class NaiveZero:
    def fit(self, *a):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class NaiveMomentum:
    """Predict yesterday's return (feature f_ret_1d passed at known index)."""

    def __init__(self, ret_idx):
        self.i = ret_idx

    def fit(self, *a):
        return self

    def predict(self, X):
        return np.asarray(X)[:, self.i]


class GridWrap:
    """Pick best alpha/depth on val RMSE, keep the winner."""

    def __init__(self, builder, grid):
        self.builder, self.grid = builder, grid
        self.best = None

    def fit(self, Xtr, ytr, Xval, yval):
        best_rmse = np.inf
        for p in self.grid:
            m = self.builder(p).fit(Xtr, ytr)
            rmse = float(np.sqrt(np.mean((m.predict(Xval) - yval) ** 2)))
            if rmse < best_rmse:
                best_rmse, self.best, self.best_p = rmse, m, p
        return self

    def predict(self, X):
        return self.best.predict(X)


class XGBWrap:
    def fit(self, Xtr, ytr, Xval, yval):
        import xgboost as xgb
        self.m = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            random_state=SEED, early_stopping_rounds=30, n_jobs=-1)
        self.m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        return self

    def predict(self, X):
        return self.m.predict(X)


class LGBMWrap:
    def fit(self, Xtr, ytr, Xval, yval):
        import lightgbm as lgb
        self.m = lgb.LGBMRegressor(
            n_estimators=600, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED,
            n_jobs=-1, verbose=-1)
        self.m.fit(Xtr, ytr, eval_set=[(Xval, yval)],
                   callbacks=[lgb.early_stopping(30, verbose=False)])
        return self

    def predict(self, X):
        return self.m.predict(X)


class SubsampleWrap(Wrap):
    """Fit on a random train subsample (for O(n^2)-ish models)."""

    def __init__(self, est, n_max=50_000):
        super().__init__(est)
        self.n_max = n_max

    def fit(self, Xtr, ytr, Xval, yval):
        rng = np.random.default_rng(SEED)
        if len(Xtr) > self.n_max:
            idx = rng.choice(len(Xtr), self.n_max, replace=False)
            Xtr, ytr = np.asarray(Xtr)[idx], np.asarray(ytr)[idx]
        self.est.fit(Xtr, ytr)
        return self


class PolyRidgeWrap:
    """Polynomial(2) Ridge on the first n_base features (explosion guard)."""

    def __init__(self, n_base=12, alpha=10.0):
        self.n_base, self.alpha = n_base, alpha

    def fit(self, Xtr, ytr, Xval, yval):
        self.pipe = make_pipeline(
            StandardScaler(), PolynomialFeatures(2, include_bias=False),
            Ridge(alpha=self.alpha, random_state=SEED))
        X = np.asarray(Xtr)[:, :self.n_base]
        if len(X) > 400_000:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(len(X), 400_000, replace=False)
            X, ytr = X[idx], np.asarray(ytr)[idx]
        self.pipe.fit(X, ytr)
        return self

    def predict(self, X):
        return self.pipe.predict(np.asarray(X)[:, :self.n_base])


def get_models(feature_names):
    lin = lambda est: make_pipeline(StandardScaler(), est)  # noqa: E731
    ret_idx = feature_names.index("f_ret_1d")
    return {
        "naive_zero": NaiveZero(),
        "naive_momentum": NaiveMomentum(ret_idx),
        "linear": Wrap(lin(LinearRegression())),
        "ridge": GridWrap(lambda a: lin(Ridge(alpha=a, random_state=SEED)),
                          [0.1, 1.0, 10.0]),
        "elasticnet": GridWrap(
            lambda a: lin(ElasticNet(alpha=a, l1_ratio=0.5,
                                     random_state=SEED, max_iter=5000)),
            [1e-4, 1e-3, 1e-2]),
        "decision_tree": GridWrap(
            lambda d: DecisionTreeRegressor(max_depth=d, random_state=SEED),
            [3, 5, 8]),
        "random_forest": Wrap(RandomForestRegressor(
            n_estimators=60, max_depth=8, n_jobs=-1, random_state=SEED, max_samples=0.25)),
        "xgboost": XGBWrap(),
        "lightgbm": LGBMWrap(),
        "svr_linear": SubsampleWrap(lin(LinearSVR(C=0.1, max_iter=5000,
                                                  random_state=SEED))),
        "poly2_ridge": PolyRidgeWrap(),
    }
