"""Every result number in README.md and paper/main.tex against the committed results files.

Reads only committed files. Run `pytest tests/test_claims.py` (add -s to print the table).
A claim is (doc, location, printed value, source file, field, actual value). The printed value
is compared at the precision it is printed with (half a unit in the last digit). A number with
no committed source is a failing check. test_every_number_is_covered fails on any number in
either document that is neither in a claim row nor in ALLOW with a reason.
"""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = {"README": ROOT / "README.md", "paper": ROOT / "paper" / "main.tex"}
_cache = {}


def csv(name):
    if name not in _cache:
        _cache[name] = pd.read_csv(ROOT / "results" / name)
    return _cache[name]


def pick(name, col, **where):
    df = csv(name)
    for k, v in where.items():
        df = df[df[k] == v]
    assert len(df) == 1, f"{name} {where}: {len(df)} rows"
    return float(df[col].iloc[0])


def hr(model, split, col):
    return pick("horse_race_results.csv", col, model=model, split=split)


def xs(model, regime, col):
    return pick("xs_regime_results.csv", col, model=model, regime=regime)


def wk(model, regime, col):
    return pick("xs_weekly_results.csv", col, model=model, regime=regime)


def rg(feats, model, regime, col):
    return pick("regime_results.csv", col, feats=feats, model=model, regime=regime)


def ra(label, col):
    return pick("risk_adjusted_results.csv", col, label=label)


def cnt(key):
    return pick("data_counts.csv", "value", key=key)


def ckpt(model, split, col):
    rows = json.loads((ROOT / "results/ckpt" / f"{model}.json").read_text().replace("NaN", "null"))
    return [r[col] for r in rows if r["split"] == split][0]


def curve(name, epoch, col="val_mse"):
    return pick(name, col, epoch=epoch)


def code(path, pattern):
    m = re.search(pattern, (ROOT / path).read_text())
    assert m, f"{path}: {pattern} not found"
    return float(m.group(1))


def cfg(name, key):
    text = (ROOT / "src/config.py").read_text()
    d = eval("dict(" + re.search(name + r" = dict\((.*?)\)\n", text, re.S).group(1) + ")")
    return d[key]


def n_rows(name):
    return len(csv(name))


C = []  # (doc, location, printed, source, field, actual)


def c(doc, loc, printed, source, field, actual):
    C.append((doc, loc, printed, source, field, actual))


HR = "results/horse_race_results.csv"
XS = "results/xs_regime_results.csv"
WK = "results/xs_weekly_results.csv"
RG = "results/regime_results.csv"
RA = "results/risk_adjusted_results.csv"
DC = "results/data_counts.csv"
CFG = "src/config.py"

# ---------------------------------------------------------------- README
R = "README"
c(R, "Data", "518", DC, "used_tickers", cnt("used_tickers"))
c(R, "Data", "501", DC, "stock_tickers", cnt("stock_tickers"))
c(R, "Data", "17", DC, "etf_tickers", cnt("etf_tickers"))
c(R, "Data", "520", DC, "workbook_tickers", cnt("workbook_tickers"))
c(R, "Data", "29", DC, "days_FDXF", cnt("days_FDXF"))
c(R, "Data", "16", DC, "days_HONA", cnt("days_HONA"))
c(R, "Data (parity median, 0 within print)", "0", "results/parity_report.csv", "median of d",
  round(float(pd.read_csv(ROOT / "results/parity_report.csv")["d"].median()), 6))
c(R, "Method split", "2021", CFG, "TRAIN_END year", code(CFG, r'TRAIN_END = "(\d{4})'))
c(R, "Method split", "2022", CFG, "VAL_END year", code(CFG, r'VAL_END = "(\d{4})'))
c(R, "Method split", "2023", CFG, "VAL_END year + 1", code(CFG, r'VAL_END = "(\d{4})') + 1)
c(R, "Method regime", "63", "src/build_dataset.py", "z window", code("src/build_dataset.py", r"def z\(s, w=(\d+)\)"))
c(R, "Method regime", "1", "src/build_dataset.py", "z threshold", code("src/build_dataset.py", r'\) > (\d+)\)\.astype'))
c(R, "Method trading", "10", "run_xs_regime.py", "COST_PER_SIDE bp", code("run_xs_regime.py", r"COST_PER_SIDE = ([\d.]+)") * 1e4)
c(R, "Method trading", "0.10", "run_xs_regime.py", "COST_PER_SIDE percent", code("run_xs_regime.py", r"COST_PER_SIDE = ([\d.]+)") * 100)
c(R, "Method trading", "252", "run_xs_regime.py", "ANN", code("run_xs_regime.py", r"ANN = (\d+)"))
c(R, "Method check", "0.02", "run_all.py", "shuffled |IC| limit", code("run_all.py", r"abs\(sm\['ic_mean'\]\) < ([\d.]+)"))
c(R, "Data (50-day features)", "50", "src/build_dataset.py", "longest window", code("src/build_dataset.py", r"rolling\((\d+)\)\.mean\(\) - 1\n\s+vol"))
c(R, "Main result", "466,426", HR, "naive_zero test n_obs", hr("naive_zero", "test", "n_obs"))
c(R, "Main result", "0.16", HR, "RMSE gain of xgboost over zero, %",
  100 * (1 - hr("xgboost", "test", "rmse") / hr("naive_zero", "test", "rmse")))
c(R, "Main result", "0.020379", HR, "xgboost test rmse", hr("xgboost", "test", "rmse"))
c(R, "Main result", "0.020411", HR, "naive_zero test rmse", hr("naive_zero", "test", "rmse"))
IC_ROWS = [("SVR", ["svr_linear"], "0.490", "0.014", "2.42"),
           ("Ridge / Linear", ["ridge", "linear"], "0.491", "0.011", "1.86"),
           ("LSTM", ["lstm"], "0.480", "0.004", "0.76"),
           ("Shuffled", ["SANITY_shuffled_ridge"], "0.498", "0.004", "0.67"),
           ("XGBoost", ["xgboost"], "0.497", "0.141", "1.83"),
           ("LightGBM", ["lightgbm"], "0.496", "-0.029", "-0.69")]
for name, models, da, ic, t in IC_ROWS:
    for m in models:
        c(R, f"IC table {name}", da, HR, f"{m} test dir_acc", hr(m, "test", "dir_acc"))
        c(R, f"IC table {name}", ic, HR, f"{m} test ic_mean", hr(m, "test", "ic_mean"))
        c(R, f"IC table {name}", t, HR, f"{m} test ic_tstat", hr(m, "test", "ic_tstat"))
c(R, "boosted-tree note", "7", RG, "with_context lightgbm all n_days", rg("with_context", "lightgbm", "all", "n_days"))
c(R, "boosted-tree note", "915", XS, "lightgbm all n_days", xs("lightgbm", "all", "n_days"))
c(R, "boosted-tree note", "168", HR, "t^2/ic^2 of xgboost test", hr("xgboost", "test", "ic_tstat") ** 2 / hr("xgboost", "test", "ic_mean") ** 2)
c(R, "boosted-tree note", "901", RG, "with_context ridge all n_days", rg("with_context", "ridge", "all", "n_days"))
c(R, "stress paragraph", "0.044", XS, "lightgbm highvol ic_mean", xs("lightgbm", "highvol", "ic_mean"))
c(R, "stress paragraph", "3.2", XS, "lightgbm highvol ic_t", xs("lightgbm", "highvol", "ic_t"))
c(R, "stress paragraph", "0.005", XS, "lightgbm normal ic_mean", xs("lightgbm", "normal", "ic_mean"))
c(R, "stress paragraph", "0.9", XS, "lightgbm normal ic_t", xs("lightgbm", "normal", "ic_t"))
c(R, "stress paragraph", "2.09", XS, "lightgbm highvol sharpe gross", xs("lightgbm", "highvol", "ls_sharpe_gross"))
XROWS = [("LightGBM high-vol", "lightgbm", "highvol", "183", "0.044", "3.17", "2.09", "-0.88", "2.7"),
         ("LightGBM normal", "lightgbm", "normal", "732", "0.005", "0.88", "-0.24", "-4.89", "2.6"),
         ("Ridge high-vol", "ridge", "highvol", None, "0.032", "1.87", "0.96", "-0.69", "2.1"),
         ("Ridge all", "ridge", "all", None, "0.015", "2.76", "0.48", "-1.99", "2.0")]
for name, m, reg, nd, ic, t, g, n, to in XROWS:
    if nd:
        c(R, f"regime table {name}", nd, XS, f"{m} {reg} n_days", xs(m, reg, "n_days"))
    c(R, f"regime table {name}", ic, XS, f"{m} {reg} ic_mean", xs(m, reg, "ic_mean"))
    c(R, f"regime table {name}", t, XS, f"{m} {reg} ic_t", xs(m, reg, "ic_t"))
    c(R, f"regime table {name}", g, XS, f"{m} {reg} ls_sharpe_gross", xs(m, reg, "ls_sharpe_gross"))
    c(R, f"regime table {name}", n, XS, f"{m} {reg} ls_sharpe_net", xs(m, reg, "ls_sharpe_net"))
    c(R, f"regime table {name}", to, XS, f"{m} {reg} avg_turnover", xs(m, reg, "avg_turnover"))
c(R, "regime note", "6", XS, "rows (regime-by-model cells)", n_rows("xs_regime_results.csv"))
c(R, "weekly", "0.0007", WK, "ridge all ic_mean", wk("ridge", "all", "ic_mean"))
c(R, "risk factors", "4", RA, "factor betas (mkt, smb, hml, mom)", len([x for x in csv("risk_adjusted_results.csv").columns if x.startswith("beta_")]))
c(R, "risk factors", "1.35", RA, "all alpha_t", ra("all", "alpha_t"))
c(R, "risk factors", "0.10", RA, "highvol alpha_t", ra("highvol", "alpha_t"))
c(R, "LSTM", "466,426", HR, "naive_zero test n_obs", hr("naive_zero", "test", "n_obs"))
c(R, "LSTM", "465,904", HR, "lstm test n_obs", hr("lstm", "test", "n_obs"))
c(R, "LSTM", "30", CFG, "LSTM_CFG_QUICK seq_len", cfg("LSTM_CFG_QUICK", "seq_len"))
c(R, "LSTM", "1", CFG, "LSTM_CFG_QUICK layers", cfg("LSTM_CFG_QUICK", "layers"))
c(R, "LSTM", "32", CFG, "LSTM_CFG_QUICK hidden", cfg("LSTM_CFG_QUICK", "hidden"))
c(R, "LSTM", "2", CFG, "LSTM_CFG_QUICK patience", cfg("LSTM_CFG_QUICK", "patience"))
c(R, "LSTM", "3", "results/lstm_training_curve.csv", "epochs run", n_rows("lstm_training_curve.csv"))
c(R, "LSTM", "1.397", "results/lstm_training_curve.csv", "epoch 1 val_mse", curve("lstm_training_curve.csv", 1))
c(R, "LSTM", "1.569", "results/lstm_training_curve.csv", "epoch 2 val_mse", curve("lstm_training_curve.csv", 2))
c(R, "LSTM", "1.673", "results/lstm_training_curve.csv", "epoch 3 val_mse", curve("lstm_training_curve.csv", 3))
c(R, "LSTM", "0.1", CFG, "LSTM_CFG_QUICK dropout", cfg("LSTM_CFG_QUICK", "dropout"))
c(R, "larger LSTM", "64", CFG, "LSTM_CFG hidden", cfg("LSTM_CFG", "hidden"))
c(R, "larger LSTM", "1.2557", "results/lstm_full_training_curve.csv", "min val_mse",
  float(csv("lstm_full_training_curve.csv").val_mse.min()))
c(R, "larger LSTM", "1.3970", "results/lstm_training_curve.csv", "min val_mse", float(csv("lstm_training_curve.csv").val_mse.min()))
c(R, "larger LSTM", "0.021138", HR, "lstm test rmse", hr("lstm", "test", "rmse"))
c(R, "larger LSTM", "0.020411", HR, "naive_zero test rmse", hr("naive_zero", "test", "rmse"))
c(R, "larger LSTM", "0.020142", "results/ckpt/lstm.json", "lstm test rmse", ckpt("lstm", "test", "rmse"))
c(R, "larger LSTM", "0.019589", "results/ckpt/naive_zero.json", "naive_zero test rmse", ckpt("naive_zero", "test", "rmse"))
c(R, "artefact table", "2", CFG, "LSTM_CFG layers", cfg("LSTM_CFG", "layers"))
c(R, "artefact table", "0.2", CFG, "LSTM_CFG dropout", cfg("LSTM_CFG", "dropout"))
c(R, "artefact table", "5", CFG, "LSTM_CFG patience", cfg("LSTM_CFG", "patience"))
c(R, "artefact table", "269,831", "results/ckpt/naive_zero.json", "test n_obs", ckpt("naive_zero", "test", "n_obs"))
c(R, "artefact table", "269,309", "results/ckpt/lstm.json", "test n_obs", ckpt("lstm", "test", "n_obs"))
c(R, "row counts", "465,811", DC, "rebuild_test_rows", cnt("rebuild_test_rows"))
c(R, "row counts", "465,289", DC, "lstm_test_rows - FISV rows dropped", cnt("lstm_test_rows") - (cnt("lstm_test_rows_FISV") - cnt("rebuild_test_rows_FISV")))
c(R, "row counts", "615", DC, "lstm_test_rows_FISV - rebuild_test_rows_FISV", cnt("lstm_test_rows_FISV") - cnt("rebuild_test_rows_FISV"))
c(R, "how to run", "2", CFG, "LSTM_CFG layers", cfg("LSTM_CFG", "layers"))
c(R, "layout", "50", "run_lstm_full.py", "max_epochs", code("run_lstm_full.py", r'\["max_epochs"\] = (\d+)'))
c(R, "what is not here", "66", DC, "prediction_files_MB", cnt("prediction_files_MB"))
c(R, "what is not here", "117", "results/ckpt/lstm_state.pt", "size in KB", (ROOT / "results/ckpt/lstm_state.pt").stat().st_size / 1000)
c(R, "Limitations", "17", DC, "etf_tickers", cnt("etf_tickers"))
c(R, "Limitations", "518", DC, "used_tickers", cnt("used_tickers"))
c(R, "Limitations", "14", DC, "zero_target_test_dates", cnt("zero_target_test_dates"))
c(R, "Limitations", "915", DC, "test_dates", cnt("test_dates"))
c(R, "Limitations", "10", "run_xs_regime.py", "COST_PER_SIDE bp", code("run_xs_regime.py", r"COST_PER_SIDE = ([\d.]+)") * 1e4)
c(R, "Correction note", "-0.24", XS, "lightgbm highvol ls_maxdd_net", xs("lightgbm", "highvol", "ls_maxdd_net"))
c(R, "Correction note", "-1.91", XS, "lightgbm normal ls_maxdd_net", xs("lightgbm", "normal", "ls_maxdd_net"))
c(R, "Correction note", "-2.02", XS, "lightgbm all ls_maxdd_net", xs("lightgbm", "all", "ls_maxdd_net"))
c(R, "Correction note", "-0.88", XS, "lightgbm highvol ls_sharpe_net", xs("lightgbm", "highvol", "ls_sharpe_net"))
c(R, "Correction note", "-4.89", XS, "lightgbm normal ls_sharpe_net", xs("lightgbm", "normal", "ls_sharpe_net"))
c(R, "Correction note", "-3.55", XS, "lightgbm all ls_sharpe_net", xs("lightgbm", "all", "ls_sharpe_net"))

# ---------------------------------------------------------------- paper
P = "paper"
c(P, "Abstract", "518", DC, "used_tickers", cnt("used_tickers"))
c(P, "Abstract", "0.044", XS, "lightgbm highvol ic_mean", xs("lightgbm", "highvol", "ic_mean"))
c(P, "Abstract", "3.2", XS, "lightgbm highvol ic_t", xs("lightgbm", "highvol", "ic_t"))
c(P, "Abstract", "0.005", XS, "lightgbm normal ic_mean", xs("lightgbm", "normal", "ic_mean"))
c(P, "Abstract", "0.9", XS, "lightgbm normal ic_t", xs("lightgbm", "normal", "ic_t"))
c(P, "Abstract", "2.09", XS, "lightgbm highvol sharpe gross", xs("lightgbm", "highvol", "ls_sharpe_gross"))
c(P, "Abstract", "0", WK, "ridge all ic_mean at 5 days, within print", round(wk("ridge", "all", "ic_mean"), 2))
c(P, "Intro", "12", HR, "distinct models excl. shuffled-target control",
  csv("horse_race_results.csv").query("model != 'SANITY_shuffled_ridge'").model.nunique())
c(P, "Data", "520", DC, "workbook_tickers", cnt("workbook_tickers"))
c(P, "Data", "518", DC, "used_tickers", cnt("used_tickers"))
c(P, "Data", "29", DC, "days_FDXF", cnt("days_FDXF"))
c(P, "Data", "16", DC, "days_HONA", cnt("days_HONA"))
c(P, "Data", "50", "src/build_dataset.py", "longest window", code("src/build_dataset.py", r"rolling\((\d+)\)\.mean\(\) - 1\n\s+vol"))
c(P, "Data", "501", DC, "stock_tickers", cnt("stock_tickers"))
c(P, "Data", "17", DC, "etf_tickers", cnt("etf_tickers"))
c(P, "Data", "12", "src/config.py", "CONTEXT_MAP entries", len(re.findall(r'^    "[^"]+": ', (ROOT / CFG).read_text().split("CONTEXT_MAP = {")[1].split("}")[0], re.M)))
c(P, "Features", "63", "src/build_dataset.py", "z window", code("src/build_dataset.py", r"def z\(s, w=(\d+)\)"))
c(P, "Splits", "466,426", HR, "naive_zero test n_obs", hr("naive_zero", "test", "n_obs"))
c(P, "Splits", "0.004", HR, "shuffled ic_mean", hr("SANITY_shuffled_ridge", "test", "ic_mean"))
c(P, "Splits", "0.67", HR, "shuffled ic_tstat", hr("SANITY_shuffled_ridge", "test", "ic_tstat"))
c(P, "Models", "30", CFG, "LSTM_CFG_QUICK seq_len", cfg("LSTM_CFG_QUICK", "seq_len"))
c(P, "Models", "32", CFG, "LSTM_CFG_QUICK hidden", cfg("LSTM_CFG_QUICK", "hidden"))
c(P, "Models", "0.1", CFG, "LSTM_CFG_QUICK dropout", cfg("LSTM_CFG_QUICK", "dropout"))
c(P, "Models", "64", CFG, "LSTM_CFG hidden", cfg("LSTM_CFG", "hidden"))
c(P, "Models", "0.2", CFG, "LSTM_CFG dropout", cfg("LSTM_CFG", "dropout"))
c(P, "Models", "2", CFG, "LSTM_CFG_QUICK patience", cfg("LSTM_CFG_QUICK", "patience"))
c(P, "Models", "5", CFG, "LSTM_CFG patience", cfg("LSTM_CFG", "patience"))
c(P, "Evaluation", "10", "run_xs_regime.py", "COST_PER_SIDE bp", code("run_xs_regime.py", r"COST_PER_SIDE = ([\d.]+)") * 1e4)
PT = [("SVR (linear)", "svr_linear", "0.490", "0.0141", "2.42"),
      ("Ridge / Linear", "ridge", "0.491", "0.0109", "1.86"),
      ("Ridge / Linear", "linear", "0.491", "0.0109", "1.86"),
      ("XGBoost", "xgboost", "0.497", "0.1413", "1.83"),
      ("LSTM", "lstm", "0.480", "0.0042", "0.76"),
      ("Shuffled-target", "SANITY_shuffled_ridge", "0.498", "0.0039", "0.67"),
      ("LightGBM", "lightgbm", "0.496", "-0.0295", "-0.69"),
      ("Decision tree", "decision_tree", "0.497", "-0.0838", "-1.60")]
for name, m, da, ic, t in PT:
    c(P, f"Table I {name}", da, HR, f"{m} test dir_acc", hr(m, "test", "dir_acc"))
    c(P, f"Table I {name}", ic, HR, f"{m} test ic_mean", hr(m, "test", "ic_mean"))
    c(P, f"Table I {name}", t, HR, f"{m} test ic_tstat", hr(m, "test", "ic_tstat"))
c(P, "Results 1", "0.011", HR, "ridge test ic_mean", hr("ridge", "test", "ic_mean"))
c(P, "Results 1", "0.014", HR, "svr_linear test ic_mean", hr("svr_linear", "test", "ic_mean"))
c(P, "Results 1", "1.9", HR, "ridge test ic_tstat", hr("ridge", "test", "ic_tstat"))
c(P, "Results 1", "2.4", HR, "svr_linear test ic_tstat", hr("svr_linear", "test", "ic_tstat"))
c(P, "Results 1", "0.004", HR, "lstm test ic_mean", hr("lstm", "test", "ic_mean"))
c(P, "Results 1", "0.76", HR, "lstm test ic_tstat", hr("lstm", "test", "ic_tstat"))
c(P, "Results 2", "183", XS, "highvol n_days", xs("ridge", "highvol", "n_days"))
c(P, "Results 2", "915", XS, "all n_days", xs("ridge", "all", "n_days"))
c(P, "Results 2", "20.0", XS, "highvol share of test days, %", 100 * xs("ridge", "highvol", "n_days") / xs("ridge", "all", "n_days"))
c(P, "Results 2", "63", "src/build_dataset.py", "z window", code("src/build_dataset.py", r"def z\(s, w=(\d+)\)"))
c(P, "Results 2", "1", "src/build_dataset.py", "z threshold", code("src/build_dataset.py", r'\) > (\d+)\)\.astype'))
c(P, "Results 2", "0.044", XS, "lightgbm highvol ic_mean", xs("lightgbm", "highvol", "ic_mean"))
c(P, "Results 2", "3.2", XS, "lightgbm highvol ic_t", xs("lightgbm", "highvol", "ic_t"))
c(P, "Results 2", "0.005", XS, "lightgbm normal ic_mean", xs("lightgbm", "normal", "ic_mean"))
c(P, "Results 2", "0.9", XS, "lightgbm normal ic_t", xs("lightgbm", "normal", "ic_t"))
for m, reg, ic, t, g, n in [("ridge", "highvol", "0.0320", "1.87", "0.96", "-0.69"),
                            ("ridge", "normal", "0.0111", "2.04", "0.28", "-2.93"),
                            ("ridge", "all", "0.0154", "2.76", "0.48", "-1.99"),
                            ("lightgbm", "highvol", "0.0436", "3.17", "2.09", "-0.88"),
                            ("lightgbm", "normal", "0.0048", "0.88", "-0.24", "-4.89"),
                            ("lightgbm", "all", "0.0127", "2.44", "0.44", "-3.55")]:
    c(P, f"Table II {m} {reg}", ic, XS, f"{m} {reg} ic_mean", xs(m, reg, "ic_mean"))
    c(P, f"Table II {m} {reg}", t, XS, f"{m} {reg} ic_t", xs(m, reg, "ic_t"))
    c(P, f"Table II {m} {reg}", g, XS, f"{m} {reg} ls_sharpe_gross", xs(m, reg, "ls_sharpe_gross"))
    c(P, f"Table II {m} {reg}", n, XS, f"{m} {reg} ls_sharpe_net", xs(m, reg, "ls_sharpe_net"))
c(P, "Results 1", "7", RG, "with_context lightgbm all n_days", rg("with_context", "lightgbm", "all", "n_days"))
c(P, "Results 1", "168", HR, "t^2/ic^2 of xgboost test", hr("xgboost", "test", "ic_tstat") ** 2 / hr("xgboost", "test", "ic_mean") ** 2)
c(P, "Results 2", "0.0515", RG, "with_context ridge highvol ic", rg("with_context", "ridge", "highvol", "ic_mean"))
c(P, "Results 2", "0.0450", RG, "no_context ridge highvol ic", rg("no_context", "ridge", "highvol", "ic_mean"))
c(P, "Results 3", "2.09", XS, "lightgbm highvol sharpe gross", xs("lightgbm", "highvol", "ls_sharpe_gross"))
c(P, "Results 3", "2.6", XS, "lightgbm all avg_turnover", xs("lightgbm", "all", "avg_turnover"))
c(P, "Results 3", "26", XS, "lightgbm all avg_turnover x 10 bp", xs("lightgbm", "all", "avg_turnover") * 10)
c(P, "Features", "21", "src/build_dataset.py", "ret and vol window", code("src/build_dataset.py", r"logc\.diff\((\d+)\)\n\s+g\[\"f_vol"))
c(P, "Splits", "2021", CFG, "TRAIN_END year", code(CFG, r'TRAIN_END = "(\d{4})'))
c(P, "Splits", "2022", CFG, "VAL_END year", code(CFG, r'VAL_END = "(\d{4})'))
c(P, "Splits", "2023", CFG, "VAL_END year + 1", code(CFG, r'VAL_END = "(\d{4})') + 1)
c(P, "Results 3", "5", "src/build_dataset.py", "fwd_ret_5d horizon (days)", code("src/build_dataset.py", r"logc\.shift\(-(\d+)\) - logc"))
c(P, "Results 3", "-0.007", WK, "ridge highvol ic_mean", wk("ridge", "highvol", "ic_mean"))
for m, reg, ic, t, n in [("ridge", "highvol", "-0.0071", "-0.19", "-0.27"),
                         ("ridge", "normal", "0.0027", "0.20", "-0.74"),
                         ("ridge", "all", "0.0007", "0.05", "-0.64")]:
    c(P, f"Table III {m} {reg}", ic, WK, f"{m} {reg} ic_mean", wk(m, reg, "ic_mean"))
    c(P, f"Table III {m} {reg}", t, WK, f"{m} {reg} ic_t", wk(m, reg, "ic_t"))
    c(P, f"Table III {m} {reg}", n, WK, f"{m} {reg} sharpe_net", wk(m, reg, "sharpe_net"))
c(P, "Table III lightgbm all", "0", WK, "lightgbm all n_weeks", wk("lightgbm", "all", "n_weeks"))


def decimals(s):
    s = s.replace(",", "")
    return len(s.split(".")[1]) if "." in s else 0


def ok(printed, actual):
    return abs(actual - float(printed.replace(",", ""))) <= 0.5 * 10 ** -decimals(printed) + 1e-9


def table():
    return "\n".join(f"{'OK  ' if ok(p, a) else 'FAIL'} {d:6} {loc:34} {p:>9} {src:42} {fld:44} {a:.6g}"
                     for d, loc, p, src, fld, a in C)


@pytest.mark.parametrize("doc,loc,printed,src,field,actual", C, ids=[f"{r[0]}:{r[1]}:{r[2]}:{r[4]}" for r in C])
def test_claim(doc, loc, printed, src, field, actual):
    assert ok(printed, actual), f"{doc} {loc}: prints {printed}, {src} {field} = {actual}"


def test_table(capsys):
    with capsys.disabled():
        print("\n" + table())


# ------------------------------------------------------------ coverage
# (doc, number as printed without sign or thousands separators) -> one-word reason
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
ALLOW = {
    ("README", "21"): "span", ("README", "500"): "name", ("README", "March 2005"): "date",
    ("README", "July 2026"): "date", ("README", "16 November 2026"): "date", ("README", "2026-08-23"): "date",
    ("paper", "March 2005"): "date", ("paper", "July 2026"): "date", ("paper", "2026"): "year",
    ("paper", "500"): "name",
}


def numbers(doc):
    t = DOCS[doc].read_text()
    if doc == "README":
        t = re.sub(r"```.*?```", " ", t, flags=re.S)
        t = re.sub(r"`[^`]*`", " ", t)
        t = re.sub(r"\]\([^)]*\)", "]", t)
        t = re.sub(r"https?://\S+", " ", t)
    else:
        t = t.split(r"\begin{document}")[1].split(r"\begin{thebibliography}")[0]
        t = re.sub(r"(?<!\\)%.*", "", t)
        t = re.sub(r"\\(?:cite|label|ref|includegraphics|multicolumn|begin|end)\*?(?:\[[^\]]*\])?(?:\{[^}]*\})+", " ", t)
        t = t.replace("{,}", ",").replace("--", " ")
        t = re.sub(r"\\times\s*10\^\{[^}]*\}", " ", t)
    dates = re.findall(r"\d{4}-\d{2}-\d{2}|\d{1,2} (?:%s) \d{4}|(?:%s) \d{4}" % (MONTHS, MONTHS), t)
    t = re.sub(r"\d{4}-\d{2}-\d{2}|\d{1,2} (?:%s) \d{4}|(?:%s) \d{4}" % (MONTHS, MONTHS), " ", t)
    return dates + [m.replace(",", "") for m in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", t)]


def covered(doc):
    return {re.sub(r"^-", "", p).replace(",", "") for d, _, p, *_ in C if d == doc}


@pytest.mark.parametrize("doc", list(DOCS))
def test_every_number_is_covered(doc):
    toks = numbers(doc)
    cov = covered(doc)
    allow = {t for (d, t) in ALLOW if d == doc}
    missing = sorted({t for t in toks if t not in cov and t not in allow})
    assert not missing, f"{doc}: numbers in neither a claim row nor ALLOW: {missing}"
    stale = sorted(t for t in allow if t not in toks)
    assert not stale, f"{doc}: ALLOW entries not in the document: {stale}"


def test_coverage_counts(capsys):
    with capsys.disabled():
        for doc in DOCS:
            toks = numbers(doc)
            cov = covered(doc)
            n_c = sum(t in cov for t in toks)
            n_a = sum(t not in cov and (doc, t) in ALLOW for t in toks)
            print(f"{doc}: {len(toks)} number occurrences, {n_c} covered by claims, {n_a} allowlisted")
