"""Fama-French risk adjustment for the cross-sectional long-short strategy.

Downloads Ken French's daily factors (public; identical to WRDS `ff_all`) and
regresses the strategy's daily long-short return on the market, size, value and
momentum factors. The intercept (alpha) tests whether the spread is genuine
excess return or just compensation for known factor exposure. Reported per
regime (all / high-vol / normal).

Runs standalone here; when WRDS is available, swap `load_ff_factors()` for a pull
of `ff_all` (same columns) with zero downstream changes.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

FF3_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_Factors_daily_CSV.zip")
MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Momentum_Factor_daily_CSV.zip")


def _read_french_zip(url, value_name):
    raw = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=90).read()))
    txt = raw.read(raw.namelist()[0]).decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(txt) if l.strip().startswith(","))
    hdr = [h.strip() for h in txt[start].split(",")]
    hdr[0] = "date"
    rows = []
    for l in txt[start + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) < 2 or not parts[0].isdigit() or len(parts[0]) != 8:
            continue
        rows.append(parts[:len(hdr)])
    df = pd.DataFrame(rows, columns=hdr)
    df = df.loc[:, ~df.columns.duplicated()]        # drop any stray dup cols
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df


def load_ff_factors():
    ff3 = _read_french_zip(FF3_URL, "ff3")
    mom = _read_french_zip(MOM_URL, "mom")
    mom = mom.rename(columns={c: "Mom" for c in mom.columns if c != "date"})
    ff = ff3.merge(mom, on="date", how="inner")
    ff.columns = [c.replace("-", "_").replace(" ", "") for c in ff.columns]
    return ff  # date, Mkt_RF, SMB, HML, RF, Mom


def risk_adjust(ls_daily, ff, label=""):
    """ls_daily: DataFrame [date, ret] of daily long-short returns (dollar-neutral,
    so RF is not subtracted). Returns alpha (annualized), t-stats, betas, R^2."""
    d = ls_daily.merge(ff, on="date", how="inner").dropna()
    if len(d) < 20:
        return dict(label=label, n=len(d), note="too few obs")
    X = sm.add_constant(d[["Mkt_RF", "SMB", "HML", "Mom"]])
    m = sm.OLS(d["ret"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    a_daily = m.params["const"]
    return dict(
        label=label, n=len(d),
        alpha_ann=round(a_daily * 252, 4),
        alpha_t=round(m.tvalues["const"], 2),
        beta_mkt=round(m.params["Mkt_RF"], 3),
        beta_smb=round(m.params["SMB"], 3),
        beta_hml=round(m.params["HML"], 3),
        beta_mom=round(m.params["Mom"], 3),
        r2=round(m.rsquared, 3),
        gross_ann=round(d["ret"].mean() * 252, 4),
    )


if __name__ == "__main__":
    # demo/self-test: risk-adjust the ridge long-short returns saved by the
    # cross-sectional runner (regenerated here if absent)
    ff = load_ff_factors()
    print(f"FF factors: {ff['date'].min().date()} -> {ff['date'].max().date()}, "
          f"cols {list(ff.columns)}")
    lsf = C.RESULTS / "ridge_ls_daily.csv"
    if lsf.exists():
        ls = pd.read_csv(lsf, parse_dates=["date"])
        for reg, name in [("all", "all"), (1, "highvol"), (0, "normal")]:
            sub = ls if reg == "all" else ls[ls.regime == reg]
            print(risk_adjust(sub[["date", "ret"]], ff, name))
    else:
        print(f"No {lsf}; run compute_ls_series first.")
