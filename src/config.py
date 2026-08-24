"""Central configuration. Split dates and universes live here and ONLY here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
for p in (RAW, PROC, RESULTS, FIGS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
START_DATE = "2005-01-01"

# Chronological splits (label date determines membership)
TRAIN_END = "2021-12-31"
VAL_END = "2022-12-31"   # test = everything after

# ----------------------------- Universes (Bloomberg convention) -------------
V1A_EQUITIES = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "BRK/B",
    "JPM", "V", "MA", "UNH", "XOM", "JNJ", "PG", "HD", "COST", "ORCL", "KO",
    "PEP", "BAC", "WMT", "CRM", "NFLX", "AMD", "INTC", "CSCO", "ADBE", "QCOM",
    "TXN", "IBM", "GE", "CAT", "GS", "MS", "LLY", "ABBV", "MRK", "PFE",
    "CVX", "WFC", "DIS", "MCD", "LIN", "HON",
]
V1A_ETFS = ["SPY", "QQQ", "IWM", "DIA",
            "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB"]

V4_ETFS = ["ACWI", "EFA", "EEM", "EWJ", "EWG", "EWU", "EWS", "EWT", "EWY",
           "INDA", "FXI", "MCHI", "KWEB", "ASHR", "VGK", "FEZ"]

# Context series: Bloomberg name -> Yahoo symbol (None = unavailable, proxy noted)
CONTEXT_MAP = {
    "SPX Index": "^GSPC",
    "NDX Index": "^NDX",
    "RTY Index": "^RUT",
    "VIX Index": "^VIX",
    "MOVE Index": "^MOVE",
    "DXY Curncy": "DX-Y.NYB",
    "USGG2YR Index": "^IRX",    # PROXY: 13w bill; Bloomberg authoritative
    "USGG10YR Index": "^TNX",
    "USGG30YR Index": "^TYX",
    "CL1 Comdty": "CL=F",
    "GC1 Comdty": "GC=F",
    "HG1 Comdty": "HG=F",
}


def to_yahoo(bbg_short: str) -> str:
    """Map an equity/ETF short ticker (no ' US Equity') to Yahoo symbol."""
    return bbg_short.replace("/", "-")


# ----------------------------- LSTM config ----------------------------------
LSTM_CFG = dict(seq_len=30, hidden=64, layers=2, dropout=0.2,
                lr=1e-3, batch_size=1024, max_epochs=30, patience=5)
LSTM_CFG_QUICK = dict(seq_len=30, hidden=32, layers=1, dropout=0.1,
                      lr=1e-3, batch_size=1024, max_epochs=6, patience=2)
