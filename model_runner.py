"""
Model-inference loop for the HFT bot.
Combines an LSTM (sequence) and LightGBM (point) model.
Writes signals to ClickHouse → streamlit_ui displays them.
"""

import json, time, logging, os, pathlib

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from clickhouse_connect import get_client


# ──────────────────────── CONFIG ────────────────────────
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", 8123))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASS = os.getenv("CLICKHOUSE_PASSWORD", "")

SYMBOL            = "ETHUSDT"
LOOKBACK          = 20            # timesteps fed to LSTM
INTERVAL_SECONDS  = 60            # model loop sleep
LSTM_WEIGHTS      = pathlib.Path("models/lstm_model.pth")
LGBM_MODEL_FILE   = pathlib.Path("models/lightgbm_model.txt")

# ───────────────────── ClickHouse client ────────────────
ck = get_client(
    host=CLICKHOUSE_HOST,
    port=CLICKHOUSE_PORT,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASS,
    compress=True,
)

# ─────────────────────── LSTM module ────────────────────
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size=32, batch_first=True)
        self.fc   = torch.nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ──────────────────── helpers ───────────────────────────
def recent_feature_df(n=LOOKBACK):
    q = (
        f"SELECT ts, features FROM futures_features "
        f"WHERE symbol = '{SYMBOL}' ORDER BY ts DESC LIMIT {n}"
    )
    rows = ck.query(q).result_rows[::-1]
    if len(rows) < n:
        return None
    return pd.DataFrame([json.loads(r[1]) for r in rows])

def log_trade(ts, signal: int, score: float):
    ck.insert(
        "executed_trades",
        [{"ts": ts, "symbol": SYMBOL, "signal": signal, "score": score}],
    )
    logging.info(f"[{ts}] signal={signal:+d}  score={score:0.4f}")

# ──────────────────── bootstrap ─────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# get feature dimension from a single row
probe = recent_feature_df(1)
if probe is None:
    raise RuntimeError("No features in futures_features yet.")
FEATURE_COLUMNS = list(probe.columns)
INPUT_SIZE      = len(FEATURE_COLUMNS)

# scaler
scaler = RobustScaler()
hist = ck.query(
    f"SELECT features FROM futures_features WHERE symbol = '{SYMBOL}' "
    f"ORDER BY ts DESC LIMIT 1000"
).result_rows
if hist:
    scaler.fit(pd.DataFrame([json.loads(r[0]) for r in hist]))
else:
    scaler.fit(probe)                   # minimal fit

# LSTM
lstm = LSTMModel(input_size=INPUT_SIZE)
if LSTM_WEIGHTS.exists():
    lstm.load_state_dict(torch.load(LSTM_WEIGHTS, map_location="cpu"))
    logging.info("Loaded LSTM weights")
else:
    logging.warning("⚠️  LSTM weights not found – random init")

lstm.eval()

# LightGBM
if LGBM_MODEL_FILE.exists():
    lgbm = lgb.Booster(model_file=str(LGBM_MODEL_FILE))
    logging.info("Loaded LightGBM model")
else:
    logging.warning("⚠️  LightGBM model file missing – using zero output")
    lgbm = None

# ──────────────────── main loop ─────────────────────────
while True:
    try:
        df_seq = recent_feature_df()
        if df_seq is None:
            time.sleep(INTERVAL_SECONDS)
            continue

        norm_seq = scaler.transform(df_seq)
        x = torch.tensor(norm_seq.reshape(1, LOOKBACK, -1), dtype=torch.float32)
        with torch.no_grad():
            lstm_prob = torch.sigmoid(lstm(x)).item()

        lgb_prob = (
            float(lgbm.predict(norm_seq[-1:].reshape(1, -1))[0]) if lgbm else 0.5
        )

        score  = 0.6 * lstm_prob + 0.4 * lgb_prob
        signal = 1 if score > 0.6 else -1 if score < 0.4 else 0
        log_trade(pd.Timestamp.utcnow(), signal, score)

    except Exception as e:
        logging.exception(f"Model loop error: {e!s}")

    time.sleep(INTERVAL_SECONDS)