"""
Model-inference loop for the HFT bot
––––––––––––––––––––––––––––––––––––
• Pulls last 20 feature rows (futures_features)
• Robust-scales → LSTM (sequence) + LightGBM (point)
• Blended score → executed_trades
• Auto-creates tables on first run and waits until features exist
"""

import json, os, time, logging, pathlib
from datetime import datetime as dt

import numpy as np
import pandas as pd
import torch, lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from clickhouse_connect import get_client


# ───────────────── CONFIG ─────────────────
CLICKHOUSE = dict(
    host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
    port=int(os.getenv("CLICKHOUSE_PORT", 8123)),
    username=os.getenv("CLICKHOUSE_USER", "default"),
    password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    compress=True,
)

SYMBOL   = "ETHUSDT"
LOOKBACK = 20
SLEEP    = 60

LSTM_WEIGHTS = pathlib.Path("models/lstm_model.pth")
LGBM_FILE    = pathlib.Path("models/lightgbm_model.txt")

# ───────── ClickHouse client & DDL ────────
ck = get_client(**CLICKHOUSE)

ck.command("""
CREATE TABLE IF NOT EXISTS futures_features (
    ts        DateTime,
    symbol    String,
    features  String,
    raw_data  String
) ENGINE = MergeTree ORDER BY (symbol, ts)
""")

ck.command("""
CREATE TABLE IF NOT EXISTS executed_trades (
    ts     DateTime,
    symbol String,
    signal Int8,
    score  Float32
) ENGINE = MergeTree ORDER BY (symbol, ts)
""")

# ───────── LSTM definition ───────────────
class LSTM(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(d, 32, batch_first=True)
        self.fc   = torch.nn.Linear(32, 1)

    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1])

# ───────── helper queries ────────────────
def last_n_features(n=LOOKBACK):
    rows = ck.query(
        f"SELECT features FROM futures_features "
        f"WHERE symbol='{SYMBOL}' ORDER BY ts DESC LIMIT {n}"
    ).result_rows[::-1]
    return None if len(rows) < n else pd.DataFrame(
        [json.loads(r[0]) for r in rows]
    )

def log_trade(ts, signal, score):
    ck.insert("executed_trades",
              [{"ts": ts, "symbol": SYMBOL,
                "signal": int(signal), "score": float(score)}])

# ───────── bootstrap / warm-up ───────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# wait until at least one feature row exists
probe = None
for _ in range(30):                           # ~3 min max
    probe = last_n_features(1)
    if probe is not None:
        break
    logging.info("Waiting for first feature row …")
    time.sleep(6)

if probe is None:
    logging.error("No features arrived; exiting.")
    raise SystemExit(1)

INPUT_DIM = probe.shape[1]

# scaler fit on up to 1000 rows
hist_rows = ck.query(
    f"SELECT features FROM futures_features "
    f"WHERE symbol='{SYMBOL}' ORDER BY ts DESC LIMIT 1000"
).result_rows
scaler = RobustScaler().fit(
    pd.DataFrame([json.loads(r[0]) for r in hist_rows]) if hist_rows else probe
)

# load / init models
lstm = LSTM(INPUT_DIM)
if LSTM_WEIGHTS.exists():
    lstm.load_state_dict(torch.load(LSTM_WEIGHTS, map_location="cpu"))
    logging.info("✓ LSTM weights loaded")
else:
    logging.warning("⚠️  No LSTM weights, using random init")

lstm.eval()

lgbm = None
if LGBM_FILE.exists():
    lgbm = lgb.Booster(model_file=str(LGBM_FILE))
    logging.info("✓ LightGBM model loaded")
else:
    logging.warning("⚠️  No LightGBM model file")

# ───────── main loop ─────────────────────
while True:
    try:
        seq_df = last_n_features()
        if seq_df is None:
            time.sleep(SLEEP)
            continue

        norm = scaler.transform(seq_df).reshape(1, LOOKBACK, -1)
        x_t  = torch.tensor(norm, dtype=torch.float32)

        with torch.no_grad():
            p_lstm = torch.sigmoid(lstm(x_t)).item()
        p_lgb  = float(lgbm.predict(norm[:, -1, :])[0]) if lgbm else 0.5

        score  = 0.6 * p_lstm + 0.4 * p_lgb
        signal = 1 if score > 0.6 else -1 if score < 0.4 else 0

        log_trade(dt.utcnow(), signal, score)
        logging.info(f"signal {signal:+d} | score {score:0.4f}")

    except Exception as e:
        logging.exception(f"Model loop error: {e}")

    time.sleep(SLEEP)