"""
LSTM + LightGBM ensemble → ClickHouse table executed_trades
"""

import json, logging, os, pathlib, time

import numpy as np
import pandas as pd
import torch, lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from clickhouse_connect import get_client


# ───────── CONFIG ─────────
CLICKHOUSE = dict(
    host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
    username=os.getenv("CLICKHOUSE_USER", "default"),
    password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    compress=True,
)

SYMBOL           = "ETHUSDT"
LOOKBACK         = 20
SLEEP_SEC        = 60
LSTM_WEIGHTS     = pathlib.Path("models/lstm_model.pth")
LGBM_FILE        = pathlib.Path("models/lightgbm_model.txt")

ck = get_client(**CLICKHOUSE)

# auto-DDL
ck.command("""
CREATE TABLE IF NOT EXISTS executed_trades (
    ts      DateTime,
    symbol  String,
    signal  Int8,
    score   Float32
) ENGINE = MergeTree ORDER BY (symbol, ts)
""")

ck.command("""
CREATE TABLE IF NOT EXISTS futures_features (
    ts        DateTime,
    symbol    String,
    features  JSON,
    raw_data  JSON
) ENGINE = MergeTree ORDER BY (symbol, ts)
""")

# ───────── models & scaler ─────────
def last_n_features(n=LOOKBACK):
    rows = ck.query(
        f"SELECT features FROM futures_features "
        f"WHERE symbol='{SYMBOL}' ORDER BY ts DESC LIMIT {n}"
    ).result_rows[::-1]
    return None if len(rows)<n else pd.DataFrame([json.loads(r[0]) for r in rows])

probe = last_n_features(1)
if probe is None:
    logging.error("No feature rows yet; model loop exits.")
    exit(1)

INPUT_DIM = probe.shape[1]
scaler = RobustScaler().fit(probe)

class LSTM(torch.nn.Module):
    def __init__(self, d): super().__init__()
    def __init__(self, d):
        super().__init__()
        self.lstm=torch.nn.LSTM(d,32,batch_first=True)
        self.fc=torch.nn.Linear(32,1)
    def forward(self,x):
        return self.fc(self.lstm(x)[0][:,-1])

lstm=LSTM(INPUT_DIM)
if LSTM_WEIGHTS.exists():
    lstm.load_state_dict(torch.load(LSTM_WEIGHTS,map_location="cpu"))
    logging.info("LSTM weights loaded")
else:
    logging.warning("No LSTM weights; random init")

lstm.eval()
lgbm = lgb.Booster(model_file=str(LGBM_FILE)) if LGBM_FILE.exists() else None

# ───────── main loop ─────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
while True:
    try:
        df = last_n_features()
        if df is not None:
            x_seq = scaler.transform(df).reshape(1,LOOKBACK,-1)
            lstm_prob = torch.sigmoid(lstm(torch.tensor(x_seq, dtype=torch.float32))).item()
            lgb_prob  = lgbm.predict(x_seq[:,-1]) [0] if lgbm else 0.5
            score = 0.6*lstm_prob + 0.4*lgb_prob
            signal = 1 if score>0.6 else -1 if score<0.4 else 0
            ck.insert("executed_trades",[dict(ts=pd.Timestamp.utcnow(),
                                              symbol=SYMBOL,signal=signal,score=score)])
            logging.info(f"signal {signal:+d}  score {score:0.4f}")
    except Exception as e:
        logging.exception(e)
    time.sleep(SLEEP_SEC)