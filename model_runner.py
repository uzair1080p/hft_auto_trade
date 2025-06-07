# model_runner.py

"""
Runs model inference loop using LSTM + LightGBM ensemble.
Consumes features from ClickHouse and logs predictions to table.
"""

import time
import json
import torch
import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
from clickhouse_connect import get_client
from sklearn.preprocessing import RobustScaler

# -------------------- Config --------------------

CLICKHOUSE_HOST = 'clickhouse'
CLICKHOUSE_USER = 'default'
CLICKHOUSE_PASS = ''
SYMBOL = 'ETHUSDT'
LOOKBACK = 20  # LSTM sequence length
INTERVAL_SECONDS = 60

# -------------------- ClickHouse Client --------------------

client = get_client(
    host=CLICKHOUSE_HOST,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASS,
    compress=True
)

# -------------------- LSTM Model --------------------

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, 32, batch_first=True)
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# -------------------- Model Inference --------------------

def load_latest_features():
    q = f"""
    SELECT ts, features
    FROM futures_features
    WHERE symbol = '{SYMBOL}'
    ORDER BY ts DESC
    LIMIT {LOOKBACK}
    """
    rows = client.query(q).result_rows[::-1]
    if len(rows) < LOOKBACK:
        return None
    return pd.DataFrame([json.loads(r[1]) for r in rows])

def predict_trade_signal(lstm_model, lgb_model, scaler, features):
    normed = scaler.transform(features)
    x_seq = torch.tensor(normed.reshape(1, LOOKBACK, -1), dtype=torch.float32)
    with torch.no_grad():
        lstm_out = torch.sigmoid(lstm_model(x_seq)).item()
    lgb_out = lgb_model.predict(normed[-1:].reshape(1, -1))[0]
    final_score = 0.6 * lstm_out + 0.4 * lgb_out
    return 1 if final_score > 0.6 else -1 if final_score < 0.4 else 0, final_score

def log_trade_decision(ts, signal, score):
    row = {
        'ts': ts,
        'symbol': SYMBOL,
        'signal': signal,
        'score': score
    }
    client.insert("executed_trades", [row])
    logging.info(f"[{ts}] Signal: {signal}, Score: {score:.4f}")

# -------------------- Main --------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scaler = RobustScaler()
    lstm_model = LSTMModel(input_size=10)
    lstm_model.load_state_dict(torch.load("lstm_model.pth"))
    lstm_model.eval()
    lgb_model = lgb.Booster(model_file="lightgbm_model.txt")

    # Warm-up scaler on historical data
    hist_q = f"""
    SELECT features FROM futures_features
    WHERE symbol = '{SYMBOL}'
    ORDER BY ts DESC LIMIT 500
    """
    hist_rows = client.query(hist_q).result_rows
    df_hist = pd.DataFrame([json.loads(r[0]) for r in hist_rows])
    scaler.fit(df_hist)

    while True:
        try:
            df = load_latest_features()
            if df is not None:
                signal, score = predict_trade_signal(lstm_model, lgb_model, scaler, df)
                log_trade_decision(pd.Timestamp.utcnow(), signal, score)
        except Exception as e:
            logging.error(f"Error in model loop: {e}")
        time.sleep(INTERVAL_SECONDS)
