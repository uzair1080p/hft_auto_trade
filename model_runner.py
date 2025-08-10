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
import os
from clickhouse_connect import get_client
from sklearn.preprocessing import RobustScaler

# -------------------- Config --------------------

CLICKHOUSE_HOST = 'clickhouse'
CLICKHOUSE_USER = 'default'
CLICKHOUSE_PASS = ''
SYMBOL = os.getenv('SYMBOL', 'DOGEUSDT')
LOOKBACK = 20  # LSTM sequence length
INTERVAL_SECONDS = 60

# -------------------- ClickHouse Client --------------------

_client = None

def get_clickhouse_client():
    """Lazily create and return a ClickHouse client.
    Avoids establishing a connection at import time so unit tests
    that import this module do not require a running ClickHouse service.
    """
    global _client
    if _client is None:
        host = os.getenv('CLICKHOUSE_HOST', CLICKHOUSE_HOST)
        user = os.getenv('CLICKHOUSE_USER', CLICKHOUSE_USER)
        pwd = os.getenv('CLICKHOUSE_PASS', CLICKHOUSE_PASS)
        _client = get_client(
            host=host,
            username=user,
            password=pwd,
            compress=True
        )
    return _client

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
    rows = get_clickhouse_client().query(q).result_rows[::-1]
    if len(rows) < LOOKBACK:
        return None
    return pd.DataFrame([json.loads(r[1]) for r in rows])

def _heuristic_score(df: pd.DataFrame) -> float:
    """Compute a heuristic probability based on recent feature values."""
    last = df.iloc[-1]
    ob = float(last.get("ob_imbalance", 0.0))
    rsi = float(last.get("rsi", 50.0))
    spread = float(last.get("spread", 0.0))
    atr = float(last.get("atr", 0.0))

    ob_component = np.tanh(ob * 5)
    rsi_component = np.tanh((50 - rsi) / 20)
    spread_component = -np.tanh(spread * 10)
    atr_component = -np.tanh(atr)

    score = (
        0.4 * ob_component
        + 0.3 * rsi_component
        + 0.2 * spread_component
        + 0.1 * atr_component
    )
    return float((score + 1) / 2)


def predict_trade_signal(lstm_model, lgb_model, scaler, features):
    normed = scaler.transform(features)
    x_seq = torch.tensor(normed.reshape(1, LOOKBACK, -1), dtype=torch.float32)
    with torch.no_grad():
        lstm_out = torch.sigmoid(lstm_model(x_seq)).item()
    # LightGBM may be optional
    try:
        lgb_out = lgb_model.predict(normed[-1:].reshape(1, -1))[0] if lgb_model is not None else 0.5
    except Exception:
        lgb_out = 0.5
    ml_score = 0.6 * lstm_out + 0.4 * lgb_out
    heur_score = _heuristic_score(features)
    final_score = 0.7 * ml_score + 0.3 * heur_score
    return 1 if final_score > 0.6 else -1 if final_score < 0.4 else 0, float(final_score)

def log_trade_decision(ts, signal, score):
    row = {
        'ts': ts,
        'symbol': SYMBOL,
        'signal': signal,
        'score': score
    }
    get_clickhouse_client().insert("executed_trades", [row])
    logging.info(f"[{ts}] Signal: {signal}, Score: {score:.4f}")

# -------------------- Main --------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scaler = RobustScaler()
    lstm_model = LSTMModel(input_size=10)

    # Try to load persisted models with graceful fallback
    try:
        lstm_model.load_state_dict(torch.load("lstm_model.pth"))
        logging.info("Loaded LSTM model weights")
    except Exception as e:
        logging.warning(f"Falling back to fresh LSTM weights: {e}")
    lstm_model.eval()

    try:
        lgb_model = lgb.Booster(model_file="lightgbm_model.txt")
        logging.info("Loaded LightGBM model")
    except Exception as e:
        logging.warning(f"Falling back to heuristic-only LightGBM (None): {e}")
        lgb_model = None

    # Warm-up scaler on historical data; fallback to synthetic if needed
    try:
        hist_q = f"""
        SELECT features FROM futures_features
        WHERE symbol = '{SYMBOL}'
        ORDER BY ts DESC LIMIT 500
        """
        hist_rows = get_clickhouse_client().query(hist_q).result_rows
        if hist_rows:
            df_hist = pd.DataFrame([json.loads(r[0]) for r in hist_rows])
            # Pick first 10 numeric features consistently
            numeric_cols = [c for c in df_hist.columns if np.issubdtype(pd.Series(df_hist[c]).dtype, np.number)]
            if len(numeric_cols) < 10:
                # Synthesize extra columns to reach 10
                for i in range(10 - len(numeric_cols)):
                    df_hist[f"synth_{i}"] = 0.0
                numeric_cols = numeric_cols + [f"synth_{i}" for i in range(10 - len(numeric_cols))]
            feature_cols = numeric_cols[:10]
            df_hist10 = df_hist[feature_cols].fillna(0.0)
            scaler.fit(df_hist10)
        else:
            raise RuntimeError("No historical features")
    except Exception as e:
        logging.warning(f"Scaler warm-up fallback due to: {e}")
        scaler.fit(np.zeros((LOOKBACK, 10)))
        feature_cols = [f"f{i}" for i in range(10)]

    while True:
        try:
            df = load_latest_features()
            if df is not None:
                # Align to 10 features using the same selected columns or fallback
                try:
                    numeric_cols_now = [c for c in df.columns if np.issubdtype(pd.Series(df[c]).dtype, np.number)]
                    for c in feature_cols:
                        if c not in df.columns:
                            df[c] = 0.0
                    df10 = df[feature_cols].fillna(0.0)
                except Exception:
                    df10 = df.select_dtypes(include=[np.number])
                    if df10.shape[1] < 10:
                        for i in range(10 - df10.shape[1]):
                            df10[f"synth_{i}"] = 0.0
                    df10 = df10.iloc[:, :10]
                signal, score = predict_trade_signal(lstm_model, lgb_model, scaler, df10)
                log_trade_decision(pd.Timestamp.utcnow(), signal, score)
            else:
                logging.warning("No features available for prediction")
        except Exception as e:
            logging.error(f"Error in model loop: {e}")
            # Don't exit on error, just continue
        time.sleep(INTERVAL_SECONDS)
