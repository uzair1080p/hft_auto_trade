# hft_trading_system.py

"""
Production-grade High-Frequency Trading Bot for ETH/BTC Futures on Binance
Stack: ClickHouse, Kafka, LSTM + LightGBM Ensemble, Adaptive Normalization, Backtrader for backtesting
"""

import time
import json
import logging
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from datetime import datetime, timedelta
from clickhouse_connect import get_client
from kafka import KafkaConsumer, KafkaProducer
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.model_selection import TimeSeriesSplit
import optuna
import shap
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import backtrader as bt
import streamlit as st
import subprocess

# -------------------- Configurations --------------------

KAFKA_BOOTSTRAP_SERVERS = ['kafka:9092']
CLICKHOUSE_HOST = 'clickhouse'
CLICKHOUSE_USER = 'default'
CLICKHOUSE_PASS = ''

# -------------------- ClickHouse Client --------------------

clickhouse_client = get_client(
    host=CLICKHOUSE_HOST, 
    username=CLICKHOUSE_USER, 
    password=CLICKHOUSE_PASS, 
    compress=True
)

# -------------------- Persistent Storage --------------------

def insert_batch_clickhouse(data_rows):
    try:
        clickhouse_client.insert(
            "INSERT INTO futures_features (ts, symbol, features, raw_data) VALUES",
            data_rows
        )
    except Exception as e:
        logging.error(f"ClickHouse insert failed: {e}")
        time.sleep(2)
        clickhouse_client.insert(
            "INSERT INTO futures_features (ts, symbol, features, raw_data) VALUES",
            data_rows
        )

# -------------------- Streamlit UI --------------------

def launch_streamlit_ui():
    st.set_page_config(layout="wide")
    st.title("Real-Time Trade Dashboard")
    df = pd.read_sql("SELECT * FROM executed_trades ORDER BY ts DESC LIMIT 100", clickhouse_client.connection)
    st.dataframe(df)

# -------------------- SHAP Drift Detection --------------------

def detect_drift(model, X_train, X_live):
    explainer = shap.Explainer(model.predict, X_train)
    shap_train = explainer(X_train[:100])
    shap_live = explainer(X_live[:100])
    mean_diff = np.abs(np.mean(shap_train.values, axis=0) - np.mean(shap_live.values, axis=0))
    if np.any(mean_diff > 0.1):
        logging.warning("Drift detected in SHAP values")
    return mean_diff

# -------------------- Inference --------------------

def predict_signal(model, scaler, live_features, threshold=0.5):
    normed = scaler.transform(live_features)
    x = torch.tensor(normed.values.reshape(1, 1, -1), dtype=torch.float32)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    return 1 if prob > threshold else -1 if prob < 1 - threshold else 0

# -------------------- Backtrader Strategy --------------------

class LSTMBacktestStrategy(bt.Strategy):
    def __init__(self, model, scaler, feature_cols):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols

    def next(self):
        if len(self.datas[0]) < 10:
            return
        features = {col: float(getattr(self.datas[0], col)[0]) for col in self.feature_cols}
        df = pd.DataFrame([features])
        normed = self.scaler.transform(df)
        x = torch.tensor(normed.reshape(1, 1, -1), dtype=torch.float32)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()
        if not self.position:
            if prob > 0.6:
                self.buy()
            elif prob < 0.4:
                self.sell()
        elif self.position.size > 0 and prob < 0.5:
            self.close()
        elif self.position.size < 0 and prob > 0.5:
            self.close()

# -------------------- Main --------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        subprocess.Popen(["streamlit", "run", __file__])
    except KeyboardInterrupt:
        print("Shutting down...")
