# collector.py

"""
Collects real-time Binance futures data and pushes features into ClickHouse.
"""

import json
import time
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from binance import ThreadedWebsocketManager
from clickhouse_connect import get_client

# -------------------- Config --------------------

API_KEY = 'YOUR_BINANCE_API_KEY'
API_SECRET = 'YOUR_BINANCE_SECRET_KEY'
SYMBOL = 'ETHUSDT'
INTERVAL = '1m'

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

# -------------------- Collector Class --------------------

class CryptoDataCollector:
    def __init__(self, symbol=SYMBOL, interval=INTERVAL):
        self.symbol = symbol.lower()
        self.interval = interval
        self.order_book = {'bids': [], 'asks': []}
        self.trade_data = []
        self.kline_data = pd.DataFrame()
        self.features = {}
        self.twm = ThreadedWebsocketManager(API_KEY, API_SECRET)

    def start(self):
        self.twm.start()
        self.twm.start_depth_socket(symbol=self.symbol, callback=self.handle_depth)
        self.twm.start_trade_socket(symbol=self.symbol, callback=self.handle_trade)
        self.twm.start_kline_socket(symbol=self.symbol, interval=self.interval, callback=self.handle_kline)

    def handle_depth(self, msg):
        self.order_book['bids'] = [(float(p), float(q)) for p, q in msg['bids']]
        self.order_book['asks'] = [(float(p), float(q)) for p, q in msg['asks']]
        self.compute_ob_features()

    def handle_trade(self, msg):
        trade = {
            'price': float(msg['p']),
            'qty': float(msg['q']),
            'time': msg['T'],
            'is_buyer_maker': msg['m']
        }
        self.trade_data.append(trade)
        if len(self.trade_data) > 500:
            self.trade_data = self.trade_data[-500:]

    def handle_kline(self, msg):
        k = msg['k']
        if k['x']:  # closed candle
            row = {
                'ts': datetime.utcfromtimestamp(k['t'] / 1000.0),
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v'])
            }
            self.kline_data = pd.concat([self.kline_data, pd.DataFrame([row])], ignore_index=True)
            self.kline_data = self.kline_data.tail(500)
            self.compute_technical_indicators()
            self.push_to_clickhouse()

    def compute_ob_features(self):
        bids = self.order_book['bids']
        asks = self.order_book['asks']
        if not bids or not asks:
            return

        bid_vol = sum(q for _, q in bids)
        ask_vol = sum(q for _, q in asks)
        ob_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        top_bid = bids[0][0]
        top_ask = asks[0][0]
        spread = top_ask - top_bid

        self.features['ob_imbalance'] = ob_imbalance
        self.features['spread'] = spread

    def compute_technical_indicators(self):
        if len(self.kline_data) < 20:
            return
        df = self.kline_data.copy()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['rsi'] = compute_rsi(df['close'])
        df['atr'] = compute_atr(df['high'], df['low'], df['close'])

        last = df.iloc[-1]
        self.features.update({
            'ema_10': last['ema_10'],
            'rsi': last['rsi'],
            'atr': last['atr']
        })

    def push_to_clickhouse(self):
        now = datetime.utcnow()
        row = {
            'ts': now,
            'symbol': SYMBOL,
            'features': json.dumps(self.features),
            'raw_data': json.dumps({
                'order_book': self.order_book,
                'trades': self.trade_data[-20:],
                'kline': self.kline_data.iloc[-1:].to_dict(orient='records')[0]
            })
        }
        try:
            clickhouse_client.insert(
                "INSERT INTO futures_features (ts, symbol, features, raw_data) VALUES",
                [row]
            )
            print(f"[{now}] Inserted features for {SYMBOL}")
        except Exception as e:
            logging.error(f"Failed to insert into ClickHouse: {e}")

# -------------------- Indicators --------------------

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rsi = 100 - (100 / (1 + ma_up / (ma_down + 1e-8)))
    return rsi

def compute_atr(high, low, close, period=14):
    tr = np.maximum(high[1:].values - low[1:].values,
                    np.abs(high[1:].values - close[:-1].values),
                    np.abs(low[1:].values - close[:-1].values))
    atr = pd.Series(tr).rolling(period).mean()
    return pd.Series([np.nan]*period + list(atr)).reindex_like(close)

# -------------------- Main --------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = CryptoDataCollector()
    collector.start()
    while True:
        time.sleep(1)
