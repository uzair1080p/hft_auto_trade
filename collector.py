"""
Collect real-time Binance data → ClickHouse
"""

import json, time, logging, os
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from binance import ThreadedWebsocketManager
from clickhouse_connect import get_client


# ──────────────────────── CONFIG ────────────────────────
API_KEY    = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
if not (API_KEY and API_SECRET):
    raise RuntimeError("Missing BINANCE_API_KEY / BINANCE_API_SECRET")

SYMBOL   = "ethusdt"
INTERVAL = "1m"

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", 8123))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASS = os.getenv("CLICKHOUSE_PASSWORD", "")

ck = get_client(
    host=CLICKHOUSE_HOST,
    port=CLICKHOUSE_PORT,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASS,
    compress=True,
)

# ──────────────────────── COLLECTOR ─────────────────────
class CryptoCollector:
    def __init__(self, symbol=SYMBOL, interval=INTERVAL):
        self.symbol      = symbol.lower()
        self.interval    = interval
        self.order_book  = {"bids": [], "asks": []}
        self.trades      = deque(maxlen=500)
        self.kline_data  = pd.DataFrame()
        self.features    = {}
        self.twm         = ThreadedWebsocketManager(API_KEY, API_SECRET, raise_if_any_error=True)

    # ───────── WS START
    def start(self):
        self.twm.start()
        self.twm.start_depth_socket(symbol=self.symbol,  callback=self.on_depth)
        self.twm.start_trade_socket(symbol=self.symbol,  callback=self.on_trade)
        self.twm.start_kline_socket(symbol=self.symbol, interval=self.interval, callback=self.on_kline)

    # ───────── handlers
    def on_depth(self, msg: dict):
        # Binance depth stream uses keys 'b' (bids) and 'a' (asks)
        self.order_book["bids"] = [(float(p), float(q)) for p, q in msg["b"]]
        self.order_book["asks"] = [(float(p), float(q)) for p, q in msg["a"]]
        self._ob_features()

    def on_trade(self, msg: dict):
        self.trades.append(
            {
                "price": float(msg["p"]),
                "qty":   float(msg["q"]),
                "time":  msg["T"],
                "is_buyer_maker": msg["m"],
            }
        )

    def on_kline(self, msg: dict):
        k = msg["k"]
        if not k["x"]:        # candle not closed yet
            return
        row = {
            "ts":     datetime.utcfromtimestamp(k["t"] / 1000),
            "open":   float(k["o"]),
            "high":   float(k["h"]),
            "low":    float(k["l"]),
            "close":  float(k["c"]),
            "volume": float(k["v"]),
        }
        self.kline_data = pd.concat([self.kline_data, pd.DataFrame([row])], ignore_index=True).tail(500)
        self._tech_indicators()
        self._push_clickhouse()

    # ───────── feature calculators
    def _ob_features(self):
        bids, asks = self.order_book["bids"], self.order_book["asks"]
        if not (bids and asks):
            return
        bid_vol, ask_vol = (sum(q for _, q in bids), sum(q for _, q in asks))
        self.features.update(
            ob_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8),
            spread       = asks[0][0] - bids[0][0],
        )

    def _tech_indicators(self):
        if len(self.kline_data) < 20:
            return
        df = self.kline_data
        df["ema_10"] = df["close"].ewm(span=10).mean()
        df["rsi"]    = _rsi(df["close"])
        df["atr"]    = _atr(df["high"], df["low"], df["close"])
        last = df.iloc[-1]
        self.features.update(ema_10=last.ema_10, rsi=last.rsi, atr=last.atr)

    # ───────── ClickHouse insert
    def _push_clickhouse(self):
        row = {
            "ts": datetime.utcnow(),
            "symbol": SYMBOL.upper(),
            "features": json.dumps(self.features),
            "raw_data": json.dumps(
                {
                    "order_book": self.order_book,
                    "trades":     list(self.trades)[-20:],
                    "kline":      self.kline_data.iloc[-1].to_dict(),
                },
                default=str,           # serialize Timestamp
            ),
        }
        try:
            ck.insert(
                "INSERT INTO futures_features (ts, symbol, features, raw_data) VALUES",
                [row],
            )
        except Exception as e:
            logging.error(f"ClickHouse insert failed: {e!s}")

# ─────────────────── helper indicators ──────────────────
def _rsi(series, period=14):
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    ma_up, ma_down = up.rolling(period).mean(), down.rolling(period).mean()
    return 100 - 100 / (1 + ma_up / (ma_down + 1e-8))

def _atr(h, l, c, period=14):
    tr = np.maximum(h.iloc[1:].values - l.iloc[1:].values,
                    np.abs(h.iloc[1:].values - c.iloc[:-1].values),
                    np.abs(l.iloc[1:].values - c.iloc[:-1].values))
    return pd.Series(np.r_[np.full(period, np.nan), pd.Series(tr).rolling(period).mean()], index=c.index)

# ───────────────────────── MAIN ─────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = CryptoCollector()
    collector.start()
    while True:
        time.sleep(1)