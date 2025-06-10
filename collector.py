"""
Real-time collector:
  • depth   → order-book imbalance, spread
  • kline   → EMA-10, RSI-14, ATR-14
Stores one JSON feature row per closed 1-minute candle.
"""

import json, os, time, logging
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from binance import Client, BinanceSocketManager
from clickhouse_connect import get_client

# ───── CONFIG ─────
API_KEY    = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOL_UC  = "ETHUSDT"           # upper-case for kline stream
SYMBOL_LC  = SYMBOL_UC.lower()   # lower-case for depth stream
INTERVAL   = "1m"

ck = get_client(
    host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
    username=os.getenv("CLICKHOUSE_USER", "default"),
    password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    compress=True,
)

ck.command("""
CREATE TABLE IF NOT EXISTS futures_features (
    ts DateTime,
    symbol String,
    features JSON,
    raw_data JSON
) ENGINE = MergeTree ORDER BY (symbol, ts)
""")

# ───── indicators ─────
def rsi(series, p=14):
    delta = series.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    return 100 - 100 / (1 + up.rolling(p).mean() / (dn.rolling(p).mean() + 1e-8))

def atr(h,l,c,p=14):
    tr = np.maximum(h[1:].values-l[1:].values,
                    np.abs(h[1:].values-c[:-1].values),
                    np.abs(l[1:].values-c[:-1].values))
    return pd.Series(np.r_[np.full(p,np.nan), pd.Series(tr).rolling(p).mean()],
                     index=c.index)

# ───── collector class ─────
class Collector:
    def __init__(self):
        self.order_book = {"bids": [], "asks": []}
        self.kline_df   = pd.DataFrame()
        self.features   = {}
        self.clt        = Client(API_KEY, API_SECRET)
        self.bsm        = BinanceSocketManager(self.clt)

    async def on_depth(self, msg):
        self.order_book["bids"] = [(float(p), float(q)) for p,q in msg["b"]]
        self.order_book["asks"] = [(float(p), float(q)) for p,q in msg["a"]]
        bids, asks = self.order_book["bids"], self.order_book["asks"]
        if bids and asks:
            bv, av = sum(q for _,q in bids), sum(q for _,q in asks)
            self.features.update(
                ob_imbalance=(bv-av)/(bv+av+1e-8),
                spread      =asks[0][0]-bids[0][0],
            )

    async def on_kline(self, msg):
        k = msg["k"]
        if not k["x"]:                          # only closed candles
            return
        row = dict(
            ts=datetime.utcfromtimestamp(k["t"]/1000),
            open=float(k["o"]), high=float(k["h"]),
            low=float(k["l"]),  close=float(k["c"]),
            volume=float(k["v"]),
        )
        self.kline_df = pd.concat([self.kline_df,
                                   pd.DataFrame([row])]).tail(500)
        if len(self.kline_df) >= 5:             # warm-up shortened
            df = self.kline_df
            df["ema10"] = df["close"].ewm(span=10).mean()
            df["rsi"]   = rsi(df["close"])
            df["atr"]   = atr(df["high"], df["low"], df["close"])
            last        = df.iloc[-1]
            self.features.update(
                ema10=last.ema10,
                rsi  =last.rsi,
                atr  =last.atr,
            )
            ck.insert(
                "futures_features",
                [{
                    "ts": datetime.utcnow(),
                    "symbol": SYMBOL_UC,
                    "features": json.dumps(self.features),
                    "raw_data": json.dumps(
                        {"order_book": self.order_book,
                         "kline": row},
                        default=str),
                }],
            )
            logging.info("Inserted features row")

    def run(self):
        loop = self.bsm._loop
        loop.create_task(
            self.bsm.start_depth_socket(symbol=SYMBOL_LC, callback=self.on_depth)
        )
        loop.create_task(
            self.bsm.start_kline_socket(symbol=SYMBOL_UC, interval=INTERVAL, callback=self.on_kline)
        )
        self.bsm.start()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    Collector().run()