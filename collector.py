"""
Real-time collector for ETH-USDT futures
────────────────────────────────────────────────────────
• depth  → order-book imbalance & spread
• kline  → EMA-10, RSI-14, ATR-14            (1-minute candles)

Each closed candle ➜ one JSON row in ClickHouse `futures_features`.
"""

import json, os, sys, socket, time, logging
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from binance import Client, ThreadedWebsocketManager
from clickhouse_connect import get_client


# ───────────────────────── ClickHouse bootstrap ──────────────────────────
def wait_clickhouse(host: str = "clickhouse", port: int = 8123, timeout: int = 90):
    """Block until the ClickHouse HTTP port is reachable (or die after timeout)."""
    logging.info("⏳ waiting for ClickHouse %s:%d …", host, port)
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), 2):
                logging.info("✅ ClickHouse is up")
                return
        except OSError:
            time.sleep(2)
    sys.exit("❌ ClickHouse not reachable after 90 s")


CK_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CK_PORT = int(os.getenv("CLICKHOUSE_PORT", 8123))
wait_clickhouse(CK_HOST, CK_PORT)

ck = get_client(
    host=CK_HOST,
    username=os.getenv("CLICKHOUSE_USER", "default"),
    password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    compress=True,
)
ck.command("""SELECT 1""")        # quick sanity ping

ck.command(
    """
CREATE TABLE IF NOT EXISTS futures_features (
    ts        DateTime,
    symbol    String,
    features  JSON,
    raw_data  JSON
) ENGINE = MergeTree ORDER BY (symbol, ts)
"""
)

# ────────────────────────── constants ────────────────────────────
API_KEY    = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOL_UC  = "ETHUSDT"           # kline stream
SYMBOL_LC  = SYMBOL_UC.lower()   # depth stream
INTERVAL   = "1m"
RSI_P      = 14
ATR_P      = 14
EMA_SPAN   = 10
WARM_UP    = max(RSI_P, ATR_P) + 1   # candles before first indicator row


# ──────────────────────── indicator helpers ──────────────────────
def rsi(close: pd.Series, p: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(p, min_periods=p).mean()
    avg_loss = loss.rolling(p, min_periods=p).mean()
    rs  = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, p: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(p, min_periods=p).mean()


# ───────────────────────── Collector class ───────────────────────
class Collector:
    def __init__(self):
        self.order_book = {"bids": [], "asks": []}
        self.kline_df   = pd.DataFrame()
        self.features   = {}

        self.client = Client(API_KEY, API_SECRET)
        self.twm    = ThreadedWebsocketManager(API_KEY, API_SECRET)

    # depth updates ───────────────────────────────────────────────
    def on_depth(self, msg: dict):
        self.order_book["bids"] = [(float(p), float(q)) for p, q in msg["b"]]
        self.order_book["asks"] = [(float(p), float(q)) for p, q in msg["a"]]

        bids, asks = self.order_book["bids"], self.order_book["asks"]
        if bids and asks:
            bid_vol, ask_vol = sum(q for _, q in bids), sum(q for _, q in asks)
            self.features.update(
                ob_imbalance=(bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8),
                spread=asks[0][0] - bids[0][0],
            )

    # closed-candle handler ───────────────────────────────────────
    def on_kline(self, msg: dict):
        k = msg["k"]
        if not k["x"]:                 # skip live (in-flight) candles
            return

        row = dict(
            ts=datetime.utcfromtimestamp(k["t"] / 1000),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
        )
        self.kline_df = pd.concat([self.kline_df, pd.DataFrame([row])]).tail(500)

        if len(self.kline_df) < WARM_UP:
            return

        df = self.kline_df.copy()
        df["ema10"] = df["close"].ewm(span=EMA_SPAN).mean()
        df["rsi"]   = rsi(df["close"], RSI_P)
        df["atr"]   = atr(df["high"], df["low"], df["close"], ATR_P)
        last        = df.iloc[-1]

        self.features.update(ema10=last.ema10, rsi=last.rsi, atr=last.atr)

        ck.insert(
            "futures_features",
            [
                {
                    "ts": datetime.utcnow(),
                    "symbol": SYMBOL_UC,
                    "features": json.dumps(self.features),
                    "raw_data": json.dumps(
                        {"order_book": self.order_book, "kline": row},
                        default=str,
                    ),
                }
            ],
        )
        logging.info("✅ Inserted features row")

    # ──────────────── start streams & keep alive ─────────────────
    def run(self):
        self.twm.start()
        self.twm.start_depth_socket(symbol=SYMBOL_LC, callback=self.on_depth)
        self.twm.start_kline_socket(symbol=SYMBOL_UC, interval=INTERVAL, callback=self.on_kline)

        # keep the main thread alive forever
        while True:
            time.sleep(60)


# ────────────────────────────── main ─────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    Collector().run()