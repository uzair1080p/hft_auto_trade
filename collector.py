"""
Real-time collector for ETH/USDT futures
-------------------------------------------------
• depth      → order-book imbalance + spread
• kline 1 m  → EMA-10, RSI-14, ATR-14
Each closed candle ⇒ one JSON row in ClickHouse `futures_features`.
"""

import json, os, time, logging
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from binance import Client, ThreadedWebsocketManager   # ← changed manager!
from clickhouse_connect import get_client

# ─────────────────────── CONFIG ───────────────────────
API_KEY    = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOL_UC  = "ETHUSDT"            # upper-case for kline stream
SYMBOL_LC  = SYMBOL_UC.lower()    # lower-case for depth stream
INTERVAL   = "1m"

CK_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CK_USER = os.getenv("CLICKHOUSE_USER", "default")
CK_PASS = os.getenv("CLICKHOUSE_PASSWORD", "")

# ---------- ClickHouse client (retry until server is up) ----------
for attempt in range(30):
    try:
        ck = get_client(host=CK_HOST, username=CK_USER,
                        password=CK_PASS, compress=True)
        ck.command("SELECT 1")
        break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError("❌  ClickHouse not reachable after 60 s")

# auto-DDL (idempotent)
ck.command("""
CREATE TABLE IF NOT EXISTS futures_features (
    ts        DateTime,
    symbol    String,
    features  JSON,
    raw_data  JSON
) ENGINE = MergeTree ORDER BY (symbol, ts)
""")

# ───────────────────── technical indicators ──────────────────────
def rsi(series: pd.Series, p: int = 14) -> pd.Series:
    delta = series.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    return 100 - 100 / (1 + up.rolling(p).mean() / (dn.rolling(p).mean() + 1e-8))

def atr(h, l, c, p: int = 14) -> pd.Series:
    tr = np.maximum(h[1:].values - l[1:].values,
                    np.abs(h[1:].values - c[:-1].values),
                    np.abs(l[1:].values - c[:-1].values))
    return pd.Series(np.r_[np.full(p, np.nan), pd.Series(tr).rolling(p).mean()],
                     index=c.index)

# ───────────────────────── Collector ─────────────────────────────
class Collector:
    def __init__(self):
        self.order_book = {"bids": [], "asks": []}
        self.kline_df   = pd.DataFrame()
        self.features   = {}

        self.client = Client(API_KEY, API_SECRET)
        self.twm    = ThreadedWebsocketManager(API_KEY, API_SECRET)  # ← new

    # ---------- order-book depth ----------
    def on_depth(self, msg):
        self.order_book["bids"] = [(float(p), float(q)) for p, q in msg["b"]]
        self.order_book["asks"] = [(float(p), float(q)) for p, q in msg["a"]]
        bids, asks = self.order_book["bids"], self.order_book["asks"]
        if bids and asks:
            bv, av = sum(q for _, q in bids), sum(q for _, q in asks)
            self.features.update(
                ob_imbalance=(bv - av) / (bv + av + 1e-8),
                spread=asks[0][0] - bids[0][0],
            )

    # ---------- closed 1-minute kline ----------
    def on_kline(self, msg):
        k = msg["k"]
        if not k["x"]:                       # ignore live (unfinished) candles
            return

        row = dict(
            ts=datetime.utcfromtimestamp(k["t"] / 1000),
            open=float(k["o"]), high=float(k["h"]),
            low=float(k["l"]),  close=float(k["c"]),
            volume=float(k["v"]),
        )
        self.kline_df = pd.concat([self.kline_df, pd.DataFrame([row])]).tail(500)

        if len(self.kline_df) < 20:          # minimal warm-up
            return

        df = self.kline_df
        df["ema10"] = df["close"].ewm(span=10).mean()
        df["rsi"]   = rsi(df["close"])
        df["atr"]   = atr(df["high"], df["low"], df["close"])
        last        = df.iloc[-1]

        self.features.update(ema10=last.ema10,
                             rsi=last.rsi,
                             atr=last.atr)

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
        logging.info("✅  Inserted features row")

    # ---------- start sockets ----------
    def run(self):
        self.twm.start()
        self.twm.start_depth_socket(symbol=SYMBOL_LC, callback=self.on_depth)
        self.twm.start_kline_socket(symbol=SYMBOL_UC,
                                    interval=INTERVAL,
                                    callback=self.on_kline)

        # keep main thread alive
        while True:
            time.sleep(60)

# ────────────────────────── main ────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    Collector().run()