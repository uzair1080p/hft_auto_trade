"""
Consume !forceOrder liquidation stream → ClickHouse
"""

import json, time, logging, os
from datetime import datetime, timedelta
from collections import deque

from binance import ThreadedWebsocketManager
from clickhouse_connect import get_client

# ─────────────────────── CONFIG ───────────────────────
API_KEY    = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
if not (API_KEY and API_SECRET):
    raise RuntimeError("Missing Binance keys")

SYMBOL = "ethusdt"      # lower-case for endpoint
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

# ───────────────── CLUSTER ACCUMULATOR ────────────────
class LiquidationWindow:
    def __init__(self, window_sec=30):
        self.window = timedelta(seconds=window_sec)
        self.events = deque()   # (ts, side, value_usd)

    def add(self, ts, side, value):
        self.events.append((ts, side, value))
        self._prune(ts)

    def _prune(self, now):
        cutoff = now - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def snapshot(self):
        longs  = sum(v for _, s, v in self.events if s == "buy")
        shorts = sum(v for _, s, v in self.events if s == "sell")
        total  = longs + shorts
        return {
            "timestamp": datetime.utcnow(),
            "symbol": SYMBOL.upper(),
            "liquidation_cluster_size": total,
            "liquidation_imbalance": (longs - shorts) / (total + 1e-8),
        }

# ───────────────────────── Main ───────────────────────
def main():
    logging.basicConfig(level=logging.INFO)
    window = LiquidationWindow()

    def on_msg(msg: dict):
        # msg format from !forceOrder@arr stream
        side  = "buy"  if msg["S"] == "SELL" else "sell"  # inverse
        price = float(msg["p"])
        qty   = float(msg["q"])
        ts    = datetime.utcfromtimestamp(msg["T"] / 1000)
        window.add(ts, side, price * qty)

    twm = ThreadedWebsocketManager(API_KEY, API_SECRET)
    twm.start()
    # generic socket path for global liquidation stream
    twm.start_socket(callback=on_msg, path="/ws/!forceOrder@arr")

    try:
        while True:
            time.sleep(30)
            row = window.snapshot()
            ck.insert("INSERT INTO liquidation_features VALUES", [row])
            logging.info(f"[{row['timestamp']}] Logged liquidation cluster")
    except KeyboardInterrupt:
        twm.stop()

if __name__ == "__main__":
    main()