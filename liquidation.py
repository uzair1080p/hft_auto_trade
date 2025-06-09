"""
Consume !forceOrder liquidation stream → ClickHouse
"""

import json, time, logging, os, threading, websocket
from datetime import datetime, timedelta
from collections import deque

from clickhouse_connect import get_client

# ─────────────────────── CONFIG ───────────────────────
API_KEY    = os.getenv("BINANCE_API_KEY")   # kept for parity (not used by public stream)
API_SECRET = os.getenv("BINANCE_API_SECRET")
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

# ensure table exists (idempotent)
ck.command("""
CREATE TABLE IF NOT EXISTS liquidation_features (
    timestamp                DateTime,
    symbol                   String,
    liquidation_cluster_size Float64,
    liquidation_imbalance    Float64
) ENGINE = MergeTree ORDER BY (symbol, timestamp)
""")

# ───────────────── CLUSTER ACCUMULATOR ────────────────
class LiquidationWindow:
    def __init__(self, window_sec: int = 30):
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
    last_flush = time.time()

    def on_msg(_, raw: str):
        """WebSocket message handler"""
        try:
            msg = json.loads(raw)
            # msg format from !forceOrder@arr
            side  = "buy"  if msg["S"] == "SELL" else "sell"  # inverse
            price = float(msg["p"])
            qty   = float(msg["q"])
            ts    = datetime.utcfromtimestamp(msg["T"] / 1000)
            window.add(ts, side, price * qty)
        except Exception as e:
            logging.error(f"Liquidation parse error: {e}")

    # ---- connect to global liquidation stream (no auth required) ----
    url = "wss://fstream.binance.com/ws/!forceOrder@arr"
    ws_app = websocket.WebSocketApp(url, on_message=on_msg)

    def _run_ws():
        while True:
            try:
                ws_app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logging.error(f"WebSocket error: {e}; reconnecting in 5s")
                time.sleep(5)

    threading.Thread(target=_run_ws, daemon=True).start()

    # ---- periodic ClickHouse flush ----
    try:
        while True:
            time.sleep(1)
            if time.time() - last_flush >= 30:
                row = window.snapshot()
                ck.insert("liquidation_features", [row])
                logging.info(f"[{row['timestamp']}] Logged liquidation cluster")
                last_flush = time.time()
    except KeyboardInterrupt:
        logging.info("Shutdown requested — exiting")

if __name__ == "__main__":
    main()