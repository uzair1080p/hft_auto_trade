"""
Consume !forceOrder liquidation stream → ClickHouse
"""

import json, time, logging, os, threading, websocket
from datetime import datetime, timedelta
from collections import deque
from clickhouse_connect import get_client

# ─────────────────────── CONFIG ───────────────────────
SYMBOL = "ethusdt"      # lower-case for endpoint

CLICKHOUSE = dict(
    host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
    port=int(os.getenv("CLICKHOUSE_PORT", 8123)),
    username=os.getenv("CLICKHOUSE_USER", "default"),
    password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    compress=True,
)

ck = get_client(**CLICKHOUSE)

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
    def __init__(self, secs: int = 30):
        self.window = timedelta(seconds=secs)
        self.events = deque()   # (ts, side, value_usd)

    def add(self, ts, side, value):
        self.events.append((ts, side, value))
        cutoff = ts - self.window
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
    win        = LiquidationWindow()
    last_flush = time.time()

    def on_msg(_, raw: str):
        """
        Binance sends either:
        • aggregated {"e":"forceOrder","o":{...}}
        • raw order  {"o":{...}}
        • legacy     {"S": ... , "p": ...}
        Always normalise to `order = msg["o"] or msg`.
        """
        try:
            msg   = json.loads(raw)
            order = msg.get("o") or msg    # tolerate both shapes
            side  = "buy" if order["S"] == "SELL" else "sell"  # inverse
            price = float(order.get("ap") or order["p"])
            qty   = float(order["q"])
            ts    = datetime.utcfromtimestamp(order["T"] / 1000)
            win.add(ts, side, price * qty)
        except (KeyError, ValueError):
            logging.debug(f"Ignoring non-order message: {raw[:100]}…")
        except Exception as exc:
            logging.error(f"Liquidation parse error: {exc}")

    url    = "wss://fstream.binance.com/ws/!forceOrder@arr"
    ws_app = websocket.WebSocketApp(url, on_message=on_msg)

    def _run_ws():
        while True:
            try:
                ws_app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                logging.error(f"WebSocket error: {exc} – reconnecting in 5 s")
                time.sleep(5)

    threading.Thread(target=_run_ws, daemon=True).start()

    try:
        while True:
            time.sleep(1)
            if time.time() - last_flush >= 30:
                row = win.snapshot()
                ck.insert("liquidation_features", [row])
                logging.info(f"[{row['timestamp']}] Logged liquidation cluster")
                last_flush = time.time()
    except KeyboardInterrupt:
        logging.info("Shutdown requested — exiting")

if __name__ == "__main__":
    main()