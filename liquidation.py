"""
Consume !forceOrder liquidation stream → ClickHouse
"""

import json, time, logging, os, threading, websocket
from datetime import datetime, timedelta
from collections import deque
from clickhouse_connect import get_client

# ───────────────────── CONFIG ─────────────────────
SYMBOL = "ethusdt"                 # endpoint expects lower-case

CLICKHOUSE = dict(
    host=os.getenv("CLICKHOUSE_HOST", "clickhouse"),
    port=int(os.getenv("CLICKHOUSE_PORT", 8123)),
    username=os.getenv("CLICKHOUSE_USER", "default"),
    password=os.getenv("CLICKHOUSE_PASSWORD", ""),
    compress=True,
)

ck = get_client(**CLICKHOUSE)

# idempotent DDL
ck.command("""
CREATE TABLE IF NOT EXISTS liquidation_features (
    timestamp                DateTime,
    symbol                   String,
    liquidation_cluster_size Float64,
    liquidation_imbalance    Float64
) ENGINE = MergeTree ORDER BY (symbol, timestamp)
""")

# ────────── sliding-window accumulator ──────────
class LiquidationWindow:
    def __init__(self, secs: int = 30):
        self.window = timedelta(seconds=secs)
        self.events: deque = deque()    # (ts, side, usd_value)

    def add(self, ts: datetime, side: str, value: float) -> None:
        """Insert new event and prune anything older than window"""
        self.events.append((ts, side, value))
        cutoff = ts - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def snapshot(self) -> dict:
        longs  = sum(v for _, s, v in self.events if s == "buy")
        shorts = sum(v for _, s, v in self.events if s == "sell")
        total  = longs + shorts
        return {
            "timestamp": datetime.utcnow(),
            "symbol":    SYMBOL.upper(),
            "liquidation_cluster_size": total,
            "liquidation_imbalance":    (longs - shorts) / (total + 1e-8),
        }

# ─────────────────── WebSocket consumer ──────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    window      = LiquidationWindow()
    last_flush  = time.time()

    # -- handle each incoming frame --
    def on_msg(_, raw: str) -> None:
        """
        Binance !forceOrder messages appear in two shapes:
        • {"e":"forceOrder","o":{...}}
        • {"o":{...}}
        Normalise by reading `order = msg.get("o") or msg`.
        """
        try:
            msg   = json.loads(raw)
            order = msg.get("o") or msg
            side  = "buy" if order["S"] == "SELL" else "sell"   # inverse
            price = float(order.get("ap") or order["p"])        # average-price if present
            qty   = float(order["q"])
            ts    = datetime.utcfromtimestamp(order["T"] / 1000)
            window.add(ts, side, price * qty)
        except (KeyError, ValueError):
            # ignore keep-alives / malformed frames
            logging.debug("non-order frame ignored")
        except Exception as exc:
            logging.error("Liquidation parse error: %s", exc)

    # keep websocket alive forever — reconnect on error
    def ws_loop() -> None:
        url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        ws  = websocket.WebSocketApp(url, on_message=on_msg)
        while True:
            try:
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                logging.error("WebSocket error: %s — reconnecting in 5 s", exc)
                time.sleep(5)

    threading.Thread(target=ws_loop, daemon=True).start()

    # -- periodic ClickHouse flush every 30 s --
    try:
        while True:
            time.sleep(1)
            if time.time() - last_flush >= 30:
                row = window.snapshot()

                ck.insert(
                    "liquidation_features",
                    [row],
                    column_names=[
                        "timestamp",
                        "symbol",
                        "liquidation_cluster_size",
                        "liquidation_imbalance",
                    ],
                )

                logging.info(
                    "[%s] cluster size=%0.0f  imbalance=%+.3f",
                    row["timestamp"],
                    row["liquidation_cluster_size"],
                    row["liquidation_imbalance"],
                )
                last_flush = time.time()
    except KeyboardInterrupt:
        logging.info("Shutdown requested — exiting")

if __name__ == "__main__":
    main()