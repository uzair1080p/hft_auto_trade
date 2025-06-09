"""
Consume !forceOrder liquidation stream → ClickHouse
Requires `websocket-client` and `clickhouse-connect`
"""

import json, time, logging, os, threading, websocket                 # pip install websocket-client
from datetime import datetime, timedelta
from collections import deque
from typing import Deque, Tuple
from clickhouse_connect import get_client

# ───────────────────── CONFIG ─────────────────────
SYMBOL = "ethusdt"                     # endpoint expects lower-case

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
    """
    Maintains last N-seconds of liquidation events and builds a
    cluster snapshot on demand.
    """
    def __init__(self, secs: int = 30):
        self.window: timedelta = timedelta(seconds=secs)
        self.events: Deque[Tuple[datetime, str, float]] = deque()  # (ts, side, usd_value)

    def add(self, ts: datetime, side: str, value: float) -> None:
        self.events.append((ts, side, value))
        cutoff = ts - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def snapshot(self) -> Tuple:
        longs  = sum(v for _, s, v in self.events if s == "buy")
        shorts = sum(v for _, s, v in self.events if s == "sell")
        total  = longs + shorts
        imbalance = (longs - shorts) / (total + 1e-8)
        return (
            datetime.utcnow(),          # timestamp
            SYMBOL.upper(),             # symbol
            total,                      # liquidation_cluster_size
            imbalance,                  # liquidation_imbalance
        )

# ─────────────────── WebSocket consumer ──────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    window      = LiquidationWindow()
    last_flush  = time.time()

    def on_msg(_, raw: str) -> None:
        """
        Handle every incoming frame from !forceOrder stream.
        Shapes observed:
        • {"e":"forceOrder","o":{...}}
        • {"o":{...}}
        """
        try:
            msg   = json.loads(raw)
            order = msg.get("o") or msg
            side  = "buy" if order["S"] == "SELL" else "sell"       # inverse
            price = float(order.get("ap") or order["p"])
            qty   = float(order["q"])
            ts    = datetime.utcfromtimestamp(order["T"] / 1000)
            window.add(ts, side, price * qty)
        except (KeyError, ValueError):
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

    # ── periodic ClickHouse flush every 30 s ──
    try:
        while True:
            time.sleep(1)
            if time.time() - last_flush >= 30:
                row_tuple = window.snapshot()      # tuple in correct order

                ck.insert(
                    "liquidation_features",
                    [row_tuple],                   # list-of-tuples ✓
                    column_names=[
                        "timestamp",
                        "symbol",
                        "liquidation_cluster_size",
                        "liquidation_imbalance",
                    ],
                )

                logging.info(
                    "[%s] cluster size=%0.0f  imbalance=%+.3f",
                    row_tuple[0], row_tuple[2], row_tuple[3]
                )
                last_flush = time.time()
    except KeyboardInterrupt:
        logging.info("Shutdown requested — exiting")

if __name__ == "__main__":
    main()