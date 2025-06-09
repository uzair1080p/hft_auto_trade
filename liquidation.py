# liquidation.py

"""
Consumes Binance liquidation events (!forceOrder) and computes liquidation cluster metrics.
Stores results in ClickHouse every 30 seconds.
"""

import json
import time
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from binance import ThreadedWebsocketManager
from clickhouse_connect import get_client
import os

# -------------------- Config --------------------

API_KEY    = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

if not API_KEY or not API_SECRET:          # fail fast if keys are missing
    raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET not set")

SYMBOL = 'ethusdt'
CLICKHOUSE_HOST = 'clickhouse'
CLICKHOUSE_USER = 'default'
CLICKHOUSE_PASS = os.getenv("CLICKHOUSE_PASSWORD")

# -------------------- ClickHouse Client --------------------

client = get_client(
    host=CLICKHOUSE_HOST,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASS,
    compress=True
)

# -------------------- Liquidation Processor --------------------

class LiquidationProcessor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.events = deque()
        self.window = timedelta(seconds=30)

    def process_event(self, event):
        ts = datetime.utcfromtimestamp(event['T'] / 1000.0)
        price = float(event['p'])
        qty = float(event['q'])
        side = 'buy' if event['S'] == 'SELL' else 'sell'  # inverse
        value_usd = price * qty
        self.events.append((ts, side, price, value_usd))
        self._clean_old(ts)

    def _clean_old(self, now):
        while self.events and self.events[0][0] < now - self.window:
            self.events.popleft()

    def get_cluster_features(self):
        longs = sum(val for ts, side, p, val in self.events if side == 'buy')
        shorts = sum(val for ts, side, p, val in self.events if side == 'sell')
        total = longs + shorts
        imbalance = (longs - shorts) / (total + 1e-8)
        return {
            'timestamp': datetime.utcnow(),
            'symbol': self.symbol.upper(),
            'liquidation_cluster_size': total,
            'liquidation_imbalance': imbalance
        }

# -------------------- WebSocket Consumer --------------------

def main():
    logging.basicConfig(level=logging.INFO)
    proc = LiquidationProcessor(SYMBOL)

    def on_message(msg):
        proc.process_event(msg)

    twm = ThreadedWebsocketManager(API_KEY, API_SECRET)
    twm.start()
    twm.start_liquidation_socket(callback=on_message, symbol=SYMBOL)

    try:
        while True:
            time.sleep(30)
            features = proc.get_cluster_features()
            client.insert("liquidation_features", [features])
            logging.info(f"[{features['timestamp']}] Logged: {features}")
    except KeyboardInterrupt:
        twm.stop()

if __name__ == "__main__":
    main()
