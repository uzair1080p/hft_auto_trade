#!/usr/bin/env python3

"""
Simple signal generator for testing live trading.
Generates basic buy/sell signals based on RSI to test the trading executor.
"""

import time
import json
import logging
import os
from datetime import datetime
from clickhouse_connect import get_client

# Config
CLICKHOUSE_HOST = os.getenv('CLICKHOUSE_HOST', 'clickhouse')
CLICKHOUSE_USER = os.getenv('CLICKHOUSE_USER', 'default')
CLICKHOUSE_PASS = os.getenv('CLICKHOUSE_PASS', '')
SYMBOL = os.getenv('SYMBOL', 'DOGEUSDT')

# ClickHouse client
client = get_client(
    host=CLICKHOUSE_HOST,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASS,
    compress=True
)

def get_latest_rsi():
    """Get the latest RSI value from features."""
    try:
        query = f"""
        SELECT features 
        FROM futures_features 
        WHERE symbol = '{SYMBOL}' 
        ORDER BY ts DESC 
        LIMIT 1
        """
        result = client.query(query)
        if result.result_rows:
            features = json.loads(result.result_rows[0][0])
            return features.get('rsi', 50.0)
        return 50.0
    except Exception as e:
        logging.error(f"Error getting RSI: {e}")
        return 50.0

def generate_signal(rsi):
    """Generate signal based on RSI."""
    if rsi < 30:
        return 1, 0.8  # Strong buy
    elif rsi < 40:
        return 1, 0.6  # Buy
    elif rsi > 70:
        return -1, 0.8  # Strong sell
    elif rsi > 60:
        return -1, 0.6  # Sell
    else:
        return 0, 0.5  # Hold

def log_signal(signal, score):
    """Log signal to executed_trades table."""
    try:
        ts = datetime.utcnow()
        rsi = get_latest_rsi()
        # Insert as a list of tuples with proper column names
        client.insert(
            "executed_trades", 
            [[ts, SYMBOL, signal, score]],
            column_names=['ts', 'symbol', 'signal', 'score']
        )
        logging.info(f"[{ts}] Signal: {signal}, Score: {score:.4f}, RSI: {rsi:.2f}")
    except Exception as e:
        logging.error(f"Error logging signal: {e}")
        logging.error(f"Signal data: ts={datetime.utcnow()}, symbol={SYMBOL}, signal={signal}, score={score}")

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Simple signal generator started for {SYMBOL}")
    
    last_signal = None
    
    while True:
        try:
            rsi = get_latest_rsi()
            signal, score = generate_signal(rsi)
            
            # Only log if signal changed
            if last_signal != signal:
                log_signal(signal, score)
                last_signal = signal
            
            time.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            logging.error(f"Error in signal loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
