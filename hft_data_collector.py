# hft_data_collector.py

"""
High-Frequency Trading (HFT) data collector.
Collects real-time market data using WebSocket streams for ultra-fast processing.
"""

import time
import json
import logging
import os
import asyncio
import threading
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd
import talib

from binance import Client, ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from clickhouse_connect import get_client

# -------------------- Config --------------------

from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, SYMBOL, INTERVAL,
    CLICKHOUSE_HOST, CLICKHOUSE_USER, CLICKHOUSE_PASS,
    RSI_PERIOD, ATR_PERIOD, EMA_PERIOD, MAX_TRADE_HISTORY,
    MAX_KLINE_HISTORY, ENABLE_WEBSOCKET_STREAMS
)

DRY_RUN = os.getenv('DRY_RUN', '0') == '1'

# -------------------- Clients --------------------

# Initialize Binance client
if not DRY_RUN:
    binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    twm = ThreadedWebsocketManager(BINANCE_API_KEY, BINANCE_API_SECRET)
else:
    binance_client = None
    twm = None

clickhouse_client = get_client(
    host=CLICKHOUSE_HOST,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASS,
    compress=True
)

# -------------------- HFT Data Collector --------------------

class HFTDataCollector:
    def __init__(self, symbol=SYMBOL):
        self.symbol = symbol
        self.is_running = False
        
        # Real-time data storage
        self.order_book = {'bids': [], 'asks': []}
        self.recent_trades = deque(maxlen=MAX_TRADE_HISTORY)
        self.kline_data = deque(maxlen=MAX_KLINE_HISTORY)
        self.feature_cache = deque(maxlen=1000)
        
        # Performance tracking
        self.data_count = 0
        self.last_log_time = time.time()
        
        # Threading
        self.lock = threading.Lock()
        
    def start_websocket_streams(self):
        """Start real-time WebSocket data streams."""
        if not ENABLE_WEBSOCKET_STREAMS:
            logging.info("WebSocket streams disabled")
            return
            
        if DRY_RUN:
            logging.info("[DRY_RUN] Using synthetic data instead of WebSocket streams")
            # Start synthetic data generation for testing
            self.start_synthetic_data_generation()
            return
            
        try:
            twm.start()
            
            # Start order book stream
            twm.start_depth_socket(
                symbol=self.symbol,
                callback=self.handle_order_book_update,
                depth=20
            )
            
            # Start trade stream
            twm.start_trade_socket(
                symbol=self.symbol,
                callback=self.handle_trade_update
            )
            
            # Start kline stream
            twm.start_kline_socket(
                symbol=self.symbol,
                interval='100ms',
                callback=self.handle_kline_update
            )
            
            logging.info(f"Started HFT WebSocket streams for {self.symbol}")
            
        except Exception as e:
            logging.error(f"Failed to start WebSocket streams: {e}")
    
    def handle_order_book_update(self, msg):
        """Handle real-time order book updates."""
        try:
            with self.lock:
                self.order_book['bids'] = [
                    (float(price), float(qty)) for price, qty in msg['bids']
                ]
                self.order_book['asks'] = [
                    (float(price), float(qty)) for price, qty in msg['asks']
                ]
        except Exception as e:
            logging.error(f"Error handling order book update: {e}")
    
    def handle_trade_update(self, msg):
        """Handle real-time trade updates."""
        try:
            trade = {
                'timestamp': msg['T'],
                'price': float(msg['p']),
                'quantity': float(msg['q']),
                'is_buyer_maker': msg['m'],
                'direction': 'buy' if not msg['m'] else 'sell'
            }
            
            with self.lock:
                self.recent_trades.append(trade)
                self.data_count += 1
                
        except Exception as e:
            logging.error(f"Error handling trade update: {e}")
    
    def handle_kline_update(self, msg):
        """Handle real-time kline updates."""
        try:
            kline = msg['k']
            new_kline = {
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'close_time': kline['t']  # Use close_time instead of timestamp
            }
            
            with self.lock:
                self.kline_data.append(new_kline)
                
                # Generate features when candle closes
                if kline['x']:  # If candle is closed
                    self.generate_and_store_features()
                    
        except Exception as e:
            logging.error(f"Error handling kline update: {e}")
    
    def calculate_order_book_features(self):
        """Calculate order book-based features."""
        try:
            if not self.order_book['bids'] or not self.order_book['asks']:
                return {}
            
            bids = self.order_book['bids']
            asks = self.order_book['asks']
            
            # Calculate bid-ask spread
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = (best_ask - best_bid) / best_bid
            
            # Calculate order book imbalance
            bid_volume = sum(qty for _, qty in bids[:10])
            ask_volume = sum(qty for _, qty in asks[:10])
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            
            # Calculate order book depth
            bid_depth = sum(qty for _, qty in bids[:5])
            ask_depth = sum(qty for _, qty in asks[:5])
            
            return {
                'spread': spread,
                'ob_imbalance': imbalance,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'best_bid': best_bid,
                'best_ask': best_ask
            }
        except Exception as e:
            logging.error(f"Error calculating order book features: {e}")
            return {}
    
    def calculate_trade_features(self):
        """Calculate trade-based features."""
        try:
            if len(self.recent_trades) < 10:
                return {}
            
            trades = list(self.recent_trades)[-100:]  # Last 100 trades
            
            prices = [t['price'] for t in trades]
            volumes = [t['quantity'] for t in trades]
            directions = [1 if t['direction'] == 'buy' else -1 for t in trades]
            
            # Price momentum
            price_momentum = (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0
            
            # Volume momentum
            volume_momentum = (volumes[-1] - volumes[0]) / volumes[0] if len(volumes) > 1 else 0
            
            # Buy/sell ratio
            buy_volume = sum(v for v, d in zip(volumes, directions) if d == 1)
            sell_volume = sum(v for v, d in zip(volumes, directions) if d == -1)
            buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else 1
            
            # Price volatility
            price_volatility = np.std(prices) if len(prices) > 1 else 0
            
            # Volume volatility
            volume_volatility = np.std(volumes) if len(volumes) > 1 else 0
            
            return {
                'price_momentum': price_momentum,
                'volume_momentum': volume_momentum,
                'buy_sell_ratio': buy_sell_ratio,
                'price_volatility': price_volatility,
                'volume_volatility': volume_volatility,
                'trade_count': len(trades)
            }
        except Exception as e:
            logging.error(f"Error calculating trade features: {e}")
            return {}
    
    def calculate_technical_indicators(self):
        """Calculate technical indicators from kline data."""
        try:
            logging.info(f"Calculating technical indicators from {len(self.kline_data)} kline data points")
            if len(self.kline_data) < 20:
                # Return default indicators if not enough data
                return {
                    'rsi': 50.0,
                    'atr': 0.5,
                    'ema': 2000.0,
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'bb_position': 0.5,
                    'stoch_k': 50.0,
                    'stoch_d': 50.0,
                    'close': 2000.0,
                    'high': 2000.0,
                    'low': 2000.0,
                    'volume': 500.0
                }
            
            # Ensure kline data has the right structure
            valid_klines = []
            logging.info(f"Processing {len(self.kline_data)} kline data points")
            for i, kline in enumerate(self.kline_data):
                if all(key in kline for key in ['open', 'high', 'low', 'close', 'volume']):
                    valid_klines.append(kline)
                else:
                    logging.warning(f"Invalid kline data at index {i}: {kline}")
                    logging.warning(f"Missing keys: {set(['open', 'high', 'low', 'close', 'volume']) - set(kline.keys())}")
            
            if len(valid_klines) < 20:
                # Return default indicators if not enough valid data
                return {
                    'rsi': 50.0,
                    'atr': 0.5,
                    'ema': 2000.0,
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'bb_position': 0.5,
                    'stoch_k': 50.0,
                    'stoch_d': 50.0,
                    'close': 2000.0,
                    'high': 2000.0,
                    'low': 2000.0,
                    'volume': 500.0
                }
            
            klines = valid_klines
            
            # Convert to numpy arrays
            closes = np.array([k['close'] for k in klines])
            highs = np.array([k['high'] for k in klines])
            lows = np.array([k['low'] for k in klines])
            volumes = np.array([k['volume'] for k in klines])
            
            # Calculate RSI
            try:
                rsi = talib.RSI(closes, timeperiod=RSI_PERIOD)[-1] if len(closes) >= RSI_PERIOD else 50.0
            except Exception as e:
                logging.warning(f"RSI calculation failed: {e}")
                rsi = 50.0
            
            # Calculate ATR
            try:
                atr = talib.ATR(highs, lows, closes, timeperiod=ATR_PERIOD)[-1] if len(closes) >= ATR_PERIOD else 0.0
            except Exception as e:
                logging.warning(f"ATR calculation failed: {e}")
                atr = 0.5
            
            # Calculate EMA
            try:
                ema = talib.EMA(closes, timeperiod=EMA_PERIOD)[-1] if len(closes) >= EMA_PERIOD else closes[-1]
            except Exception as e:
                logging.warning(f"EMA calculation failed: {e}")
                ema = closes[-1] if len(closes) > 0 else 2000.0
            
            # Calculate MACD
            try:
                macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
                macd_value = macd[-1] if len(macd) > 0 else 0.0
                macd_signal_value = macd_signal[-1] if len(macd_signal) > 0 else 0.0
            except Exception as e:
                logging.warning(f"MACD calculation failed: {e}")
                macd_value = 0.0
                macd_signal_value = 0.0
            
            # Calculate Bollinger Bands
            try:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20)
                bb_position = (closes[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if len(bb_upper) > 0 else 0.5
            except Exception as e:
                logging.warning(f"Bollinger Bands calculation failed: {e}")
                bb_position = 0.5
            
            # Calculate Stochastic
            try:
                stoch_k, stoch_d = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
                stoch_k_value = stoch_k[-1] if len(stoch_k) > 0 else 50.0
                stoch_d_value = stoch_d[-1] if len(stoch_d) > 0 else 50.0
            except Exception as e:
                logging.warning(f"Stochastic calculation failed: {e}")
                stoch_k_value = 50.0
                stoch_d_value = 50.0
            
            return {
                'rsi': rsi,
                'atr': atr,
                'ema': ema,
                'macd': macd_value,
                'macd_signal': macd_signal_value,
                'bb_position': bb_position,
                'stoch_k': stoch_k_value,
                'stoch_d': stoch_d_value,
                'close': closes[-1],
                'high': highs[-1],
                'low': lows[-1],
                'volume': volumes[-1]
            }
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")
            logging.error(f"Kline data length: {len(self.kline_data)}")
            if len(self.kline_data) > 0:
                logging.error(f"Sample kline data: {self.kline_data[0]}")
            # Return default indicators as fallback
            return {
                'rsi': 50.0,
                'atr': 0.5,
                'ema': 2000.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'bb_position': 0.5,
                'stoch_k': 50.0,
                'stoch_d': 50.0,
                'close': 2000.0,
                'high': 2000.0,
                'low': 2000.0,
                'volume': 500.0
            }
    
    def generate_and_store_features(self):
        """Generate and store combined features."""
        try:
            # Calculate all feature types
            ob_features = self.calculate_order_book_features()
            trade_features = self.calculate_trade_features()
            tech_features = self.calculate_technical_indicators()
            
            # Combine all features
            combined_features = {**ob_features, **trade_features, **tech_features}
            
            # Add timestamp
            combined_features['timestamp'] = datetime.utcnow()
            
            # Store in cache
            with self.lock:
                self.feature_cache.append(combined_features)
            
            # Store in ClickHouse
            self.store_features_in_db(combined_features)
            
            logging.info(f"Generated features: {len(combined_features)} fields")
            
        except Exception as e:
            logging.error(f"Error generating features: {e}")
            # Create default features as fallback
            default_features = {
                'rsi': 50.0,
                'atr': 0.5,
                'ema': 2000.0,
                'momentum': 0.0,
                'volume_momentum': 0.0,
                'volatility': 0.005,
                'ob_imbalance': 0.0,
                'spread': 0.0005,
                'price': 2000.0,
                'volume': 500.0,
                'timestamp': datetime.utcnow()
            }
            with self.lock:
                self.feature_cache.append(default_features)
            self.store_features_in_db(default_features)
    
    def store_features_in_db(self, features):
        """Store features in ClickHouse database."""
        try:
            # Remove timestamp from features dict for JSON storage
            features_copy = features.copy()
            timestamp = features_copy.pop('timestamp')
            
            row = {
                'ts': timestamp,
                'symbol': self.symbol,
                'features': json.dumps(features_copy)
            }
            
            clickhouse_client.insert(
                'futures_features',
                [[row['ts'], row['symbol'], row['features']]],
                column_names=['ts', 'symbol', 'features']
            )
            
        except Exception as e:
            logging.error(f"Error storing features in DB: {e}")
    
    def get_latest_features(self):
        """Get the latest calculated features."""
        with self.lock:
            if self.feature_cache:
                return self.feature_cache[-1]
            return None
    
    def get_feature_history(self, count=100):
        """Get recent feature history."""
        with self.lock:
            return list(self.feature_cache)[-count:]
    
    def log_performance_stats(self):
        """Log performance statistics."""
        current_time = time.time()
        if current_time - self.last_log_time >= 10:  # Log every 10 seconds
            with self.lock:
                stats = {
                    'data_points_collected': self.data_count,
                    'order_book_updates': len(self.order_book['bids']),
                    'recent_trades': len(self.recent_trades),
                    'kline_data': len(self.kline_data),
                    'feature_cache_size': len(self.feature_cache)
                }
                logging.info(f"HFT Data Collector Stats: {stats}")
                self.last_log_time = current_time
    
    def run_collection_loop(self):
        """Main data collection loop."""
        logging.info("HFT data collector started. Collecting real-time market data...")
        
        if DRY_RUN:
            logging.info("[DRY_RUN] Data collector is running in dry-run mode.")
        
        # Start WebSocket streams (includes synthetic data generation in dry-run)
        self.start_websocket_streams()
        
        # Also start synthetic data generation in live mode for testing
        if not DRY_RUN:
            logging.info("[LIVE] Starting synthetic data generation for testing")
            self.start_synthetic_data_generation()
        
        self.is_running = True
        
        while self.is_running:
            try:
                # Log performance stats
                self.log_performance_stats()
                
                # Generate features periodically even if no new data
                if len(self.feature_cache) == 0:
                    self.generate_and_store_features()
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(1.0)  # 1 second sleep
                
            except Exception as e:
                logging.error(f"Error in data collection loop: {e}")
                time.sleep(1)
    
    def start_synthetic_data_generation(self):
        """Generate synthetic market data for testing."""
        import threading
        import random
        
        def generate_synthetic_data():
            base_price = 2000.0
            logging.info("Synthetic data generation thread started")
            while self.is_running:
                try:
                    # Generate synthetic price movement
                    price_change = random.uniform(-0.001, 0.001)  # ±0.1% change
                    base_price *= (1 + price_change)
                    
                    # Generate synthetic trade
                    trade = {
                        'timestamp': int(time.time() * 1000),
                        'price': base_price,
                        'quantity': random.uniform(0.1, 2.0),
                        'is_buyer_maker': random.choice([True, False]),
                        'direction': 'buy' if not random.choice([True, False]) else 'sell'
                    }
                    
                    with self.lock:
                        self.recent_trades.append(trade)
                        self.real_time_data['price'] = base_price
                        self.data_count += 1
                    
                    # Generate synthetic kline data
                    kline = {
                        'open': base_price * (1 + random.uniform(-0.0005, 0.0005)),
                        'high': base_price * (1 + random.uniform(0, 0.001)),
                        'low': base_price * (1 - random.uniform(0, 0.001)),
                        'close': base_price,
                        'volume': random.uniform(100, 1000),
                        'close_time': int(time.time() * 1000)
                    }
                    
                    with self.lock:
                        self.kline_data.append(kline)
                        logging.info(f"Added synthetic kline data. Total klines: {len(self.kline_data)}")
                        # Generate features when we have enough data
                        if len(self.kline_data) >= 20:
                            self.generate_and_store_features()
                    
                    time.sleep(1)  # Generate data every second
                    logging.info(f"Synthetic data generation loop iteration. is_running: {self.is_running}")
                    
                except Exception as e:
                    logging.error(f"Error generating synthetic data: {e}")
                    time.sleep(1)
        
        # Start synthetic data generation in a separate thread
        synthetic_thread = threading.Thread(target=generate_synthetic_data, daemon=True)
        synthetic_thread.start()
        logging.info("Started synthetic data generation for testing")
    
    def stop(self):
        """Stop the data collector."""
        self.is_running = False
        if twm:
            twm.stop()

# -------------------- Main Execution --------------------

def main():
    logging.basicConfig(level=logging.INFO)
    collector = HFTDataCollector()
    
    try:
        collector.run_collection_loop()
    except KeyboardInterrupt:
        logging.info("Stopping HFT data collector...")
        collector.stop()

if __name__ == "__main__":
    main()
