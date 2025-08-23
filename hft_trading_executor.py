# hft_trading_executor.py

"""
High-Frequency Trading (HFT) executor for Binance Futures.
Executes trades in milliseconds with real-time WebSocket data processing.
"""

import time
import json
import logging
import os
import asyncio
import threading
from datetime import datetime, timedelta
from collections import deque
from decimal import Decimal, ROUND_DOWN
import numpy as np
import pandas as pd

from binance import Client, ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from clickhouse_connect import get_client

# -------------------- Config --------------------

from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, SYMBOL, LEVERAGE,
    POSITION_SIZE_PCT, MAX_POSITION_SIZE_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    CLICKHOUSE_HOST, CLICKHOUSE_USER, CLICKHOUSE_PASS, SIGNAL_CHECK_INTERVAL,
    MODEL_INFERENCE_INTERVAL, MAX_TRADES_PER_MINUTE, MIN_TRADE_INTERVAL,
    MAX_SLIPPAGE_PCT, ENABLE_REALTIME_PROCESSING, ENABLE_WEBSOCKET_STREAMS
)
from risk_manager import RiskManager

DRY_RUN = os.getenv('DRY_RUN', '0') == '1'

# -------------------- Clients --------------------

# Initialize Binance client only if not in DRY_RUN mode
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

# -------------------- HFT Trading Executor --------------------

class HFTTradingExecutor:
    def __init__(self, symbol=SYMBOL, leverage=LEVERAGE):
        self.symbol = symbol
        self.leverage = leverage
        self.current_position = None
        self.last_signal = None
        self.risk_manager = RiskManager()
        self.setup_futures_account()
        
        # HFT specific attributes
        self.last_trade_time = 0
        self.trade_count = 0
        self.trade_count_reset_time = time.time()
        self.real_time_data = {
            'price': None,
            'order_book': {'bids': [], 'asks': []},
            'recent_trades': deque(maxlen=100),
            'volume': 0
        }
        self.signal_queue = deque(maxlen=1000)
        self.is_running = False
        
        # Performance tracking
        self.execution_times = deque(maxlen=1000)
        self.slippage_tracker = deque(maxlen=1000)
        
    def setup_futures_account(self):
        """Initialize futures account settings."""
        if DRY_RUN:
            logging.info("[DRY_RUN] Skipping futures account setup")
            return
        try:
            # Set leverage
            binance_client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            logging.info(f"Set leverage to {self.leverage}x for {self.symbol}")
            
            # Set margin type to isolated
            try:
                binance_client.futures_change_margin_type(symbol=self.symbol, marginType='ISOLATED')
                logging.info(f"Set margin type to ISOLATED for {self.symbol}")
            except BinanceAPIException as e:
                if "No need to change margin type" not in str(e):
                    raise e
                    
        except Exception as e:
            logging.error(f"Failed to setup futures account: {e}")
            raise
    
    def start_websocket_streams(self):
        """Start real-time WebSocket data streams."""
        if DRY_RUN or not ENABLE_WEBSOCKET_STREAMS:
            logging.info("[DRY_RUN] Skipping WebSocket streams")
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
            
            logging.info(f"Started WebSocket streams for {self.symbol}")
            
        except Exception as e:
            logging.error(f"Failed to start WebSocket streams: {e}")
    
    def handle_order_book_update(self, msg):
        """Handle real-time order book updates."""
        try:
            self.real_time_data['order_book']['bids'] = [
                (float(price), float(qty)) for price, qty in msg['bids']
            ]
            self.real_time_data['order_book']['asks'] = [
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
                'is_buyer_maker': msg['m']
            }
            self.real_time_data['recent_trades'].append(trade)
            self.real_time_data['price'] = trade['price']
            self.real_time_data['volume'] += trade['quantity']
        except Exception as e:
            logging.error(f"Error handling trade update: {e}")
    
    def handle_kline_update(self, msg):
        """Handle real-time kline updates."""
        try:
            kline = msg['k']
            if kline['x']:  # Candle closed
                self.real_time_data['price'] = float(kline['c'])
        except Exception as e:
            logging.error(f"Error handling kline update: {e}")
    
    def get_market_price(self):
        """Get current market price with real-time data."""
        if self.real_time_data['price']:
            return self.real_time_data['price']
        
        # Fallback to REST API
        if DRY_RUN or binance_client is None:
            return 2000.0
        try:
            ticker = binance_client.futures_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except Exception as e:
            logging.error(f"Failed to get market price: {e}")
            return None
    
    def calculate_position_size(self, price):
        """Calculate position size based on risk management rules."""
        balance = self.get_account_balance()
        if balance <= 0:
            return 0.0
        
        # Calculate position size in USDT
        position_value = balance * POSITION_SIZE_PCT
        
        # Convert to quantity
        quantity = position_value / price
        
        # Round down to appropriate precision
        precision = self.get_quantity_precision()
        quantity = Decimal(str(quantity)).quantize(Decimal(f'0.{precision * "0"}'), rounding=ROUND_DOWN)
        
        return float(quantity)
    
    def get_account_balance(self):
        """Get USDT balance from futures account."""
        if DRY_RUN or binance_client is None:
            return 1000.0
        try:
            account = binance_client.futures_account()
            for balance in account['assets']:
                if balance['asset'] == 'USDT':
                    return float(balance['availableBalance'])
            return 0.0
        except Exception as e:
            logging.error(f"Failed to get account balance: {e}")
            return 0.0
    
    def get_current_position(self):
        """Get current position for the symbol."""
        if DRY_RUN or binance_client is None:
            return None
        try:
            positions = binance_client.futures_position_information(symbol=self.symbol)
            for position in positions:
                if position['symbol'] == self.symbol:
                    size = float(position['positionAmt'])
                    if size != 0:
                        return {
                            'size': size,
                            'side': 'LONG' if size > 0 else 'SHORT',
                            'entry_price': float(position['entryPrice']),
                            'unrealized_pnl': float(position['unRealizedProfit'])
                        }
            return None
        except Exception as e:
            logging.error(f"Failed to get current position: {e}")
            return None
    
    def get_quantity_precision(self):
        """Get quantity precision for the symbol."""
        if DRY_RUN:
            return 3
        try:
            exchange_info = binance_client.futures_exchange_info()
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == self.symbol:
                    for filter_info in symbol_info['filters']:
                        if filter_info['filterType'] == 'LOT_SIZE':
                            step_size = float(filter_info['stepSize'])
                            precision = len(str(step_size).split('.')[-1].rstrip('0'))
                            return precision
            return 3  # Default precision
        except Exception as e:
            logging.error(f"Failed to get quantity precision: {e}")
            return 3
    
    def can_execute_trade(self):
        """Check if we can execute a trade based on conservative HFT limits."""
        current_time = time.time()
        
        # Reset trade count every minute
        if current_time - self.trade_count_reset_time >= 60:
            self.trade_count = 0
            self.trade_count_reset_time = current_time
        
        # Check trade frequency limits (1 trade per minute = 60 per hour)
        if self.trade_count >= MAX_TRADES_PER_MINUTE:
            return False, f"Maximum trades per minute reached ({MAX_TRADES_PER_MINUTE})"
        
        # Check minimum trade interval (60 seconds between trades)
        if current_time - self.last_trade_time < MIN_TRADE_INTERVAL:
            remaining_time = MIN_TRADE_INTERVAL - (current_time - self.last_trade_time)
            return False, f"Minimum trade interval not met. Wait {remaining_time:.1f} seconds"
        
        return True, "OK"
    
    def place_market_order(self, side, quantity):
        """Place a market order with HFT optimization."""
        start_time = time.time()
        
        if DRY_RUN:
            logging.info(f"[DRY_RUN] Would place {side} MARKET {quantity} {self.symbol}")
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            return {'orderId': 'dry-run'}
        
        try:
            order = binance_client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            
            logging.info(f"HFT {side} order executed: {quantity} {self.symbol} in {execution_time*1000:.2f}ms")
            return order
            
        except Exception as e:
            logging.error(f"Failed to place {side} order: {e}")
            return None
    
    def calculate_slippage(self, expected_price, actual_price):
        """Calculate slippage percentage."""
        if expected_price == 0:
            return 0
        return abs(actual_price - expected_price) / expected_price
    
    def execute_signal(self, signal, score):
        """Execute trading signal with HFT optimization."""
        start_time = time.time()
        
        # Check if we can execute a trade
        can_trade, reason = self.can_execute_trade()
        if not can_trade:
            logging.debug(f"Trade blocked: {reason}")
            return False
        
        current_price = self.get_market_price()
        if not current_price:
            return False
        
        current_position = self.get_current_position()
        
        # Calculate position size for risk check
        position_value_usdt = self.calculate_position_size(current_price) * current_price
        
        # Check risk management rules
        allowed, reason = self.risk_manager.should_allow_trade(signal, position_value_usdt)
        if not allowed:
            logging.warning(f"Trade blocked by risk manager: {reason}")
            return False
        
        # Signal: 1 = buy, -1 = sell, 0 = hold
        if signal == 1:  # Buy signal
            if current_position and current_position['side'] == 'SHORT':
                # Close short position first
                self.close_position()
                time.sleep(0.01)  # 10ms delay
            
            if not current_position or current_position['side'] == 'LONG':
                # Open or add to long position
                quantity = self.calculate_position_size(current_price)
                if quantity > 0:
                    order = self.place_market_order('BUY', quantity)
                    if order:
                        # Calculate slippage
                        if 'avgPrice' in order:
                            slippage = self.calculate_slippage(current_price, float(order['avgPrice']))
                            self.slippage_tracker.append(slippage)
                        
                        # Update trade tracking
                        self.last_trade_time = time.time()
                        self.trade_count += 1
                        
                        # Place stop loss and take profit
                        stop_price = current_price * (1 - STOP_LOSS_PCT)
                        tp_price = current_price * (1 + TAKE_PROFIT_PCT)
                        self.place_stop_loss_order('BUY', quantity, stop_price)
                        self.place_take_profit_order('BUY', quantity, tp_price)
                        
                        total_time = time.time() - start_time
                        logging.info(f"HFT BUY executed in {total_time*1000:.2f}ms")
                        return True
        
        elif signal == -1:  # Sell signal
            if current_position and current_position['side'] == 'LONG':
                # Close long position first
                self.close_position()
                time.sleep(0.01)  # 10ms delay
            
            if not current_position or current_position['side'] == 'SHORT':
                # Open or add to short position
                quantity = self.calculate_position_size(current_price)
                if quantity > 0:
                    order = self.place_market_order('SELL', quantity)
                    if order:
                        # Calculate slippage
                        if 'avgPrice' in order:
                            slippage = self.calculate_slippage(current_price, float(order['avgPrice']))
                            self.slippage_tracker.append(slippage)
                        
                        # Update trade tracking
                        self.last_trade_time = time.time()
                        self.trade_count += 1
                        
                        # Place stop loss and take profit
                        stop_price = current_price * (1 + STOP_LOSS_PCT)
                        tp_price = current_price * (1 - TAKE_PROFIT_PCT)
                        self.place_stop_loss_order('SELL', quantity, stop_price)
                        self.place_take_profit_order('SELL', quantity, tp_price)
                        
                        total_time = time.time() - start_time
                        logging.info(f"HFT SELL executed in {total_time*1000:.2f}ms")
                        return True
        
        elif signal == 0:  # Hold signal
            # Close position if we have one
            if current_position:
                self.close_position()
                self.last_trade_time = time.time()
                self.trade_count += 1
                return True
        
        return False
    
    def close_position(self):
        """Close current position."""
        position = self.get_current_position()
        if not position:
            return None
        
        side = 'SELL' if position['side'] == 'LONG' else 'BUY'
        quantity = abs(position['size'])
        
        return self.place_market_order(side, quantity)
    
    def place_stop_loss_order(self, side, quantity, price):
        """Place a stop loss order."""
        if DRY_RUN:
            logging.info(f"[DRY_RUN] Would place STOP {('SELL' if side=='BUY' else 'BUY')} {quantity} @ {price}")
            return {'orderId': 'dry-run-sl'}
        try:
            stop_side = 'SELL' if side == 'BUY' else 'BUY'
            order = binance_client.futures_create_order(
                symbol=self.symbol,
                side=stop_side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=price
            )
            return order
        except Exception as e:
            logging.error(f"Failed to place stop loss order: {e}")
            return None
    
    def place_take_profit_order(self, side, quantity, price):
        """Place a take profit order."""
        if DRY_RUN:
            logging.info(f"[DRY_RUN] Would place TAKE_PROFIT {('SELL' if side=='BUY' else 'BUY')} {quantity} @ {price}")
            return {'orderId': 'dry-run-tp'}
        try:
            tp_side = 'SELL' if side == 'BUY' else 'BUY'
            order = binance_client.futures_create_order(
                symbol=self.symbol,
                side=tp_side,
                type='TAKE_PROFIT_MARKET',
                quantity=quantity,
                stopPrice=price
            )
            return order
        except Exception as e:
            logging.error(f"Failed to place take profit order: {e}")
            return None
    
    def log_trade_execution(self, signal, score, executed, price):
        """Log trade execution to ClickHouse."""
        try:
            row = {
                'ts': datetime.utcnow(),
                'symbol': self.symbol,
                'signal': signal,
                'score': score,
                'executed': executed,
                'price': price,
                'position_size': self.calculate_position_size(price) if executed else 0.0
            }
            clickhouse_client.insert(
                'trade_executions',
                [[row['ts'], row['symbol'], row['signal'], row['score'], 1 if row['executed'] else 0, row['price'], row['position_size']]],
                column_names=['ts','symbol','signal','score','executed','price','position_size']
            )
        except Exception as e:
            logging.error(f"Failed to log trade execution: {e}")
    
    def get_performance_stats(self):
        """Get HFT performance statistics."""
        if not self.execution_times:
            return {}
        
        avg_execution_time = np.mean(self.execution_times) * 1000  # Convert to ms
        max_execution_time = np.max(self.execution_times) * 1000
        min_execution_time = np.min(self.execution_times) * 1000
        
        avg_slippage = np.mean(self.slippage_tracker) * 100 if self.slippage_tracker else 0  # Convert to percentage
        
        return {
            'avg_execution_time_ms': avg_execution_time,
            'max_execution_time_ms': max_execution_time,
            'min_execution_time_ms': min_execution_time,
            'avg_slippage_pct': avg_slippage,
            'trades_per_minute': self.trade_count,
            'total_trades': len(self.execution_times)
        }
    
    def run_hft_loop(self):
        """Main HFT execution loop."""
        logging.info("HFT trading executor started. Running in high-frequency mode...")
        
        if DRY_RUN:
            logging.info("[DRY_RUN] Executor is running in dry-run mode. No real orders will be placed.")
        
        # Start WebSocket streams
        self.start_websocket_streams()
        
        self.is_running = True
        
        while self.is_running:
            try:
                # Check for new signals from model_runner
                query = f"""
                SELECT ts, signal, score 
                FROM executed_trades 
                WHERE symbol = '{SYMBOL}' 
                ORDER BY ts DESC 
                LIMIT 1
                """
                result = clickhouse_client.query(query)
                
                if result.result_rows:
                    ts, signal, score = result.result_rows[0]
                    
                    # Check if this is a new signal (not processed yet)
                    if self.last_signal != (ts, signal, score):
                        logging.info(f"New signal received: {signal} (score: {score:.4f})")
                        current_price = self.get_market_price()
                        if current_price:
                            executed = self.execute_signal(signal, score)
                            self.log_trade_execution(signal, score, executed, current_price)
                            self.last_signal = (ts, signal, score)
                            logging.info(f"Signal executed: {executed}")
                        else:
                            logging.warning("Could not get market price for signal execution")
                else:
                    logging.debug("No signals found in database")
                
                # Log performance stats every 10 seconds
                if int(time.time()) % 10 == 0:
                    stats = self.get_performance_stats()
                    if stats:
                        logging.info(f"HFT Performance: {stats}")
                
                time.sleep(SIGNAL_CHECK_INTERVAL)  # Check every 5 seconds
                
            except Exception as e:
                logging.error(f"Error in HFT loop: {e}")
                time.sleep(1)
    
    def stop(self):
        """Stop the HFT executor."""
        self.is_running = False
        if twm:
            twm.stop()

# -------------------- Main Execution --------------------

def main():
    logging.basicConfig(level=logging.INFO)
    executor = HFTTradingExecutor()
    
    try:
        executor.run_hft_loop()
    except KeyboardInterrupt:
        logging.info("Stopping HFT executor...")
        executor.stop()

if __name__ == "__main__":
    main()
