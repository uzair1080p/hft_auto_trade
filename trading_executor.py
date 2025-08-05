# trading_executor.py

"""
Real-time trading execution module for Binance Futures.
Executes trades based on signals from model_runner.py with proper risk management.
"""

import time
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from binance import Client
from binance.exceptions import BinanceAPIException
from clickhouse_connect import get_client
from decimal import Decimal, ROUND_DOWN

# -------------------- Config --------------------

from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, SYMBOL, LEVERAGE,
    POSITION_SIZE_PCT, MAX_POSITION_SIZE_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    CLICKHOUSE_HOST, CLICKHOUSE_USER, CLICKHOUSE_PASS, SIGNAL_CHECK_INTERVAL
)
from risk_manager import RiskManager

# -------------------- Clients --------------------

binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
clickhouse_client = get_client(
    host=CLICKHOUSE_HOST,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASS,
    compress=True
)

# -------------------- Trading Executor --------------------

class TradingExecutor:
    def __init__(self, symbol=SYMBOL, leverage=LEVERAGE):
        self.symbol = symbol
        self.leverage = leverage
        self.current_position = None
        self.last_signal = None
        self.risk_manager = RiskManager()
        self.setup_futures_account()
        
    def setup_futures_account(self):
        """Initialize futures account settings."""
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
    
    def get_account_balance(self):
        """Get USDT balance from futures account."""
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
    
    def get_market_price(self):
        """Get current market price."""
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
    
    def get_quantity_precision(self):
        """Get quantity precision for the symbol."""
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
    
    def place_market_order(self, side, quantity):
        """Place a market order."""
        try:
            order = binance_client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            logging.info(f"Placed {side} order: {quantity} {self.symbol} at market")
            return order
        except Exception as e:
            logging.error(f"Failed to place {side} order: {e}")
            return None
    
    def place_stop_loss_order(self, side, quantity, price):
        """Place a stop loss order."""
        try:
            stop_side = 'SELL' if side == 'BUY' else 'BUY'
            order = binance_client.futures_create_order(
                symbol=self.symbol,
                side=stop_side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=price
            )
            logging.info(f"Placed stop loss order: {stop_side} {quantity} at {price}")
            return order
        except Exception as e:
            logging.error(f"Failed to place stop loss order: {e}")
            return None
    
    def place_take_profit_order(self, side, quantity, price):
        """Place a take profit order."""
        try:
            tp_side = 'SELL' if side == 'BUY' else 'BUY'
            order = binance_client.futures_create_order(
                symbol=self.symbol,
                side=tp_side,
                type='TAKE_PROFIT_MARKET',
                quantity=quantity,
                stopPrice=price
            )
            logging.info(f"Placed take profit order: {tp_side} {quantity} at {price}")
            return order
        except Exception as e:
            logging.error(f"Failed to place take profit order: {e}")
            return None
    
    def close_position(self):
        """Close current position."""
        position = self.get_current_position()
        if not position:
            return None
        
        side = 'SELL' if position['side'] == 'LONG' else 'BUY'
        quantity = abs(position['size'])
        
        return self.place_market_order(side, quantity)
    
    def execute_signal(self, signal, score):
        """Execute trading signal."""
        current_price = self.get_market_price()
        if not current_price:
            return False
        
        current_position = self.get_current_position()
        
        # Calculate position size for risk check
        position_value_usdt = self.calculate_position_size(current_price) * current_price
        
        # Check risk management rules
        allowed, reason = self.risk_manager.should_allow_trade(signal, position_value_usdt)
        self.risk_manager.log_risk_check(signal, allowed, reason, position_value_usdt)
        
        if not allowed:
            logging.warning(f"Trade blocked by risk manager: {reason}")
            return False
        
        # Signal: 1 = buy, -1 = sell, 0 = hold
        if signal == 1:  # Buy signal
            if current_position and current_position['side'] == 'SHORT':
                # Close short position first
                self.close_position()
                time.sleep(1)
            
            if not current_position or current_position['side'] == 'LONG':
                # Open or add to long position
                quantity = self.calculate_position_size(current_price)
                if quantity > 0:
                    order = self.place_market_order('BUY', quantity)
                    if order:
                        # Place stop loss and take profit
                        stop_price = current_price * (1 - STOP_LOSS_PCT)
                        tp_price = current_price * (1 + TAKE_PROFIT_PCT)
                        self.place_stop_loss_order('BUY', quantity, stop_price)
                        self.place_take_profit_order('BUY', quantity, tp_price)
                        return True
        
        elif signal == -1:  # Sell signal
            if current_position and current_position['side'] == 'LONG':
                # Close long position first
                self.close_position()
                time.sleep(1)
            
            if not current_position or current_position['side'] == 'SHORT':
                # Open or add to short position
                quantity = self.calculate_position_size(current_price)
                if quantity > 0:
                    order = self.place_market_order('SELL', quantity)
                    if order:
                        # Place stop loss and take profit
                        stop_price = current_price * (1 + STOP_LOSS_PCT)
                        tp_price = current_price * (1 - TAKE_PROFIT_PCT)
                        self.place_stop_loss_order('SELL', quantity, stop_price)
                        self.place_take_profit_order('SELL', quantity, tp_price)
                        return True
        
        elif signal == 0:  # Hold signal
            # Close position if we have one
            if current_position:
                self.close_position()
                return True
        
        return False
    
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
                "INSERT INTO trade_executions (ts, symbol, signal, score, executed, price, position_size) VALUES",
                [row]
            )
            logging.info(f"Logged trade execution: signal={signal}, executed={executed}, price={price}")
        except Exception as e:
            logging.error(f"Failed to log trade execution: {e}")

# -------------------- Main Execution Loop --------------------

def main():
    logging.basicConfig(level=logging.INFO)
    executor = TradingExecutor()
    
    logging.info("Trading executor started. Waiting for signals...")
    
    while True:
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
                if executor.last_signal != (ts, signal, score):
                    current_price = executor.get_market_price()
                    if current_price:
                        executed = executor.execute_signal(signal, score)
                        executor.log_trade_execution(signal, score, executed, current_price)
                        executor.last_signal = (ts, signal, score)
            
            time.sleep(SIGNAL_CHECK_INTERVAL)  # Check every 5 seconds
            
        except Exception as e:
            logging.error(f"Error in trading loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main() 