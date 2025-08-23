# config.py

"""
Configuration file for the HFT trading system.
Centralizes all trading parameters, API settings, and risk management rules.
"""

import os
from typing import Dict, Any

# -------------------- API Configuration --------------------

# Binance API credentials
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'YOUR_BINANCE_SECRET_KEY')

# ClickHouse configuration
CLICKHOUSE_HOST = os.getenv('CLICKHOUSE_HOST', 'clickhouse')
CLICKHOUSE_USER = os.getenv('CLICKHOUSE_USER', 'default')
CLICKHOUSE_PASS = os.getenv('CLICKHOUSE_PASS', '')

# -------------------- Trading Configuration --------------------

# Trading pair
SYMBOL = 'DOGEUSDT'  # Much cheaper than ETH, ~$0.07 vs $4,200+

# Time intervals - HFT optimized
INTERVAL = '1s'  # 1-second data collection interval
SIGNAL_CHECK_INTERVAL = 5.0  # 5 seconds between signal checks
MODEL_INFERENCE_INTERVAL = 10.0  # 10 seconds model inference interval

# Futures trading settings
LEVERAGE = 10
MARGIN_TYPE = 'ISOLATED'

# -------------------- Risk Management --------------------

# Position sizing - Conservative HFT
POSITION_SIZE_PCT = 0.15  # 15% of account per trade
MAX_POSITION_SIZE_PCT = 0.20  # Maximum 20% of account
MIN_POSITION_SIZE_USDT = 5.0   # Minimum 5.0 USDT to meet Binance requirements

# Stop loss and take profit - Conservative HFT
STOP_LOSS_PCT = 0.01  # 1% stop loss
TAKE_PROFIT_PCT = 0.02  # 2% take profit

# Risk limits
MAX_DAILY_LOSS_PCT = 0.05  # 5% maximum daily loss
MAX_DRAWDOWN_PCT = 0.15  # 15% maximum drawdown

# HFT specific settings
MAX_TRADES_PER_MINUTE = 5  # Maximum 5 trades per minute (300 per hour)
MIN_TRADE_INTERVAL = 12.0  # Minimum 12 seconds between trades
MAX_SLIPPAGE_PCT = 0.001  # Maximum 0.1% slippage tolerance

# -------------------- Model Configuration --------------------

# LSTM settings - HFT optimized
LSTM_SEQUENCE_LENGTH = 10  # Reduced for faster inference
LSTM_HIDDEN_SIZE = 16  # Reduced for faster inference
LSTM_LEARNING_RATE = 0.001

# Signal thresholds - Conservative HFT
BUY_THRESHOLD = 0.51  # Extremely sensitive for more trading signals
SELL_THRESHOLD = 0.49  # Extremely sensitive for more trading signals
HOLD_THRESHOLD = 0.5

# Model ensemble weights (must sum to 1.0)
LSTM_WEIGHT = 0.4
LIGHTGBM_WEIGHT = 0.3
HEURISTIC_WEIGHT = 0.3

# -------------------- Data Collection --------------------

# WebSocket settings - HFT optimized
WEBSOCKET_DEPTH = 20
MAX_TRADE_HISTORY = 100  # Reduced for HFT
MAX_KLINE_HISTORY = 100  # Reduced for HFT

# Feature computation - HFT optimized
RSI_PERIOD = 7  # Shorter period for HFT
ATR_PERIOD = 7  # Shorter period for HFT
EMA_PERIOD = 5  # Shorter period for HFT

# -------------------- Logging Configuration --------------------

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# -------------------- Database Configuration --------------------

# Table names
FEATURES_TABLE = 'futures_features'
SIGNALS_TABLE = 'executed_trades'
EXECUTIONS_TABLE = 'trade_executions'
LIQUIDATION_TABLE = 'liquidation_events'

# Data retention
DATA_RETENTION_DAYS = 7  # Reduced for HFT

# -------------------- UI Configuration --------------------

# Dashboard settings
DASHBOARD_PORT = 8501
DASHBOARD_HOST = '0.0.0.0'

# Chart settings
CHART_HEIGHT = 400
CHART_WIDTH = 800

# -------------------- HFT Performance Settings --------------------

# Enable real-time processing
ENABLE_REALTIME_PROCESSING = True
ENABLE_WEBSOCKET_STREAMS = True
ENABLE_ORDER_BOOK_ANALYSIS = True

# Threading settings
MAX_WORKER_THREADS = 4
QUEUE_SIZE = 1000

# -------------------- Validation Functions --------------------

def validate_config() -> Dict[str, Any]:
    """Validate configuration and return any issues."""
    issues = []
    
    # Check API credentials
    if BINANCE_API_KEY == 'YOUR_BINANCE_API_KEY':
        issues.append("BINANCE_API_KEY not set")
    if BINANCE_API_SECRET == 'YOUR_BINANCE_SECRET_KEY':
        issues.append("BINANCE_API_SECRET not set")
    
    # Check risk parameters
    if POSITION_SIZE_PCT <= 0 or POSITION_SIZE_PCT > 1:
        issues.append("POSITION_SIZE_PCT must be between 0 and 1")
    if STOP_LOSS_PCT <= 0 or STOP_LOSS_PCT > 1:
        issues.append("STOP_LOSS_PCT must be between 0 and 1")
    if TAKE_PROFIT_PCT <= 0 or TAKE_PROFIT_PCT > 1:
        issues.append("TAKE_PROFIT_PCT must be between 0 and 1")
    
    # Check model parameters
    if BUY_THRESHOLD <= 0 or BUY_THRESHOLD >= 1:
        issues.append("BUY_THRESHOLD must be between 0 and 1")
    if SELL_THRESHOLD <= 0 or SELL_THRESHOLD >= 1:
        issues.append("SELL_THRESHOLD must be between 0 and 1")
    
    # Check HFT parameters
    if SIGNAL_CHECK_INTERVAL < 0.01:
        issues.append("SIGNAL_CHECK_INTERVAL too low (< 10ms)")
    if MODEL_INFERENCE_INTERVAL < 0.01:
        issues.append("MODEL_INFERENCE_INTERVAL too low (< 10ms)")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues
    }

def get_trading_config() -> Dict[str, Any]:
    """Get trading configuration as a dictionary."""
    return {
        'symbol': SYMBOL,
        'leverage': LEVERAGE,
        'margin_type': MARGIN_TYPE,
        'position_size_pct': POSITION_SIZE_PCT,
        'max_position_size_pct': MAX_POSITION_SIZE_PCT,
        'min_position_size_usdt': MIN_POSITION_SIZE_USDT,
        'stop_loss_pct': STOP_LOSS_PCT,
        'take_profit_pct': TAKE_PROFIT_PCT,
        'max_daily_loss_pct': MAX_DAILY_LOSS_PCT,
        'max_drawdown_pct': MAX_DRAWDOWN_PCT,
        'buy_threshold': BUY_THRESHOLD,
        'sell_threshold': SELL_THRESHOLD,
        'hold_threshold': HOLD_THRESHOLD,
        'max_trades_per_minute': MAX_TRADES_PER_MINUTE,
        'min_trade_interval': MIN_TRADE_INTERVAL,
        'max_slippage_pct': MAX_SLIPPAGE_PCT
    }

def get_model_config() -> Dict[str, Any]:
    """Get model configuration as a dictionary."""
    return {
        'lstm_sequence_length': LSTM_SEQUENCE_LENGTH,
        'lstm_hidden_size': LSTM_HIDDEN_SIZE,
        'lstm_learning_rate': LSTM_LEARNING_RATE,
        'lstm_weight': LSTM_WEIGHT,
        'lightgbm_weight': LIGHTGBM_WEIGHT,
        'heuristic_weight': HEURISTIC_WEIGHT,
        'rsi_period': RSI_PERIOD,
        'atr_period': ATR_PERIOD,
        'ema_period': EMA_PERIOD
    } 