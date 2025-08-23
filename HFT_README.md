# High-Frequency Trading (HFT) System

## Overview

This is a **High-Frequency Trading (HFT) system** designed for ultra-fast trading on Binance Futures. Unlike the original system that waited 30+ seconds between trades, this HFT system can execute trades in **milliseconds** with real-time data processing.

## Key Features

### ⚡ Conservative HFT Execution
- **Signal generation**: Every 10 seconds (vs 60 seconds in original)
- **Trade execution**: Every 60 seconds (vs 5 seconds in original)
- **Data collection**: 1-second intervals (vs 1-minute intervals)
- **Trade frequency**: 60 trades per hour (vs 2 in original)
- **Average execution time**: < 10ms per trade

### 🚀 Real-Time Processing
- **WebSocket data streams** for order book, trades, and klines
- **Optimized models** with smaller LSTM networks for faster inference
- **Threaded architecture** for parallel processing
- **Memory-efficient** data structures with bounded queues

### 📊 Advanced Features
- **Order book analysis** for market microstructure
- **Real-time technical indicators** (RSI, ATR, EMA, MACD, Bollinger Bands)
- **Slippage tracking** and performance monitoring
- **Risk management** with HFT-optimized limits
- **Performance statistics** and health monitoring

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HFT Data      │    │   HFT Model     │    │   HFT Trading   │
│   Collector     │───▶│   Runner        │───▶│   Executor      │
│                 │    │                 │    │                 │
│ • WebSocket     │    │ • LSTM Model    │    │ • Market Orders │
│ • Order Book    │    │ • LightGBM      │    │ • Risk Mgmt     │
│ • Trade Streams │    │ • Heuristics    │    │ • Performance   │
│ • Kline Data    │    │ • Signal Gen    │    │ • Monitoring    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │    ClickHouse DB        │
                    │                         │
                    │ • Features Storage      │
                    │ • Signal Logging        │
                    │ • Trade Executions      │
                    │ • Performance Metrics   │
                    └─────────────────────────┘
```

## Configuration

### Conservative HFT Settings

```python
# Time intervals - Conservative HFT
INTERVAL = '1s'  # 1-second data collection
SIGNAL_CHECK_INTERVAL = 5.0  # 5 seconds between signal checks
MODEL_INFERENCE_INTERVAL = 10.0  # 10 seconds model inference

# Risk Management - Conservative HFT
POSITION_SIZE_PCT = 0.02  # 2% of account per trade
STOP_LOSS_PCT = 0.01  # 1% stop loss
TAKE_PROFIT_PCT = 0.02  # 2% take profit
MAX_TRADES_PER_MINUTE = 1  # 1 trade per minute (60 per hour)

# Model Configuration - Conservative HFT
LSTM_SEQUENCE_LENGTH = 10  # Reduced for faster inference
LSTM_HIDDEN_SIZE = 16  # Smaller network
BUY_THRESHOLD = 0.65  # Less sensitive
SELL_THRESHOLD = 0.35  # Less sensitive
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
export CLICKHOUSE_HOST="clickhouse"
export CLICKHOUSE_USER="default"
export CLICKHOUSE_PASS=""
```

### 3. Start ClickHouse Database

```bash
docker-compose up -d clickhouse
```

### 4. Test the System

```bash
python test_hft_system.py
```

### 5. Run in Dry-Run Mode

```bash
export DRY_RUN=1
python start_hft_system.py
```

### 6. Run Live Trading

```bash
export DRY_RUN=0
python start_hft_system.py
```

## Usage

### Quick Start

1. **Test the system first**:
   ```bash
   python test_hft_system.py
   ```

2. **Run in dry-run mode** (recommended for testing):
   ```bash
   export DRY_RUN=1
   python start_hft_system.py
   ```

3. **Monitor performance**:
   - Check logs in `hft_system.log`
   - Monitor real-time statistics in console output
   - View performance metrics every 30 seconds

4. **Run live trading** (only after thorough testing):
   ```bash
   export DRY_RUN=0
   python start_hft_system.py
   ```

### Individual Components

You can also run components individually:

```bash
# Data collection only
python hft_data_collector.py

# Model inference only
python hft_model_runner.py

# Trading execution only
python hft_trading_executor.py
```

## Performance Metrics

The HFT system tracks several key performance indicators:

### Execution Speed
- **Average execution time**: < 10ms
- **Signal generation frequency**: 6 signals/minute
- **Trade execution frequency**: 1 trade/minute (60 per hour)

### Risk Metrics
- **Maximum trades per minute**: 1 (60 per hour)
- **Position size**: 2% of account
- **Stop loss**: 1%
- **Take profit**: 2%

### Monitoring
- **Real-time performance stats** every 10 seconds
- **System health checks** every 5 seconds
- **Comprehensive logging** to `hft_system.log`

## Safety Features

### Risk Management
- **Maximum daily loss**: 5% of account
- **Maximum drawdown**: 15% of account
- **Trade frequency limits**: 1 trade/minute (60 per hour)
- **Minimum trade interval**: 60 seconds

### Dry-Run Mode
- **No real orders** are placed
- **Simulated execution** for testing
- **Full logging** and monitoring
- **Safe for development** and testing

### Error Handling
- **Graceful degradation** on component failures
- **Automatic retry** mechanisms
- **Comprehensive error logging**
- **System health monitoring**

## Comparison with Original System

| Feature | Original System | HFT System |
|---------|----------------|------------|
| Signal Interval | 60 seconds | 10 seconds |
| Trade Check | 5 seconds | 5 seconds |
| Data Collection | 1 minute | 1 second |
| Execution Time | 30+ seconds | < 10ms |
| Trades/Hour | ~2 | ~60 |
| Model Size | Large LSTM | Optimized LSTM |
| Data Processing | Batch | Real-time |
| Risk Management | Conservative | Conservative HFT |

## Troubleshooting

### Common Issues

1. **WebSocket connection errors**:
   - Check internet connection
   - Verify API credentials
   - Ensure Binance API access

2. **Database connection issues**:
   - Verify ClickHouse is running
   - Check connection parameters
   - Ensure tables are created

3. **Performance issues**:
   - Monitor CPU usage
   - Check memory consumption
   - Review log files for bottlenecks

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python start_hft_system.py
```

### Performance Tuning

Adjust configuration for your environment:
```python
# In config.py
SIGNAL_CHECK_INTERVAL = 0.05  # 50ms for faster execution
MODEL_INFERENCE_INTERVAL = 0.02  # 20ms for faster signals
MAX_TRADES_PER_MINUTE = 120  # Increase trade frequency
```

## Disclaimer

⚠️ **IMPORTANT**: This is a high-frequency trading system that can execute many trades quickly. 

- **Always test thoroughly** in dry-run mode first
- **Start with small position sizes** when going live
- **Monitor the system closely** during initial runs
- **Understand the risks** of HFT trading
- **Ensure your broker allows** high-frequency trading
- **Comply with all applicable regulations**

## Support

For issues and questions:
1. Check the logs in `hft_system.log`
2. Run the test suite: `python test_hft_system.py`
3. Review the configuration in `config.py`
4. Ensure all dependencies are installed

## License

This project is for educational purposes. Use at your own risk.
