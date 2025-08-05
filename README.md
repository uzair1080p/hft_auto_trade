# 🚀 HFT AUTO TRADE

A production-ready, real-time cryptocurrency futures trading bot for ETH/BTC on Binance, with model-driven signal generation, **real trading execution**, liquidation clustering, and interactive performance dashboards.

---

## 🧠 Key Features

- **Real-time data ingestion** from Binance WebSocket
- **Feature-rich time-series dataset** using ClickHouse
- **LSTM + LightGBM ensemble** with Optuna tuning
- **SHAP-based drift detection**
- **Liquidation clustering** from Binance `!forceOrder` stream
- **🆕 Real trading execution** with risk management
- **PnL simulation with Backtrader**
- **Streamlit dashboard** with performance metrics and trade logs
- **Dockerized infrastructure** with Kafka, ClickHouse, and UI services

---

## 🗂️ Project Structure

```bash
TRADING_AGENT/
├── collector.py             # Binance WebSocket feature streamer
├── model_runner.py          # LSTM + LightGBM prediction and logging
├── liquidation.py           # Real-time liquidation clustering
├── backtest.py              # Backtrader backtesting engine
├── ui_dashboard.py          # Streamlit performance dashboard
├── hft_trading_system.py    # Legacy strategy class and SHAP drift utility
├── entrypoint.sh            # App process bootstrapper
├── Dockerfile               # Shared image for all Python services
├── docker-compose.yml       # Orchestrates ClickHouse, Kafka, app, UI
├── requirements.txt         # Python dependencies
├── makefile                 # Common CLI tasks
├── tests/                   # Unit tests
│   └── test_strategy.py     # Strategy-level unit tests
└── data_collection.py       # (Experimental – not used in prod)
```

---

## 🆕 Real Trading Functionality

The system now includes **real trading execution** capabilities with comprehensive risk management:

### Trading Components

- **`trading_executor.py`**: Executes real trades on Binance Futures based on model signals
- **`risk_manager.py`**: Monitors positions, tracks drawdown, and enforces risk limits
- **`config.py`**: Centralized configuration for all trading parameters

### Risk Management Features

- **Position Sizing**: Configurable position size (default: 2% of account per trade)
- **Stop Loss**: Automatic stop loss orders (default: 2%)
- **Take Profit**: Automatic take profit orders (default: 4%)
- **Daily Loss Limits**: Maximum daily loss protection (default: 5%)
- **Drawdown Protection**: Maximum drawdown limits (default: 15%)
- **Exposure Limits**: Maximum position exposure (default: 50%)

### Getting Started with Real Trading

1. **Set API Credentials**:
   ```bash
   export BINANCE_API_KEY="your_api_key"
   export BINANCE_API_SECRET="your_secret_key"
   ```

2. **Configure Trading Parameters** in `config.py`:
   ```python
   POSITION_SIZE_PCT = 0.02  # 2% per trade
   STOP_LOSS_PCT = 0.02      # 2% stop loss
   TAKE_PROFIT_PCT = 0.04    # 4% take profit
   ```

3. **Start the Trading System**:
   ```bash
   python start_trading_system.py
   ```

4. **Monitor Performance** via the Streamlit dashboard:
   ```bash
   streamlit run ui_dashboard.py
   ```

### Safety Features

- **Paper Trading Mode**: Test with simulated trades first
- **Risk Validation**: All trades are validated against risk limits
- **Automatic Monitoring**: Continuous position and PnL tracking
- **Emergency Stop**: Automatic shutdown on risk limit breaches