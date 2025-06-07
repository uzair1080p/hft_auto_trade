# 🚀 TRADING_AGENT

A production-ready, real-time cryptocurrency futures trading bot for ETH/BTC on Binance, with model-driven signal generation, liquidation clustering, and interactive performance dashboards.

---

## 🧠 Key Features

- **Real-time data ingestion** from Binance WebSocket
- **Feature-rich time-series dataset** using ClickHouse
- **LSTM + LightGBM ensemble** with Optuna tuning
- **SHAP-based drift detection**
- **Liquidation clustering** from Binance `!forceOrder` stream
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