#!/bin/bash

echo "🚀 Setting up Live Trading Mode for HFT System"
echo "=============================================="

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Binance API Configuration
BINANCE_API_KEY=YOUR_BINANCE_API_KEY
BINANCE_API_SECRET=YOUR_BINANCE_SECRET_KEY

# ClickHouse Configuration
CLICKHOUSE_PASSWORD=secret

# Dashboard Configuration
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=trading123

# Trading Configuration
DRY_RUN=0
KAFKA_ENABLED=0
LIQUIDATION_ENABLED=0
EOF
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

# Make entrypoint executable
chmod +x hft_entrypoint.sh

echo ""
echo "⚠️  IMPORTANT: Before running live trading, you need to:"
echo "1. Edit .env file and add your real Binance API credentials"
echo "2. Make sure you have sufficient funds in your Binance account"
echo "3. Test with small amounts first"
echo ""

read -p "Do you want to start the live trading system now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting Live Trading System..."
    docker-compose -f docker-compose-hft.yml up --build
else
    echo "To start later, run: docker-compose -f docker-compose-hft.yml up --build"
fi
