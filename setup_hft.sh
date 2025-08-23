#!/bin/bash

echo "🚀 HFT Trading System Setup"
echo "=========================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# HFT Trading System Environment Variables
# ======================================

# Binance API Credentials (REQUIRED for live trading)
# Get these from: https://www.binance.com/en/my/settings/api-management
BINANCE_API_KEY=YOUR_BINANCE_API_KEY
BINANCE_API_SECRET=YOUR_BINANCE_SECRET_KEY

# ClickHouse Database Configuration
CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_USER=default
CLICKHOUSE_PASS=secret

# Trading Configuration
DRY_RUN=1                    # Set to 0 for live trading (BE CAREFUL!)
SYMBOL=DOGEUSDT              # Trading pair
LEVERAGE=10                  # Futures leverage

# Dashboard Configuration
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=trading123

# System Configuration
KAFKA_ENABLED=0              # Disable Kafka for HFT
LIQUIDATION_ENABLED=0        # Disable liquidation processor for HFT

# Logging
LOG_LEVEL=INFO
EOF
    echo "✅ .env file created"
    echo "⚠️  Please edit .env file and add your Binance API credentials"
else
    echo "✅ .env file already exists"
fi

# Make entrypoint script executable
chmod +x hft_entrypoint.sh

echo ""
echo "🔧 Configuration Options:"
echo "1. Test the system (dry-run mode)"
echo "2. Run live trading (BE CAREFUL!)"
echo "3. View logs"
echo "4. Stop the system"
echo ""

read -p "Choose an option (1-4): " choice

case $choice in
    1)
        echo "🧪 Starting HFT system in TEST mode (dry-run)..."
        echo "⚠️  No real trades will be executed"
        docker-compose -f docker-compose-hft.yml up --build
        ;;
    2)
        echo "⚠️  WARNING: Starting LIVE trading mode!"
        echo "⚠️  This will execute real trades with real money!"
        read -p "Are you sure? Type 'YES' to continue: " confirm
        if [ "$confirm" = "YES" ]; then
            # Update .env to disable dry-run
            sed -i '' 's/DRY_RUN=1/DRY_RUN=0/' .env
            echo "🚀 Starting HFT system in LIVE mode..."
            docker-compose -f docker-compose-hft.yml up --build
        else
            echo "❌ Live trading cancelled"
        fi
        ;;
    3)
        echo "📊 Viewing logs..."
        docker-compose -f docker-compose-hft.yml logs -f
        ;;
    4)
        echo "🛑 Stopping HFT system..."
        docker-compose -f docker-compose-hft.yml down
        ;;
    *)
        echo "❌ Invalid option"
        ;;
esac
