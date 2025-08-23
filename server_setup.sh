#!/bin/bash

echo "🔧 Server Setup for HFT Auto Trading System"
echo "==========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    systemctl start docker
    systemctl enable docker
    rm get-docker.sh
else
    echo "✅ Docker is already installed"
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "🐳 Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
else
    echo "✅ Docker Compose is already installed"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
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
    echo "⚠️  Please edit .env file and add your real Binance API credentials!"
else
    echo "✅ .env file already exists"
fi

# Make scripts executable
chmod +x *.sh
chmod +x hft_entrypoint.sh

echo ""
echo "✅ Server setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file: nano .env"
echo "2. Add your Binance API credentials"
echo "3. Run: ./deploy_to_server.sh"
echo "4. Check logs: docker-compose -f docker-compose-hft.yml logs -f"
