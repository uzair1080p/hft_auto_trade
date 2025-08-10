#!/bin/bash

# HFT Trading System Deployment Script
# This script helps deploy the system on a server

echo "🚀 HFT Trading System Deployment Script"
echo "========================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please create a .env file with your Binance API credentials:"
    echo ""
    echo "BINANCE_API_KEY=your_api_key_here"
    echo "BINANCE_API_SECRET=your_api_secret_here"
    echo "DASHBOARD_USERNAME=admin"
    echo "DASHBOARD_PASSWORD=trading123"
    echo ""
    exit 1
fi

# Check if port 8502 is in use
if lsof -Pi :8502 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8502 is already in use. Stopping existing containers..."
    docker-compose down
    sleep 5
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Remove old containers and images (optional)
echo "🧹 Cleaning up old containers..."
docker-compose rm -f
docker system prune -f

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up -d --build

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service status
echo "📊 Checking service status..."
docker-compose ps

# Show logs for troubleshooting
echo "📋 Recent logs:"
docker-compose logs --tail=20

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Dashboard: http://localhost:8502"
echo "   Username: admin"
echo "   Password: trading123"
echo ""
echo "📊 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"
