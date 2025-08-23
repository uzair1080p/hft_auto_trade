#!/bin/bash

echo "🚀 HFT Auto Trading System - Server Deployment Script"
echo "====================================================="

# Server details
SERVER_IP="64.227.135.15"
SERVER_USER="root"
DEPLOY_PATH="/root/selenium_tool/hft_auto_trade"

echo "📋 Deployment Details:"
echo "Server: $SERVER_USER@$SERVER_IP"
echo "Path: $DEPLOY_PATH"
echo ""

# Check if we're on the local machine or server
if [ "$(hostname)" = "localhost" ] || [ "$(hostname)" = "DESKTOP" ]; then
    echo "📍 Local machine detected. Preparing files for deployment..."
    
    # Create deployment package
    echo "📦 Creating deployment package..."
    tar -czf hft_deployment.tar.gz \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='.venv' \
        --exclude='.pytest_cache' \
        --exclude='*.pyc' \
        --exclude='.DS_Store' \
        .
    
    echo "📤 Uploading files to server..."
    scp hft_deployment.tar.gz $SERVER_USER@$SERVER_IP:/tmp/
    
    echo "🔧 Running deployment commands on server..."
    ssh $SERVER_USER@$SERVER_IP << 'EOF'
        echo "📥 Extracting deployment package..."
        cd /root/selenium_tool
        rm -rf hft_auto_trade_backup
        mv hft_auto_trade hft_auto_trade_backup 2>/dev/null || true
        mkdir -p hft_auto_trade
        cd hft_auto_trade
        tar -xzf /tmp/hft_deployment.tar.gz
        
        echo "🔧 Setting up environment..."
        chmod +x *.sh
        chmod +x hft_entrypoint.sh
        
        echo "🐳 Building and starting Docker containers..."
        docker-compose -f docker-compose-hft.yml down 2>/dev/null || true
        docker-compose -f docker-compose-hft.yml up --build -d
        
        echo "✅ Deployment complete!"
        echo "📊 Check logs with: docker-compose -f docker-compose-hft.yml logs -f"
        echo "🌐 Dashboard available at: http://64.227.135.15:8502"
    EOF
    
    echo "🧹 Cleaning up local files..."
    rm hft_deployment.tar.gz
    
else
    echo "📍 Server detected. Running deployment directly..."
    
    # We're on the server
    cd $DEPLOY_PATH
    
    echo "🔧 Setting up environment..."
    chmod +x *.sh
    chmod +x hft_entrypoint.sh
    
    echo "🐳 Building and starting Docker containers..."
    docker-compose -f docker-compose-hft.yml down 2>/dev/null || true
    docker-compose -f docker-compose-hft.yml up --build -d
    
    echo "✅ Deployment complete!"
    echo "📊 Check logs with: docker-compose -f docker-compose-hft.yml logs -f"
    echo "🌐 Dashboard available at: http://64.227.135.15:8502"
fi

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. SSH to server: ssh root@64.227.135.15"
echo "2. Navigate to: cd /root/selenium_tool/hft_auto_trade"
echo "3. Check logs: docker-compose -f docker-compose-hft.yml logs -f"
echo "4. Access dashboard: http://64.227.135.15:8502"
echo ""
echo "⚠️  Important: Make sure to set up your Binance API credentials in the .env file!"
