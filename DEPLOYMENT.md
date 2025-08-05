# 🚀 HFT Trading System Deployment Guide

This guide will help you fix the Docker deployment issues and get your trading system running properly.

## 🔧 **Current Issues Fixed**

1. **Startup Order**: Containers now wait for dependencies to be healthy
2. **Database Initialization**: Tables are created automatically
3. **Health Checks**: All services have proper health monitoring
4. **Error Handling**: Better error handling and logging
5. **Component Monitoring**: Automatic restart of failed components

## 📋 **Prerequisites**

1. **Docker & Docker Compose** installed
2. **Binance API credentials** (optional for testing)
3. **Server with at least 2GB RAM**

## 🚀 **Deployment Steps**

### **Step 1: Set Environment Variables**

Create a `.env` file in your project root:

```bash
# Binance API (optional for testing)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_key_here

# ClickHouse settings
CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_USER=default
CLICKHOUSE_PASS=

# Trading settings
SYMBOL=ETHUSDT
LEVERAGE=10
POSITION_SIZE_PCT=0.02
```

### **Step 2: Test Setup**

Before starting the full system, test the setup:

```bash
# Test database and API connections
python test_setup.py
```

### **Step 3: Start the System**

```bash
# Stop any existing containers
docker-compose down

# Remove old volumes (if needed)
docker-compose down -v

# Start the system
docker-compose up -d

# Check logs
docker-compose logs -f hft_app
```

### **Step 4: Monitor Startup**

Watch the logs to ensure everything starts properly:

```bash
# Watch all logs
docker-compose logs -f

# Watch specific service
docker logs -f hft_app
```

## 🔍 **Expected Startup Sequence**

You should see this sequence in the logs:

```
🚀 Starting HFT Trading System...
⏳ Waiting for ClickHouse to be ready...
✅ ClickHouse is ready!
⏳ Waiting for Kafka to be ready...
✅ Kafka is ready!
🗄️ Initializing database tables...
✅ Database initialization complete
🚀 Starting Collector...
🚀 Starting Model Runner...
🚀 Starting Trading Executor...
🚀 Starting Liquidation Processor...
✅ All components started successfully!
📊 Monitoring component health...
```

## 🛠️ **Troubleshooting**

### **Issue: ClickHouse Connection Refused**

**Symptoms**: `Connection refused` errors in logs

**Solution**:
```bash
# Check if ClickHouse is running
docker ps | grep clickhouse

# Check ClickHouse logs
docker logs clickhouse

# Restart ClickHouse
docker-compose restart clickhouse
```

### **Issue: No Features Arriving**

**Symptoms**: Model runner waiting indefinitely

**Solution**:
```bash
# Check collector logs
docker logs hft_app | grep -i collector

# Check Binance API credentials
docker exec hft_app python -c "from config import BINANCE_API_KEY; print('API Key set:', BINANCE_API_KEY != 'YOUR_BINANCE_API_KEY')"

# Restart collector
docker-compose restart hft_app
```

### **Issue: Dashboard Connection Error**

**Symptoms**: External connection errors in Streamlit

**Solution**:
```bash
# Use the fixed dashboard
docker-compose restart streamlit_ui

# Check dashboard logs
docker logs streamlit_ui
```

### **Issue: Components Not Starting**

**Symptoms**: Missing components in logs

**Solution**:
```bash
# Check all container status
docker-compose ps

# Restart all services
docker-compose restart

# Check resource usage
docker stats
```

## 📊 **Monitoring**

### **Check System Status**

```bash
# All containers
docker-compose ps

# Resource usage
docker stats

# Logs
docker-compose logs -f
```

### **Access Dashboard**

Open your browser to: `http://your-server-ip:8501`

### **Check Database**

```bash
# Connect to ClickHouse
docker exec -it clickhouse clickhouse-client

# Check tables
SHOW TABLES;

# Check recent data
SELECT * FROM futures_features ORDER BY ts DESC LIMIT 5;
```

## 🔄 **Updates and Maintenance**

### **Update System**

```bash
# Pull latest code
git pull

# Rebuild containers
docker-compose build

# Restart services
docker-compose up -d
```

### **Backup Data**

```bash
# Backup ClickHouse data
docker exec clickhouse clickhouse-client --query "BACKUP TABLE futures_features TO '/backup/features'"

# Backup volume
docker run --rm -v hft_auto_trade_clickhouse_data:/data -v $(pwd):/backup alpine tar czf /backup/clickhouse_backup.tar.gz /data
```

### **Clean Up**

```bash
# Remove old containers and images
docker system prune -a

# Remove unused volumes
docker volume prune
```

## 🚨 **Emergency Procedures**

### **Stop Trading**

```bash
# Stop trading executor only
docker exec hft_app pkill -f trading_executor.py

# Stop all trading
docker-compose stop hft_app
```

### **Reset System**

```bash
# Complete reset
docker-compose down -v
docker system prune -a
docker-compose up -d
```

## 📞 **Support**

If you encounter issues:

1. **Check logs**: `docker-compose logs -f`
2. **Run tests**: `python test_setup.py`
3. **Verify config**: Check `config.py` settings
4. **Check resources**: Ensure sufficient RAM/CPU

## ✅ **Success Indicators**

Your system is working correctly when you see:

- ✅ All containers running: `docker-compose ps`
- ✅ Data flowing: `docker logs hft_app | grep "Inserted features"`
- ✅ Dashboard accessible: `http://your-server:8501`
- ✅ No connection errors in logs
- ✅ Regular trading signals generated

---

**Happy Trading! 🚀** 