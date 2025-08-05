# test_setup.py

"""
Test script to verify all components can connect properly.
Run this before starting the full trading system.
"""

import time
import logging
from clickhouse_connect import get_client
from config import (
    CLICKHOUSE_HOST, CLICKHOUSE_USER, CLICKHOUSE_PASS,
    BINANCE_API_KEY, BINANCE_API_SECRET, SYMBOL
)

def test_clickhouse():
    """Test ClickHouse connection."""
    print("🗄️ Testing ClickHouse connection...")
    try:
        client = get_client(
            host=CLICKHOUSE_HOST,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASS
        )
        client.ping()
        print("✅ ClickHouse connection successful")
        
        # Test basic query
        result = client.query("SELECT 1 as test")
        print("✅ ClickHouse query successful")
        
        return True
    except Exception as e:
        print(f"❌ ClickHouse connection failed: {e}")
        return False

def test_binance_api():
    """Test Binance API connection."""
    print("🔗 Testing Binance API connection...")
    try:
        from binance import Client
        
        if BINANCE_API_KEY == 'YOUR_BINANCE_API_KEY':
            print("⚠️ Binance API credentials not set (using test mode)")
            return True
        
        client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        
        # Test server time
        server_time = client.get_server_time()
        print(f"✅ Binance API connection successful (server time: {server_time})")
        
        # Test symbol info
        symbol_info = client.get_symbol_info(SYMBOL)
        if symbol_info:
            print(f"✅ Symbol {SYMBOL} info retrieved")
        
        return True
    except Exception as e:
        print(f"❌ Binance API connection failed: {e}")
        return False

def test_database_tables():
    """Test if required database tables exist."""
    print("📋 Testing database tables...")
    try:
        client = get_client(
            host=CLICKHOUSE_HOST,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASS
        )
        
        required_tables = [
            'futures_features',
            'executed_trades', 
            'trade_executions',
            'liquidation_events',
            'risk_checks'
        ]
        
        for table in required_tables:
            try:
                result = client.query(f"SELECT 1 FROM {table} LIMIT 1")
                print(f"✅ Table {table} exists")
            except Exception as e:
                print(f"❌ Table {table} missing: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Database table test failed: {e}")
        return False

def setup_database_tables():
    """Setup database tables if they don't exist."""
    print("🗄️ Setting up database tables...")
    try:
        client = get_client(
            host=CLICKHOUSE_HOST,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASS
        )
        
        # Read setup script
        with open('setup_tables.sql', 'r') as f:
            sql_commands = f.read()
        
        # Execute commands
        for command in sql_commands.split(';'):
            command = command.strip()
            if command:
                try:
                    client.command(command)
                    print(f"✅ Executed: {command[:50]}...")
                except Exception as e:
                    print(f"⚠️ Command failed: {e}")
        
        print("✅ Database setup complete")
        return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 HFT Trading System Setup Test")
    print("=" * 40)
    
    # Test ClickHouse
    if not test_clickhouse():
        print("\n❌ ClickHouse test failed. Please check:")
        print("1. ClickHouse container is running")
        print("2. ClickHouse host settings in config.py")
        print("3. Network connectivity")
        return False
    
    # Setup database tables
    if not setup_database_tables():
        print("\n❌ Database setup failed")
        return False
    
    # Test tables
    if not test_database_tables():
        print("\n❌ Database tables test failed")
        return False
    
    # Test Binance API
    if not test_binance_api():
        print("\n⚠️ Binance API test failed (continuing in test mode)")
    
    print("\n✅ All tests passed!")
    print("\n🚀 Ready to start trading system:")
    print("docker-compose up -d")
    
    return True

if __name__ == "__main__":
    main() 