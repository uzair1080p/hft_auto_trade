#!/bin/bash

# Function to wait for ClickHouse to be ready
wait_for_clickhouse() {
    echo "⏳ Waiting for ClickHouse to be ready..."
    while ! curl -sf -o /dev/null http://clickhouse:8123/ping; do
        echo "ClickHouse not ready yet, waiting..."
        sleep 5
    done
    echo "✅ ClickHouse is ready!"
}

# Function to initialize database tables
init_database() {
    echo "🗄️ Initializing database tables..."
    python -c "
from clickhouse_connect import get_client
import time
import os

# Wait for ClickHouse
while True:
    try:
        client = get_client(host='clickhouse', username='default', password=os.getenv('CLICKHOUSE_PASS',''))
        client.ping()
        break
    except:
        print('Waiting for ClickHouse...')
        time.sleep(5)

# Read and execute setup script
with open('setup_tables.sql', 'r') as f:
    sql_commands = f.read()

for command in sql_commands.split(';'):
    command = command.strip()
    if command:
        try:
            client.command(command)
            print(f'Executed: {command[:50]}...')
        except Exception as e:
            print(f'Error executing: {e}')

print('✅ Database initialization complete')
"
}

# Function to start a component with error handling
start_component() {
    local name=$1
    local script=$2
    echo "🚀 Starting $name..."
    python "$script" &
    local pid=$!
    echo "$name started with PID: $pid"
    return $pid
}

# Function to monitor and restart components
monitor_components() {
    local pids=("$@")
    local names=("HFT Data Collector" "HFT Model Runner" "HFT Trading Executor")
    
    while true; do
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                echo "⚠️ ${names[$i]} process died, restarting..."
                case $i in
                    0) start_component "HFT Data Collector" "hft_data_collector.py" ;;
                    1) start_component "HFT Model Runner" "hft_model_runner.py" ;;
                    2) start_component "HFT Trading Executor" "hft_trading_executor.py" ;;
                esac
                pids[$i]=$!
            fi
        done
        sleep 10
    done
}

# Main startup sequence
echo "🚀 Starting High-Frequency Trading (HFT) System..."
echo "⚡ Conservative HFT Mode: 60 trades per hour"
echo "🔒 Dry Run Mode: ${DRY_RUN:-1}"

# Wait for dependencies
wait_for_clickhouse

# Initialize database
init_database

# Start HFT components
start_component "HFT Data Collector" "hft_data_collector.py"
collector_pid=$!

start_component "HFT Model Runner" "hft_model_runner.py"
model_runner_pid=$!

start_component "HFT Trading Executor" "hft_trading_executor.py"
executor_pid=$!

echo "✅ All HFT components started successfully!"
echo "📊 Monitoring component health..."
echo "🔍 Check logs for real-time performance metrics"

# Monitor and restart components if they fail
monitor_components $collector_pid $model_runner_pid $executor_pid
