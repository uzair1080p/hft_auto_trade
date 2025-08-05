#!/bin/bash

# Function to wait for ClickHouse to be ready
wait_for_clickhouse() {
    echo "⏳ Waiting for ClickHouse to be ready..."
    while ! wget --no-verbose --tries=1 --spider http://clickhouse:8123/ping 2>/dev/null; do
        echo "ClickHouse not ready yet, waiting..."
        sleep 5
    done
    echo "✅ ClickHouse is ready!"
}

# Function to wait for Kafka to be ready
wait_for_kafka() {
    echo "⏳ Waiting for Kafka to be ready..."
    while ! nc -z kafka 9092; do
        echo "Kafka not ready yet, waiting..."
        sleep 5
    done
    echo "✅ Kafka is ready!"
}

# Function to initialize database tables
init_database() {
    echo "🗄️ Initializing database tables..."
    python -c "
from clickhouse_connect import get_client
import time

# Wait for ClickHouse
while True:
    try:
        client = get_client(host='clickhouse', username='default', password='')
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
    local names=("Collector" "Model Runner" "Trading Executor" "Liquidation Processor")
    
    while true; do
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                echo "⚠️ ${names[$i]} process died, restarting..."
                case $i in
                    0) start_component "Collector" "collector.py" ;;
                    1) start_component "Model Runner" "model_runner.py" ;;
                    2) start_component "Trading Executor" "trading_executor.py" ;;
                    3) start_component "Liquidation Processor" "liquidation.py" ;;
                esac
                pids[$i]=$!
            fi
        done
        sleep 10
    done
}

# Main startup sequence
echo "🚀 Starting HFT Trading System..."

# Wait for dependencies
wait_for_clickhouse
wait_for_kafka

# Initialize database
init_database

# Start components
start_component "Collector" "collector.py"
collector_pid=$!

start_component "Model Runner" "model_runner.py"
model_pid=$!

start_component "Trading Executor" "trading_executor.py"
executor_pid=$!

start_component "Liquidation Processor" "liquidation.py"
liquidation_pid=$!

echo "✅ All components started successfully!"
echo "📊 Monitoring component health..."

# Monitor and restart components if they fail
monitor_components $collector_pid $model_pid $executor_pid $liquidation_pid