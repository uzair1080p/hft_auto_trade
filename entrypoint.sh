#!/bin/bash

echo "Starting Collector..."
python collector.py &

echo "Starting Model Runner..."
python model_runner.py &

echo "Starting Liquidation Stream..."
python liquidation.py &

# Keep container alive
tail -f /dev/null