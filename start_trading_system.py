# start_trading_system.py

"""
Startup script for the HFT trading system.
Initializes database tables and starts all components.
"""

import time
import logging
import subprocess
import sys
from pathlib import Path
from clickhouse_connect import get_client
from config import (
    CLICKHOUSE_HOST, CLICKHOUSE_USER, CLICKHOUSE_PASS,
    validate_config, LOG_LEVEL, LOG_FORMAT
)

# -------------------- Setup --------------------

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('trading_system.log')
        ]
    )

def setup_database():
    """Setup ClickHouse database tables."""
    try:
        client = get_client(
            host=CLICKHOUSE_HOST,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASS,
            compress=True
        )
        
        # Read and execute setup_tables.sql
        with open('setup_tables.sql', 'r') as f:
            sql_commands = f.read()
        
        # Split by semicolon and execute each command
        for command in sql_commands.split(';'):
            command = command.strip()
            if command:
                client.command(command)
        
        logging.info("Database tables setup complete")
        return True
        
    except Exception as e:
        logging.error(f"Failed to setup database: {e}")
        return False

def validate_environment():
    """Validate the trading environment."""
    # Check configuration
    config_validation = validate_config()
    if not config_validation['valid']:
        logging.error("Configuration validation failed:")
        for issue in config_validation['issues']:
            logging.error(f"  - {issue}")
        return False
    
    # Check required files
    required_files = [
        'collector.py',
        'model_runner.py',
        'trading_executor.py',
        'risk_manager.py',
        'ui_dashboard.py'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logging.error(f"Required file not found: {file_path}")
            return False
    
    logging.info("Environment validation passed")
    return True

def start_component(script_name, description):
    """Start a trading system component."""
    try:
        logging.info(f"Starting {description}...")
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logging.info(f"{description} started with PID: {process.pid}")
        return process
    except Exception as e:
        logging.error(f"Failed to start {description}: {e}")
        return None

def start_trading_system():
    """Start the complete trading system."""
    setup_logging()
    
    logging.info("🚀 Starting HFT Trading System...")
    
    # Validate environment
    if not validate_environment():
        logging.error("Environment validation failed. Exiting.")
        return False
    
    # Setup database
    if not setup_database():
        logging.error("Database setup failed. Exiting.")
        return False
    
    # Start components
    processes = []
    
    # Start data collector
    collector_process = start_component('collector.py', 'Data Collector')
    if collector_process:
        processes.append(('Data Collector', collector_process))
    
    # Start model runner
    model_process = start_component('model_runner.py', 'Model Runner')
    if model_process:
        processes.append(('Model Runner', model_process))
    
    # Start trading executor
    executor_process = start_component('trading_executor.py', 'Trading Executor')
    if executor_process:
        processes.append(('Trading Executor', executor_process))
    
    # Start liquidation processor
    liquidation_process = start_component('liquidation.py', 'Liquidation Processor')
    if liquidation_process:
        processes.append(('Liquidation Processor', liquidation_process))
    
    # Start UI dashboard
    dashboard_process = start_component('ui_dashboard.py', 'UI Dashboard')
    if dashboard_process:
        processes.append(('UI Dashboard', dashboard_process))
    
    if not processes:
        logging.error("No components started successfully. Exiting.")
        return False
    
    logging.info(f"✅ Trading system started with {len(processes)} components")
    
    # Monitor processes
    try:
        while True:
            for name, process in processes:
                if process.poll() is not None:
                    logging.warning(f"{name} process terminated with code: {process.returncode}")
                    # Restart the component
                    new_process = start_component(f"{name.lower().replace(' ', '_')}.py", name)
                    if new_process:
                        processes.remove((name, process))
                        processes.append((name, new_process))
                        logging.info(f"{name} restarted")
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        logging.info("Shutting down trading system...")
        
        # Terminate all processes
        for name, process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                logging.info(f"{name} terminated")
            except subprocess.TimeoutExpired:
                process.kill()
                logging.warning(f"{name} force killed")
            except Exception as e:
                logging.error(f"Error terminating {name}: {e}")
        
        logging.info("Trading system shutdown complete")
        return True

def start_single_component(component_name):
    """Start a single component for development/testing."""
    setup_logging()
    
    component_map = {
        'collector': ('collector.py', 'Data Collector'),
        'model': ('model_runner.py', 'Model Runner'),
        'executor': ('trading_executor.py', 'Trading Executor'),
        'liquidation': ('liquidation.py', 'Liquidation Processor'),
        'dashboard': ('ui_dashboard.py', 'UI Dashboard'),
        'risk': ('risk_manager.py', 'Risk Manager')
    }
    
    if component_name not in component_map:
        logging.error(f"Unknown component: {component_name}")
        logging.info(f"Available components: {', '.join(component_map.keys())}")
        return False
    
    script_name, description = component_map[component_name]
    
    if not Path(script_name).exists():
        logging.error(f"Component script not found: {script_name}")
        return False
    
    logging.info(f"Starting {description}...")
    process = start_component(script_name, description)
    
    if process:
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            logging.info(f"{description} terminated")
    
    return True

# -------------------- Main --------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Start single component
        component = sys.argv[1]
        start_single_component(component)
    else:
        # Start full system
        start_trading_system() 