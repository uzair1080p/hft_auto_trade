# start_hft_system.py

"""
High-Frequency Trading (HFT) System Launcher.
Starts all HFT components in parallel for ultra-fast trading.
"""

import time
import logging
import os
import signal
import sys
import threading
from datetime import datetime

# Import HFT components
from hft_data_collector import HFTDataCollector
from hft_model_runner import HFTModelRunner
from hft_trading_executor import HFTTradingExecutor

# -------------------- Configuration --------------------

DRY_RUN = os.getenv('DRY_RUN', '0') == '1'

# -------------------- HFT System Coordinator --------------------

class HFTSystemCoordinator:
    def __init__(self):
        self.components = {}
        self.is_running = False
        self.start_time = None
        
        # Initialize components
        self.data_collector = HFTDataCollector()
        self.model_runner = HFTModelRunner()
        self.trading_executor = HFTTradingExecutor()
        
        # Performance tracking
        self.performance_stats = {
            'signals_generated': 0,
            'trades_executed': 0,
            'data_points_collected': 0,
            'execution_times': [],
            'errors': []
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logging.info(f"Received signal {signum}, shutting down HFT system...")
        self.stop()
        sys.exit(0)
    
    def start_component(self, name, component, start_method, stop_method):
        """Start a component in a separate thread."""
        try:
            logging.info(f"Starting {name}...")
            
            # Start the component
            if hasattr(component, start_method):
                getattr(component, start_method)()
            else:
                # Run in thread if no start method
                thread = threading.Thread(
                    target=getattr(component, start_method),
                    name=f"{name}_thread",
                    daemon=True
                )
                thread.start()
                self.components[name] = {
                    'component': component,
                    'thread': thread,
                    'stop_method': stop_method
                }
            
            logging.info(f"{name} started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start {name}: {e}")
            self.performance_stats['errors'].append(f"{name}_start_error: {e}")
            return False
    
    def start_system(self):
        """Start the entire HFT system."""
        logging.info("Starting High-Frequency Trading System...")
        self.start_time = datetime.utcnow()
        
        if DRY_RUN:
            logging.info("[DRY_RUN] HFT system is running in dry-run mode. No real trades will be executed.")
        
        # Start components in order
        components_to_start = [
            ('Data Collector', self.data_collector, 'run_collection_loop', 'stop'),
            ('Model Runner', self.model_runner, 'run_hft_loop', 'stop'),
            ('Trading Executor', self.trading_executor, 'run_hft_loop', 'stop')
        ]
        
        success_count = 0
        for name, component, start_method, stop_method in components_to_start:
            if self.start_component(name, component, start_method, stop_method):
                success_count += 1
            else:
                logging.error(f"Failed to start {name}, stopping system...")
                self.stop()
                return False
        
        if success_count == len(components_to_start):
            self.is_running = True
            logging.info("HFT system started successfully!")
            logging.info(f"System start time: {self.start_time}")
            logging.info("All components are running in high-frequency mode")
            return True
        else:
            logging.error("Failed to start all components")
            return False
    
    def stop(self):
        """Stop the entire HFT system."""
        if not self.is_running:
            return
        
        logging.info("Stopping HFT system...")
        self.is_running = False
        
        # Stop all components
        for name, component_info in self.components.items():
            try:
                logging.info(f"Stopping {name}...")
                if hasattr(component_info['component'], component_info['stop_method']):
                    getattr(component_info['component'], component_info['stop_method'])()
                logging.info(f"{name} stopped")
            except Exception as e:
                logging.error(f"Error stopping {name}: {e}")
        
        # Stop individual components
        try:
            self.data_collector.stop()
            self.model_runner.stop()
            self.trading_executor.stop()
        except Exception as e:
            logging.error(f"Error stopping components: {e}")
        
        # Log final statistics
        self.log_final_stats()
        
        logging.info("HFT system stopped")
    
    def log_final_stats(self):
        """Log final system statistics."""
        if self.start_time:
            runtime = datetime.utcnow() - self.start_time
            logging.info(f"HFT System Runtime: {runtime}")
        
        logging.info("Final Performance Statistics:")
        for key, value in self.performance_stats.items():
            if isinstance(value, list):
                logging.info(f"  {key}: {len(value)}")
            else:
                logging.info(f"  {key}: {value}")
    
    def run_monitoring_loop(self):
        """Run the main monitoring loop."""
        logging.info("Starting HFT system monitoring...")
        
        if not self.start_system():
            logging.error("Failed to start HFT system")
            return
        
        try:
            while self.is_running:
                # Monitor system health
                self.check_system_health()
                
                # Log periodic stats
                self.log_periodic_stats()
                
                # Sleep for monitoring interval
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt")
        except Exception as e:
            logging.error(f"Error in monitoring loop: {e}")
        finally:
            self.stop()
    
    def check_system_health(self):
        """Check the health of all system components."""
        try:
            # Check if all threads are still alive
            for name, component_info in self.components.items():
                if 'thread' in component_info and not component_info['thread'].is_alive():
                    logging.error(f"{name} thread is not alive!")
                    self.performance_stats['errors'].append(f"{name}_thread_dead")
            
            # Check component-specific health
            if hasattr(self.data_collector, 'get_latest_features'):
                latest_features = self.data_collector.get_latest_features()
                if latest_features:
                    self.performance_stats['data_points_collected'] += 1
            
            if hasattr(self.trading_executor, 'get_performance_stats'):
                stats = self.trading_executor.get_performance_stats()
                if stats:
                    self.performance_stats['trades_executed'] = stats.get('total_trades', 0)
                    if 'avg_execution_time_ms' in stats:
                        self.performance_stats['execution_times'].append(stats['avg_execution_time_ms'])
            
        except Exception as e:
            logging.error(f"Error checking system health: {e}")
            self.performance_stats['errors'].append(f"health_check_error: {e}")
    
    def log_periodic_stats(self):
        """Log periodic system statistics."""
        try:
            current_time = datetime.utcnow()
            
            # Log every 30 seconds
            if self.start_time and (current_time - self.start_time).seconds % 30 == 0:
                runtime = current_time - self.start_time
                
                stats = {
                    'runtime': str(runtime),
                    'data_points': self.performance_stats['data_points_collected'],
                    'trades_executed': self.performance_stats['trades_executed'],
                    'errors': len(self.performance_stats['errors'])
                }
                
                if self.performance_stats['execution_times']:
                    avg_execution = sum(self.performance_stats['execution_times']) / len(self.performance_stats['execution_times'])
                    stats['avg_execution_time_ms'] = f"{avg_execution:.2f}"
                
                logging.info(f"HFT System Stats: {stats}")
                
        except Exception as e:
            logging.error(f"Error logging periodic stats: {e}")

# -------------------- Main Execution --------------------

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('hft_system.log')
        ]
    )
    
    logging.info("=" * 60)
    logging.info("HIGH-FREQUENCY TRADING SYSTEM")
    logging.info("=" * 60)
    logging.info(f"Start time: {datetime.utcnow()}")
    logging.info(f"Dry run mode: {DRY_RUN}")
    logging.info("=" * 60)
    
    # Create and run HFT system
    coordinator = HFTSystemCoordinator()
    
    try:
        coordinator.run_monitoring_loop()
    except Exception as e:
        logging.error(f"Fatal error in HFT system: {e}")
        coordinator.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
