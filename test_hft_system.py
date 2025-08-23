# test_hft_system.py

"""
Test script for the High-Frequency Trading (HFT) system.
Verifies that all components work correctly before running live.
"""

import time
import logging
import os
import sys
from datetime import datetime

# Import HFT components
from hft_data_collector import HFTDataCollector
from hft_model_runner import HFTModelRunner
from hft_trading_executor import HFTTradingExecutor

# -------------------- Test Configuration --------------------

DRY_RUN = True  # Always test in dry-run mode
os.environ['DRY_RUN'] = '1'

# -------------------- HFT System Tester --------------------

class HFTSystemTester:
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.utcnow()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def test_data_collector(self):
        """Test the HFT data collector."""
        logging.info("Testing HFT Data Collector...")
        
        try:
            collector = HFTDataCollector()
            
            # Test initialization
            assert collector.symbol is not None, "Symbol not set"
            assert collector.order_book is not None, "Order book not initialized"
            assert collector.recent_trades is not None, "Recent trades not initialized"
            
            # Test feature calculation methods
            test_features = collector.calculate_order_book_features()
            assert isinstance(test_features, dict), "Order book features should be dict"
            
            test_trade_features = collector.calculate_trade_features()
            assert isinstance(test_trade_features, dict), "Trade features should be dict"
            
            test_tech_features = collector.calculate_technical_indicators()
            assert isinstance(test_tech_features, dict), "Technical features should be dict"
            
            # Test feature generation
            collector.generate_and_store_features()
            
            self.test_results['data_collector'] = {
                'status': 'PASS',
                'message': 'Data collector initialized and methods working'
            }
            logging.info("✓ Data Collector test PASSED")
            
        except Exception as e:
            self.test_results['data_collector'] = {
                'status': 'FAIL',
                'message': f'Data collector test failed: {e}'
            }
            logging.error(f"✗ Data Collector test FAILED: {e}")
    
    def test_model_runner(self):
        """Test the HFT model runner."""
        logging.info("Testing HFT Model Runner...")
        
        try:
            runner = HFTModelRunner()
            
            # Test initialization
            assert runner.signal_generator is not None, "Signal generator not initialized"
            assert runner.signal_generator.lstm_model is not None, "LSTM model not initialized"
            assert runner.signal_generator.feature_processor is not None, "Feature processor not initialized"
            
            # Test feature processor
            processor = runner.signal_generator.feature_processor
            
            # Test technical indicators calculation
            test_df = self.create_test_dataframe()
            features = processor.calculate_technical_indicators(test_df)
            assert isinstance(features, dict), "Technical indicators should be dict"
            
            # Test signal generation
            signal, score = runner.signal_generator.predict_signal(features)
            assert signal in [-1, 0, 1], f"Signal should be -1, 0, or 1, got {signal}"
            assert 0 <= score <= 1, f"Score should be between 0 and 1, got {score}"
            
            self.test_results['model_runner'] = {
                'status': 'PASS',
                'message': 'Model runner initialized and signal generation working'
            }
            logging.info("✓ Model Runner test PASSED")
            
        except Exception as e:
            self.test_results['model_runner'] = {
                'status': 'FAIL',
                'message': f'Model runner test failed: {e}'
            }
            logging.error(f"✗ Model Runner test FAILED: {e}")
    
    def test_trading_executor(self):
        """Test the HFT trading executor."""
        logging.info("Testing HFT Trading Executor...")
        
        try:
            executor = HFTTradingExecutor()
            
            # Test initialization
            assert executor.symbol is not None, "Symbol not set"
            assert executor.risk_manager is not None, "Risk manager not initialized"
            assert executor.real_time_data is not None, "Real-time data not initialized"
            
            # Test market price retrieval
            price = executor.get_market_price()
            assert price is not None, "Market price should not be None"
            assert price > 0, f"Market price should be positive, got {price}"
            
            # Test position size calculation
            position_size = executor.calculate_position_size(price)
            assert position_size >= 0, f"Position size should be non-negative, got {position_size}"
            
            # Test trade execution limits
            can_trade, reason = executor.can_execute_trade()
            assert isinstance(can_trade, bool), "Can trade should be boolean"
            assert isinstance(reason, str), "Reason should be string"
            
            # Test signal execution (dry run)
            executed = executor.execute_signal(1, 0.7)  # Buy signal
            assert isinstance(executed, bool), "Execute signal should return boolean"
            
            self.test_results['trading_executor'] = {
                'status': 'PASS',
                'message': 'Trading executor initialized and methods working'
            }
            logging.info("✓ Trading Executor test PASSED")
            
        except Exception as e:
            self.test_results['trading_executor'] = {
                'status': 'FAIL',
                'message': f'Trading executor test failed: {e}'
            }
            logging.error(f"✗ Trading Executor test FAILED: {e}")
    
    def test_integration(self):
        """Test integration between components."""
        logging.info("Testing HFT System Integration...")
        
        try:
            # Create components
            collector = HFTDataCollector()
            runner = HFTModelRunner()
            executor = HFTTradingExecutor()
            
            # Test data flow
            # 1. Generate test features
            test_df = self.create_test_dataframe()
            features = collector.calculate_technical_indicators()
            
            # 2. Generate signal
            signal, score = runner.signal_generator.predict_signal(features)
            
            # 3. Execute signal
            executed = executor.execute_signal(signal, score)
            
            # All steps should complete without error
            assert isinstance(signal, int), "Signal should be integer"
            assert isinstance(score, float), "Score should be float"
            assert isinstance(executed, bool), "Executed should be boolean"
            
            self.test_results['integration'] = {
                'status': 'PASS',
                'message': 'Integration test completed successfully'
            }
            logging.info("✓ Integration test PASSED")
            
        except Exception as e:
            self.test_results['integration'] = {
                'status': 'FAIL',
                'message': f'Integration test failed: {e}'
            }
            logging.error(f"✗ Integration test FAILED: {e}")
    
    def create_test_dataframe(self):
        """Create a test dataframe for testing."""
        import pandas as pd
        import numpy as np
        
        # Create synthetic market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 2000.0
        returns = np.random.normal(0, 0.001, 100)  # Small random returns
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = {
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        }
        
        df = pd.DataFrame(data)
        
        # Add some technical features
        df['rsi'] = 50 + np.random.normal(0, 10, 100)
        df['atr'] = np.random.uniform(0.1, 1.0, 100)
        df['ema'] = prices
        df['ob_imbalance'] = np.random.normal(0, 0.1, 100)
        df['spread'] = np.random.uniform(0.0001, 0.001, 100)
        
        return df
    
    def run_all_tests(self):
        """Run all tests."""
        logging.info("=" * 60)
        logging.info("HFT SYSTEM TESTING")
        logging.info("=" * 60)
        logging.info(f"Test start time: {self.start_time}")
        logging.info(f"Dry run mode: {DRY_RUN}")
        logging.info("=" * 60)
        
        # Run individual component tests
        self.test_data_collector()
        self.test_model_runner()
        self.test_trading_executor()
        self.test_integration()
        
        # Print results
        self.print_test_results()
        
        # Return overall success
        all_passed = all(result['status'] == 'PASS' for result in self.test_results.values())
        return all_passed
    
    def print_test_results(self):
        """Print test results summary."""
        logging.info("=" * 60)
        logging.info("TEST RESULTS SUMMARY")
        logging.info("=" * 60)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.test_results.items():
            status = result['status']
            message = result['message']
            
            if status == 'PASS':
                logging.info(f"✓ {test_name}: PASS")
                passed += 1
            else:
                logging.error(f"✗ {test_name}: FAIL - {message}")
                failed += 1
        
        logging.info("=" * 60)
        logging.info(f"Total tests: {len(self.test_results)}")
        logging.info(f"Passed: {passed}")
        logging.info(f"Failed: {failed}")
        logging.info(f"Success rate: {passed/len(self.test_results)*100:.1f}%")
        logging.info("=" * 60)
        
        if failed == 0:
            logging.info("🎉 ALL TESTS PASSED! HFT system is ready to run.")
        else:
            logging.error("❌ Some tests failed. Please fix issues before running live.")

# -------------------- Main Execution --------------------

def main():
    tester = HFTSystemTester()
    
    try:
        success = tester.run_all_tests()
        
        if success:
            logging.info("HFT system testing completed successfully!")
            return 0
        else:
            logging.error("HFT system testing failed!")
            return 1
            
    except Exception as e:
        logging.error(f"Fatal error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
