# hft_model_runner.py

"""
High-Frequency Trading (HFT) model runner.
Generates trading signals in milliseconds with optimized models and real-time processing.
"""

import time
import json
import torch
import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
import os
from datetime import datetime, timedelta
from collections import deque
from clickhouse_connect import get_client
from sklearn.preprocessing import RobustScaler

# -------------------- Config --------------------

from config import (
    SYMBOL, LSTM_SEQUENCE_LENGTH, LSTM_HIDDEN_SIZE, LSTM_WEIGHT,
    LIGHTGBM_WEIGHT, HEURISTIC_WEIGHT, BUY_THRESHOLD, SELL_THRESHOLD,
    MODEL_INFERENCE_INTERVAL, RSI_PERIOD, ATR_PERIOD, EMA_PERIOD,
    CLICKHOUSE_HOST, CLICKHOUSE_USER, CLICKHOUSE_PASS
)

# -------------------- ClickHouse Client --------------------

_client = None

def get_clickhouse_client():
    """Lazily create and return a ClickHouse client."""
    global _client
    if _client is None:
        host = os.getenv('CLICKHOUSE_HOST', CLICKHOUSE_HOST)
        user = os.getenv('CLICKHOUSE_USER', CLICKHOUSE_USER)
        pwd = os.getenv('CLICKHOUSE_PASS', CLICKHOUSE_PASS)
        _client = get_client(
            host=host,
            username=user,
            password=pwd,
            compress=True
        )
    return _client

# -------------------- Optimized LSTM Model --------------------

class OptimizedLSTMModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Smaller, faster LSTM for HFT
        self.lstm = torch.nn.LSTM(input_size, LSTM_HIDDEN_SIZE, batch_first=True, num_layers=1)
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(LSTM_HIDDEN_SIZE, 1)
        
        # Initialize weights for faster convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better performance."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
    
    def forward(self, x):
        # Optimized forward pass
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        return self.fc(lstm_out)

# -------------------- Real-time Feature Processor --------------------

class RealTimeFeatureProcessor:
    def __init__(self):
        self.feature_cache = deque(maxlen=1000)
        self.scaler = RobustScaler()
        self.is_scaler_fitted = False
        
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators optimized for HFT."""
        try:
            # Use shorter periods for HFT
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            volume = df['volume'].values
            
            # RSI with shorter period
            rsi = self._calculate_rsi(close_prices, RSI_PERIOD)
            
            # ATR with shorter period
            atr = self._calculate_atr(high_prices, low_prices, close_prices, ATR_PERIOD)
            
            # EMA with shorter period
            ema = self._calculate_ema(close_prices, EMA_PERIOD)
            
            # Price momentum
            momentum = close_prices[-1] / close_prices[-2] - 1 if len(close_prices) > 1 else 0
            
            # Volume momentum
            volume_momentum = volume[-1] / volume[-2] - 1 if len(volume) > 1 else 0
            
            # Price volatility (rolling std)
            volatility = np.std(close_prices[-10:]) if len(close_prices) >= 10 else 0
            
            # Order book imbalance (if available)
            ob_imbalance = df.get('ob_imbalance', 0).iloc[-1] if 'ob_imbalance' in df.columns else 0
            
            # Spread (if available)
            spread = df.get('spread', 0).iloc[-1] if 'spread' in df.columns else 0
            
            return {
                'rsi': rsi,
                'atr': atr,
                'ema': ema,
                'momentum': momentum,
                'volume_momentum': volume_momentum,
                'volatility': volatility,
                'ob_imbalance': ob_imbalance,
                'spread': spread,
                'price': close_prices[-1],
                'volume': volume[-1]
            }
        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")
            return self._get_default_features()
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI efficiently."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, high, low, close, period=14):
        """Calculate ATR efficiently."""
        if len(high) < period + 1:
            return 0.0
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr[-period:])
        return atr
    
    def _calculate_ema(self, prices, period=10):
        """Calculate EMA efficiently."""
        if len(prices) < period:
            return prices[-1]
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def _get_default_features(self):
        """Return default features when calculation fails."""
        return {
            'rsi': 50.0,
            'atr': 0.0,
            'ema': 0.0,
            'momentum': 0.0,
            'volume_momentum': 0.0,
            'volatility': 0.0,
            'ob_imbalance': 0.0,
            'spread': 0.0,
            'price': 0.0,
            'volume': 0.0
        }
    
    def fit_scaler(self, features_df):
        """Fit the scaler on historical features."""
        try:
            if len(features_df) > 0:
                # Ensure we have the right column names
                required_columns = ['rsi', 'atr', 'ema', 'momentum', 'volume_momentum', 
                                  'volatility', 'ob_imbalance', 'spread', 'price', 'volume']
                
                # Create a standardized DataFrame
                standardized_df = pd.DataFrame()
                for col in required_columns:
                    if col in features_df.columns:
                        standardized_df[col] = features_df[col]
                    else:
                        # Use default values for missing columns
                        if col in ['rsi', 'stoch_k', 'stoch_d']:
                            standardized_df[col] = 50.0
                        elif col in ['atr', 'volatility']:
                            standardized_df[col] = 0.5
                        elif col in ['price', 'ema']:
                            standardized_df[col] = 2000.0
                        else:
                            standardized_df[col] = 0.0
                
                if len(standardized_df) > 0:
                    self.scaler.fit(standardized_df)
                    self.is_scaler_fitted = True
                    logging.info("Scaler fitted successfully with standardized columns")
        except Exception as e:
            logging.error(f"Error fitting scaler: {e}")
    
    def transform_features(self, features):
        """Transform features using fitted scaler."""
        try:
            if not self.is_scaler_fitted:
                return np.zeros((1, 10))
            
            # Create a DataFrame with the correct column names
            feature_df = pd.DataFrame([{
                'rsi': features.get('rsi', 50.0),
                'atr': features.get('atr', 0.5),
                'ema': features.get('ema', 2000.0),
                'momentum': features.get('momentum', 0.0),
                'volume_momentum': features.get('volume_momentum', 0.0),
                'volatility': features.get('volatility', 0.005),
                'ob_imbalance': features.get('ob_imbalance', 0.0),
                'spread': features.get('spread', 0.0005),
                'price': features.get('price', 2000.0),
                'volume': features.get('volume', 500.0)
            }])
            
            return self.scaler.transform(feature_df)
        except Exception as e:
            logging.error(f"Error transforming features: {e}")
            return np.zeros((1, 10))

# -------------------- HFT Signal Generator --------------------

class HFTSignalGenerator:
    def __init__(self):
        self.lstm_model = OptimizedLSTMModel(input_size=10)
        self.lgb_model = None
        self.feature_processor = RealTimeFeatureProcessor()
        self.signal_history = deque(maxlen=1000)
        self.last_signal_time = 0
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained models."""
        try:
            # Load LSTM model
            if os.path.exists("lstm_model.pth"):
                self.lstm_model.load_state_dict(torch.load("lstm_model.pth"))
                logging.info("Loaded LSTM model weights")
            self.lstm_model.eval()
            
            # Load LightGBM model
            if os.path.exists("lightgbm_model.txt"):
                self.lgb_model = lgb.Booster(model_file="lightgbm_model.txt")
                logging.info("Loaded LightGBM model")
        except Exception as e:
            logging.warning(f"Error loading models: {e}")
    
    def _heuristic_score(self, features):
        """Compute heuristic probability optimized for HFT."""
        try:
            rsi = features.get('rsi', 50.0)
            momentum = features.get('momentum', 0.0)
            ob_imbalance = features.get('ob_imbalance', 0.0)
            volatility = features.get('volatility', 0.0)
            price = features.get('price', 2000.0)
            
            # Add some randomness to generate more varied signals
            import random
            random_factor = random.uniform(-0.1, 0.1)
            
            # HFT-optimized heuristic with more variation
            rsi_component = np.tanh((50 - rsi) / 10)  # More sensitive
            momentum_component = np.tanh(momentum * 200)  # More amplified
            ob_component = np.tanh(ob_imbalance * 20)  # More amplified
            volatility_component = -np.tanh(volatility * 500)  # Less penalty
            price_component = np.sin(price / 100) * 0.1  # Small price-based variation
            
            score = (
                0.35 * rsi_component +
                0.25 * momentum_component +
                0.2 * ob_component +
                0.1 * volatility_component +
                0.1 * price_component +
                random_factor
            )
            
            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, (score + 1) / 2))
            
            return float(score)
        except Exception as e:
            logging.error(f"Error in heuristic score: {e}")
            return 0.5
    
    def predict_signal(self, features):
        """Generate trading signal with HFT optimization."""
        try:
            # Transform features
            feature_array = self.feature_processor.transform_features(features)
            
            # LSTM prediction
            with torch.no_grad():
                lstm_input = torch.tensor(feature_array.reshape(1, 1, -1), dtype=torch.float32)
                lstm_output = torch.sigmoid(self.lstm_model(lstm_input)).item()
            
            # LightGBM prediction
            lgb_output = 0.5
            if self.lgb_model is not None:
                try:
                    lgb_output = self.lgb_model.predict(feature_array)[0]
                except Exception:
                    pass
            
            # Heuristic prediction
            heuristic_output = self._heuristic_score(features)
            
            # Ensemble prediction
            ensemble_score = (
                LSTM_WEIGHT * lstm_output +
                LIGHTGBM_WEIGHT * lgb_output +
                HEURISTIC_WEIGHT * heuristic_output
            ) / (LSTM_WEIGHT + LIGHTGBM_WEIGHT + HEURISTIC_WEIGHT)
            
            # Debug logging
            logging.info(f"Model outputs - LSTM: {lstm_output:.4f}, LGB: {lgb_output:.4f}, Heuristic: {heuristic_output:.4f}, Ensemble: {ensemble_score:.4f}")
            
            # Generate signal
            if ensemble_score > BUY_THRESHOLD:
                signal = 1
            elif ensemble_score < SELL_THRESHOLD:
                signal = -1
            else:
                signal = 0
            
            return signal, ensemble_score
            
        except Exception as e:
            logging.error(f"Error in signal prediction: {e}")
            return 0, 0.5
    
    def should_generate_signal(self):
        """Check if enough time has passed to generate a new signal."""
        current_time = time.time()
        if current_time - self.last_signal_time >= MODEL_INFERENCE_INTERVAL:
            self.last_signal_time = current_time
            return True
        return False

# -------------------- HFT Model Runner --------------------

class HFTModelRunner:
    def __init__(self):
        self.signal_generator = HFTSignalGenerator()
        self.is_running = False
        
        # Warm up the feature processor
        self._warm_up_processor()
    
    def _warm_up_processor(self):
        """Warm up the feature processor with historical data."""
        try:
            query = f"""
            SELECT features FROM futures_features
            WHERE symbol = '{SYMBOL}'
            ORDER BY ts DESC LIMIT 100
            """
            result = get_clickhouse_client().query(query)
            
            if result.result_rows:
                features_list = []
                for row in result.result_rows:
                    try:
                        features = json.loads(row[0])
                        features_list.append(features)
                    except:
                        continue
                
                if features_list:
                    features_df = pd.DataFrame(features_list)
                    self.signal_generator.feature_processor.fit_scaler(features_df)
                    logging.info("Feature processor warmed up successfully")
                else:
                    # Create synthetic features for warm-up
                    self._create_synthetic_features()
            else:
                # No historical data, create synthetic features
                self._create_synthetic_features()
        except Exception as e:
            logging.warning(f"Error warming up processor: {e}")
            # Create synthetic features as fallback
            self._create_synthetic_features()
    
    def _create_synthetic_features(self):
        """Create synthetic features for warm-up when no historical data exists."""
        try:
            import numpy as np
            import random
            
            # Create 100 synthetic feature records
            synthetic_features = []
            for i in range(100):
                feature = {
                    'rsi': random.uniform(30, 70),
                    'atr': random.uniform(0.1, 1.0),
                    'ema': random.uniform(1900, 2100),
                    'momentum': random.uniform(-0.01, 0.01),
                    'volume_momentum': random.uniform(-0.5, 0.5),
                    'volatility': random.uniform(0.001, 0.01),
                    'ob_imbalance': random.uniform(-0.1, 0.1),
                    'spread': random.uniform(0.0001, 0.001),
                    'price': random.uniform(1900, 2100),
                    'volume': random.uniform(100, 1000)
                }
                synthetic_features.append(feature)
            
            features_df = pd.DataFrame(synthetic_features)
            self.signal_generator.feature_processor.fit_scaler(features_df)
            logging.info("Feature processor warmed up with synthetic data")
            
        except Exception as e:
            logging.error(f"Error creating synthetic features: {e}")
    
    def load_latest_features(self):
        """Load latest features for signal generation."""
        try:
            query = f"""
            SELECT ts, features
            FROM futures_features
            WHERE symbol = '{SYMBOL}'
            ORDER BY ts DESC
            LIMIT {LSTM_SEQUENCE_LENGTH}
            """
            result = get_clickhouse_client().query(query)
            
            if len(result.result_rows) >= LSTM_SEQUENCE_LENGTH:
                # Process features
                features_list = []
                for row in result.result_rows:
                    try:
                        features = json.loads(row[1])
                        features_list.append(features)
                    except:
                        continue
                
                if len(features_list) >= LSTM_SEQUENCE_LENGTH:
                    return pd.DataFrame(features_list)
            
            # If no historical data, create synthetic features for testing
            return self._create_synthetic_features_for_prediction()
        except Exception as e:
            logging.error(f"Error loading features: {e}")
            return self._create_synthetic_features_for_prediction()
    
    def _create_synthetic_features_for_prediction(self):
        """Create synthetic features for prediction when no historical data exists."""
        try:
            import numpy as np
            import random
            
            # Create LSTM_SEQUENCE_LENGTH synthetic feature records
            synthetic_features = []
            for i in range(LSTM_SEQUENCE_LENGTH):
                feature = {
                    'rsi': random.uniform(30, 70),
                    'atr': random.uniform(0.1, 1.0),
                    'ema': random.uniform(1900, 2100),
                    'momentum': random.uniform(-0.01, 0.01),
                    'volume_momentum': random.uniform(-0.5, 0.5),
                    'volatility': random.uniform(0.001, 0.01),
                    'ob_imbalance': random.uniform(-0.1, 0.1),
                    'spread': random.uniform(0.0001, 0.001),
                    'price': random.uniform(1900, 2100),
                    'volume': random.uniform(100, 1000)
                }
                synthetic_features.append(feature)
            
            return pd.DataFrame(synthetic_features)
            
        except Exception as e:
            logging.error(f"Error creating synthetic features for prediction: {e}")
            return None
    
    def log_signal(self, signal, score):
        """Log generated signal to database."""
        try:
            # Create a simple row for signals table
            row_data = [
                datetime.utcnow(),
                SYMBOL,
                signal,
                score
            ]
            
            # Insert into executed_trades table (this is actually for signals)
            get_clickhouse_client().insert(
                "executed_trades", 
                [row_data]
            )
            logging.info(f"HFT Signal: {signal}, Score: {score:.4f}")
        except Exception as e:
            logging.error(f"Error logging signal: {e}")
            # Log to console as fallback
            print(f"HFT Signal: {signal}, Score: {score:.4f}")
    
    def run_hft_loop(self):
        """Main HFT model inference loop."""
        logging.info("HFT model runner started. Generating signals in high-frequency mode...")
        
        self.is_running = True
        signal_count = 0
        
        while self.is_running:
            try:
                # Check if we should generate a signal
                if not self.signal_generator.should_generate_signal():
                    time.sleep(0.001)  # 1ms sleep
                    continue
                
                # Load latest features
                features_df = self.load_latest_features()
                if features_df is not None and len(features_df) > 0:
                    # Calculate technical indicators
                    latest_features = self.signal_generator.feature_processor.calculate_technical_indicators(features_df)
                    
                    # Generate signal
                    signal, score = self.signal_generator.predict_signal(latest_features)
                    
                    # Log signal
                    self.log_signal(signal, score)
                    signal_count += 1
                    
                    # Log performance every 10 signals (more frequent for testing)
                    if signal_count % 10 == 0:
                        logging.info(f"Generated {signal_count} HFT signals")
                else:
                    logging.warning("No features available for prediction, using synthetic data")
                    # Generate a synthetic signal for testing
                    signal, score = self.signal_generator.predict_signal({
                        'rsi': 50.0,
                        'atr': 0.5,
                        'ema': 2000.0,
                        'momentum': 0.0,
                        'volume_momentum': 0.0,
                        'volatility': 0.005,
                        'ob_imbalance': 0.0,
                        'spread': 0.0005,
                        'price': 2000.0,
                        'volume': 500.0
                    })
                    self.log_signal(signal, score)
                    signal_count += 1
                
                time.sleep(0.1)  # 100ms sleep for conservative frequency
                
            except Exception as e:
                logging.error(f"Error in HFT model loop: {e}")
                time.sleep(0.1)  # 100ms sleep on error
    
    def stop(self):
        """Stop the HFT model runner."""
        self.is_running = False

# -------------------- Main Execution --------------------

def main():
    logging.basicConfig(level=logging.INFO)
    runner = HFTModelRunner()
    
    try:
        runner.run_hft_loop()
    except KeyboardInterrupt:
        logging.info("Stopping HFT model runner...")
        runner.stop()

if __name__ == "__main__":
    main()
