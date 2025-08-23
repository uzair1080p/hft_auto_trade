#!/usr/bin/env python3

"""
AI Signal Generator
Uses trained LSTM + Transformer model for real-time trading decisions
Features:
- Real-time AI predictions
- Adaptive thresholds based on performance
- Confidence-based decision making
- Continuous model improvement feedback
"""

import time
import json
import logging
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from clickhouse_connect import get_client
from sklearn.preprocessing import RobustScaler
import joblib
from advanced_ai_model import AdvancedTradingModel
from ai_training_system import engineer_features, calculate_performance_metrics

# Configuration
SYMBOL = os.getenv('SYMBOL', 'DOGEUSDT')
SEQUENCE_LENGTH = 50
FEATURE_DIM = 15
BASE_BUY_THRESHOLD = 0.65
BASE_SELL_THRESHOLD = 0.35
ADAPTIVE_THRESHOLD_MARGIN = 0.1

def get_clickhouse_client():
    return get_client(
        host=os.getenv('CLICKHOUSE_HOST', 'clickhouse'),
        username=os.getenv('CLICKHOUSE_USER', 'default'),
        password=os.getenv('CLICKHOUSE_PASS', ''),
        compress=True
    )

def calculate_adaptive_thresholds(performance_metrics: dict) -> tuple:
    """Calculate adaptive buy/sell thresholds based on performance."""
    accuracy = performance_metrics.get('accuracy', 0.5)
    avg_confidence = performance_metrics.get('avg_confidence', 0.5)
    
    # Adjust thresholds based on model performance
    if accuracy > 0.6:
        # High accuracy: be more aggressive
        buy_threshold = BASE_BUY_THRESHOLD - ADAPTIVE_THRESHOLD_MARGIN * 0.5
        sell_threshold = BASE_SELL_THRESHOLD + ADAPTIVE_THRESHOLD_MARGIN * 0.5
    elif accuracy < 0.45:
        # Low accuracy: be more conservative
        buy_threshold = BASE_BUY_THRESHOLD + ADAPTIVE_THRESHOLD_MARGIN
        sell_threshold = BASE_SELL_THRESHOLD - ADAPTIVE_THRESHOLD_MARGIN
    else:
        # Medium accuracy: use base thresholds
        buy_threshold = BASE_BUY_THRESHOLD
        sell_threshold = BASE_SELL_THRESHOLD
    
    # Adjust based on confidence
    confidence_factor = (avg_confidence - 0.5) * 0.2
    buy_threshold -= confidence_factor
    sell_threshold += confidence_factor
    
    return max(0.5, min(0.8, buy_threshold)), max(0.2, min(0.5, sell_threshold))

class AISignalGenerator:
    def __init__(self):
        self.model = AdvancedTradingModel(FEATURE_DIM, SEQUENCE_LENGTH)
        self.scaler = RobustScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.buy_threshold = BASE_BUY_THRESHOLD
        self.sell_threshold = BASE_SELL_THRESHOLD
        self.last_performance_update = None
        self.performance_update_interval = 3600  # 1 hour
        
        # Load trained model
        self.load_model()
        
    def load_model(self):
        """Load trained model and scaler."""
        try:
            self.model.load_state_dict(torch.load('advanced_ai_model.pth', map_location=self.device))
            self.scaler = joblib.load('advanced_ai_scaler.pkl')
            self.model.eval()
            logging.info("✅ Loaded trained AI model successfully")
        except Exception as e:
            logging.error(f"❌ Failed to load AI model: {e}")
            logging.info("⚠️ Falling back to simple RSI-based signals")
            self.model = None
    
    def update_thresholds(self):
        """Update adaptive thresholds based on recent performance."""
        if self.last_performance_update and \
           (datetime.now() - self.last_performance_update).total_seconds() < self.performance_update_interval:
            return
        
        try:
            performance_metrics = calculate_performance_metrics()
            if performance_metrics:
                self.buy_threshold, self.sell_threshold = calculate_adaptive_thresholds(performance_metrics)
                self.last_performance_update = datetime.now()
                logging.info(f"🔄 Updated thresholds: Buy={self.buy_threshold:.3f}, Sell={self.sell_threshold:.3f}")
        except Exception as e:
            logging.error(f"Failed to update thresholds: {e}")
    
    def get_latest_features(self) -> np.ndarray:
        """Get latest features for AI prediction."""
        client = get_clickhouse_client()
        
        query = f"""
        SELECT features, raw_data
        FROM futures_features 
        WHERE symbol = '{SYMBOL}' 
        ORDER BY ts DESC 
        LIMIT {SEQUENCE_LENGTH}
        """
        
        result = client.query(query)
        if not result.result_rows or len(result.result_rows) < SEQUENCE_LENGTH:
            return None
        
        # Parse features
        data = []
        for row in reversed(result.result_rows):  # Reverse to get chronological order
            features_str, raw_data_str = row
            features = json.loads(features_str)
            raw_data = json.loads(raw_data_str)
            combined = {**features, **raw_data}
            data.append(combined)
        
        df = pd.DataFrame(data)
        df = engineer_features(df)
        
        # Select features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > FEATURE_DIM:
            important_features = [
                'rsi', 'ob_imbalance', 'spread', 'atr', 'returns', 'volatility',
                'price_momentum', 'rsi_momentum', 'ob_momentum', 'price_vs_sma10',
                'sma_cross', 'rsi_oversold', 'rsi_overbought', 'hour', 'day_of_week'
            ]
            selected_features = [col for col in important_features if col in numeric_cols]
            if len(selected_features) < FEATURE_DIM:
                selected_features.extend([col for col in numeric_cols if col not in selected_features][:FEATURE_DIM - len(selected_features)])
        else:
            selected_features = numeric_cols.tolist()
        
        features = df[selected_features].values
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        return features_scaled.reshape(1, SEQUENCE_LENGTH, -1)
    
    def predict_with_ai(self) -> tuple:
        """Make prediction using the AI model."""
        if self.model is None:
            return self._fallback_prediction()
        
        features = self.get_latest_features()
        if features is None:
            return self._fallback_prediction()
        
        try:
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).to(self.device)
                signal_logits, confidence = self.model(features_tensor)
                
                # Get signal probabilities
                signal_probs = torch.softmax(signal_logits, dim=1)
                signal_prob = signal_probs.squeeze().cpu().numpy()
                confidence_score = confidence.squeeze().cpu().numpy()
                
                # Convert to signal
                buy_prob = signal_prob[0]
                hold_prob = signal_prob[1]
                sell_prob = signal_prob[2]
                
                # Apply adaptive thresholds
                if buy_prob > self.buy_threshold:
                    signal = 1  # Buy
                    score = buy_prob
                elif sell_prob > self.sell_threshold:
                    signal = -1  # Sell
                    score = sell_prob
                else:
                    signal = 0  # Hold
                    score = hold_prob
                
                return signal, score, confidence_score
                
        except Exception as e:
            logging.error(f"AI prediction failed: {e}")
            return self._fallback_prediction()
    
    def _fallback_prediction(self) -> tuple:
        """Fallback to simple RSI-based prediction."""
        try:
            client = get_clickhouse_client()
            query = f"""
            SELECT features 
            FROM futures_features 
            WHERE symbol = '{SYMBOL}' 
            ORDER BY ts DESC 
            LIMIT 1
            """
            result = client.query(query)
            
            if result.result_rows:
                features = json.loads(result.result_rows[0][0])
                rsi = features.get('rsi', 50.0)
                
                # Simple RSI-based logic
                if rsi < 30:
                    return 1, 0.8, 0.6  # Strong buy
                elif rsi < 40:
                    return 1, 0.6, 0.5  # Buy
                elif rsi > 70:
                    return -1, 0.8, 0.6  # Strong sell
                elif rsi > 60:
                    return -1, 0.6, 0.5  # Sell
                else:
                    return 0, 0.5, 0.4  # Hold
            else:
                return 0, 0.5, 0.3  # Default hold
                
        except Exception as e:
            logging.error(f"Fallback prediction failed: {e}")
            return 0, 0.5, 0.3  # Default hold
    
    def log_signal(self, signal: int, score: float, confidence: float):
        """Log AI signal to database."""
        client = get_clickhouse_client()
        
        ts = datetime.utcnow()
        
        # Insert signal
        client.insert(
            "executed_trades",
            [[ts, SYMBOL, signal, score]],
            column_names=['ts', 'symbol', 'signal', 'score']
        )
        
        # Log with AI indicators
        ai_indicator = "🤖 AI" if self.model is not None else "📊 RSI"
        logging.info(f"[{ts}] {ai_indicator} Signal: {signal}, Score: {score:.3f}, Confidence: {confidence:.3f}")
    
    def run(self):
        """Main AI signal generation loop."""
        logging.info("🚀 Advanced AI Signal Generator started")
        
        if self.model is not None:
            logging.info("✅ Using trained LSTM + Transformer AI model")
        else:
            logging.info("⚠️ Using fallback RSI-based signals")
        
        last_signal = None
        
        while True:
            try:
                # Update adaptive thresholds
                self.update_thresholds()
                
                # Make AI prediction
                signal, score, confidence = self.predict_with_ai()
                
                # Only log if signal changed
                if last_signal != signal:
                    self.log_signal(signal, score, confidence)
                    last_signal = signal
                
                time.sleep(10)  # Generate signals every 10 seconds
                
            except Exception as e:
                logging.error(f"Error in AI signal loop: {e}")
                time.sleep(30)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize and run AI signal generator
    ai_generator = AISignalGenerator()
    ai_generator.run()
