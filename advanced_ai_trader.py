#!/usr/bin/env python3

"""
Advanced AI Trading System with Continuous Learning
Features:
- LSTM + Transformer ensemble architecture
- Real-time feature engineering
- Continuous model retraining with feedback loops
- Performance tracking and model improvement
- Adaptive thresholds based on market conditions
"""

import time
import json
import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from clickhouse_connect import get_client
from datetime import datetime, timedelta
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# -------------------- Configuration --------------------

SYMBOL = os.getenv('SYMBOL', 'DOGEUSDT')
CLICKHOUSE_HOST = os.getenv('CLICKHOUSE_HOST', 'clickhouse')
CLICKHOUSE_USER = os.getenv('CLICKHOUSE_USER', 'default')
CLICKHOUSE_PASS = os.getenv('CLICKHOUSE_PASS', '')

# Model parameters
SEQUENCE_LENGTH = 50
FEATURE_DIM = 15
LSTM_HIDDEN_SIZE = 128
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 4
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# Training parameters
RETRAIN_INTERVAL_HOURS = 24
MIN_SAMPLES_FOR_TRAINING = 1000
PERFORMANCE_WINDOW_DAYS = 7
MIN_ACCURACY_THRESHOLD = 0.55

# Signal thresholds (adaptive)
BASE_BUY_THRESHOLD = 0.65
BASE_SELL_THRESHOLD = 0.35
ADAPTIVE_THRESHOLD_MARGIN = 0.1

# -------------------- ClickHouse Client --------------------

def get_clickhouse_client():
    return get_client(
        host=CLICKHOUSE_HOST,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASS,
        compress=True
    )

# -------------------- Advanced Neural Network Architecture --------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class AdvancedTradingModel(nn.Module):
    def __init__(self, input_dim, sequence_length, hidden_size=128):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_dim, hidden_size, 
            num_layers=2, 
            dropout=DROPOUT_RATE, 
            batch_first=True,
            bidirectional=True
        )
        
        # Transformer for attention mechanisms
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size * 2, TRANSFORMER_HEADS)
            for _ in range(TRANSFORMER_LAYERS)
        ])
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # Output layers
        self.signal_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size // 4, 3)  # Buy, Hold, Sell
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Transformer processing
        transformer_out = lstm_out
        for transformer_block in self.transformer_blocks:
            transformer_out = transformer_block(transformer_out)
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)
        
        # Feature extraction
        features = self.feature_extractor(pooled)
        
        # Outputs
        signal_logits = self.signal_head(features)
        confidence = self.confidence_head(features)
        
        return signal_logits, confidence

# -------------------- Feature Engineering --------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced feature engineering with technical indicators."""
    features = df.copy()
    
    # Price-based features
    if 'close' in features.columns:
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        features['price_momentum'] = features['close'] / features['close'].shift(5) - 1
        features['price_acceleration'] = features['price_momentum'].diff()
        
        # Volatility features
        features['volatility'] = features['returns'].rolling(20).std()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
        
        # Moving averages
        features['sma_10'] = features['close'].rolling(10).mean()
        features['sma_20'] = features['close'].rolling(20).mean()
        features['ema_10'] = features['close'].ewm(span=10).mean()
        features['ema_20'] = features['close'].ewm(span=20).mean()
        
        # Price position relative to moving averages
        features['price_vs_sma10'] = features['close'] / features['sma_10'] - 1
        features['price_vs_sma20'] = features['close'] / features['sma_20'] - 1
        features['sma_cross'] = (features['sma_10'] > features['sma_20']).astype(int)
    
    # Volume features (if available)
    if 'volume' in features.columns:
        features['volume_ma'] = features['volume'].rolling(20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma']
        features['volume_momentum'] = features['volume'] / features['volume'].shift(5) - 1
    
    # RSI and other technical indicators
    if 'rsi' in features.columns:
        features['rsi_momentum'] = features['rsi'].diff()
        features['rsi_acceleration'] = features['rsi_momentum'].diff()
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
    
    # Order book features
    if 'ob_imbalance' in features.columns:
        features['ob_momentum'] = features['ob_imbalance'].diff()
        features['ob_ma'] = features['ob_imbalance'].rolling(10).mean()
        features['ob_deviation'] = features['ob_imbalance'] - features['ob_ma']
    
    # Spread features
    if 'spread' in features.columns:
        features['spread_ma'] = features['spread'].rolling(20).mean()
        features['spread_ratio'] = features['spread'] / features['spread_ma']
    
    # ATR features
    if 'atr' in features.columns:
        features['atr_ma'] = features['atr'].rolling(20).mean()
        features['atr_ratio'] = features['atr'] / features['atr_ma']
    
    # Time-based features
    features['hour'] = pd.to_datetime(features.index).hour
    features['day_of_week'] = pd.to_datetime(features.index).dayofweek
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        if 'returns' in features.columns:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        if 'rsi' in features.columns:
            features[f'rsi_lag_{lag}'] = features['rsi'].shift(lag)
        if 'ob_imbalance' in features.columns:
            features[f'ob_imbalance_lag_{lag}'] = features['ob_imbalance'].shift(lag)
    
    # Interaction features
    if 'rsi' in features.columns and 'ob_imbalance' in features.columns:
        features['rsi_ob_interaction'] = features['rsi'] * features['ob_imbalance']
    
    # Fill NaN values
    features = features.fillna(method='ffill').fillna(0)
    
    return features

# -------------------- Data Loading and Preprocessing --------------------

def load_training_data(days_back: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess training data with labels."""
    client = get_clickhouse_client()
    
    # Load features
    query = f"""
    SELECT ts, features, raw_data
    FROM futures_features 
    WHERE symbol = '{SYMBOL}' 
    AND ts > now() - INTERVAL {days_back} DAY
    ORDER BY ts
    """
    
    result = client.query(query)
    if not result.result_rows:
        return None, None, None
    
    # Parse features
    data = []
    for row in result.result_rows:
        ts, features_str, raw_data_str = row
        features = json.loads(features_str)
        raw_data = json.loads(raw_data_str)
        
        # Combine features and raw data
        combined = {**features, **raw_data}
        combined['ts'] = ts
        data.append(combined)
    
    df = pd.DataFrame(data)
    df.set_index('ts', inplace=True)
    
    # Engineer features
    df = engineer_features(df)
    
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > FEATURE_DIM:
        # Select most important features
        important_features = [
            'rsi', 'ob_imbalance', 'spread', 'atr', 'returns', 'volatility',
            'price_momentum', 'volume_ratio', 'rsi_momentum', 'ob_momentum',
            'price_vs_sma10', 'sma_cross', 'rsi_oversold', 'rsi_overbought',
            'hour', 'day_of_week'
        ]
        selected_features = [col for col in important_features if col in numeric_cols]
        if len(selected_features) < FEATURE_DIM:
            selected_features.extend([col for col in numeric_cols if col not in selected_features][:FEATURE_DIM - len(selected_features)])
    else:
        selected_features = numeric_cols.tolist()
    
    # Prepare sequences
    X, y, confidence = [], [], []
    
    for i in range(SEQUENCE_LENGTH, len(df) - 1):
        # Input sequence
        sequence = df[selected_features].iloc[i-SEQUENCE_LENGTH:i].values
        X.append(sequence)
        
        # Target: next period's price movement
        current_price = df['close'].iloc[i] if 'close' in df.columns else 1.0
        next_price = df['close'].iloc[i+1] if 'close' in df.columns else 1.0
        price_change = (next_price - current_price) / current_price
        
        # Create labels
        if price_change > 0.005:  # 0.5% threshold
            label = 0  # Buy
        elif price_change < -0.005:
            label = 2  # Sell
        else:
            label = 1  # Hold
        
        y.append(label)
        
        # Confidence based on volatility
        vol = df['volatility'].iloc[i] if 'volatility' in df.columns else 0.01
        confidence.append(max(0.1, min(0.9, 1.0 - vol * 10)))
    
    return np.array(X), np.array(y), np.array(confidence)

# -------------------- Model Training --------------------

def train_model(model: AdvancedTradingModel, X: np.ndarray, y: np.ndarray, confidence: np.ndarray) -> Dict:
    """Train the advanced trading model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)
    confidence_tensor = torch.FloatTensor(confidence).to(device)
    
    # Create data loader
    dataset = TensorDataset(X_tensor, y_tensor, confidence_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Loss functions
    signal_criterion = nn.CrossEntropyLoss()
    confidence_criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    model.train()
    total_loss = 0
    signal_losses = []
    confidence_losses = []
    
    for epoch in range(50):  # Max 50 epochs
        epoch_loss = 0
        for batch_X, batch_y, batch_confidence in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            signal_logits, pred_confidence = model(batch_X)
            
            # Calculate losses
            signal_loss = signal_criterion(signal_logits, batch_y)
            confidence_loss = confidence_criterion(pred_confidence.squeeze(), batch_confidence)
            
            # Combined loss
            loss = signal_loss + 0.1 * confidence_loss
            epoch_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            signal_losses.append(signal_loss.item())
            confidence_losses.append(confidence_loss.item())
        
        # Learning rate scheduling
        scheduler.step(epoch_loss)
        
        # Early stopping
        if epoch > 10 and epoch_loss > total_loss * 1.1:
            break
        
        total_loss = epoch_loss
        
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
    
    # Calculate metrics
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        
        for batch_X, batch_y, _ in dataloader:
            signal_logits, _ = model(batch_X)
            preds = torch.argmax(signal_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'final_loss': total_loss,
        'signal_loss': np.mean(signal_losses),
        'confidence_loss': np.mean(confidence_losses)
    }

# -------------------- Performance Tracking --------------------

def calculate_performance_metrics() -> Dict:
    """Calculate recent performance metrics for model improvement."""
    client = get_clickhouse_client()
    
    # Get recent predictions and actual outcomes
    query = f"""
    SELECT 
        t1.ts,
        t1.signal as predicted_signal,
        t1.score as confidence,
        CASE 
            WHEN t2.price > t1.price * 1.005 THEN 0  -- Buy was correct
            WHEN t2.price < t1.price * 0.995 THEN 2  -- Sell was correct
            ELSE 1  -- Hold was correct
        END as actual_outcome
    FROM trade_executions t1
    LEFT JOIN (
        SELECT ts, price 
        FROM futures_features 
        WHERE symbol = '{SYMBOL}'
    ) t2 ON t1.ts = t2.ts
    WHERE t1.symbol = '{SYMBOL}'
    AND t1.ts > now() - INTERVAL {PERFORMANCE_WINDOW_DAYS} DAY
    ORDER BY t1.ts
    """
    
    result = client.query(query)
    if not result.result_rows:
        return {}
    
    # Calculate metrics
    predictions = []
    actuals = []
    confidences = []
    
    for row in result.result_rows:
        ts, pred_signal, confidence, actual = row
        if actual is not None:
            predictions.append(pred_signal)
            actuals.append(actual)
            confidences.append(confidence)
    
    if not predictions:
        return {}
    
    # Convert signals to labels
    signal_to_label = {1: 0, 0: 1, -1: 2}  # Buy=0, Hold=1, Sell=2
    pred_labels = [signal_to_label.get(p, 1) for p in predictions]
    
    metrics = {
        'accuracy': accuracy_score(actuals, pred_labels),
        'precision': precision_score(actuals, pred_labels, average='weighted'),
        'recall': recall_score(actuals, pred_labels, average='weighted'),
        'f1': f1_score(actuals, pred_labels, average='weighted'),
        'avg_confidence': np.mean(confidences),
        'total_predictions': len(predictions)
    }
    
    return metrics

# -------------------- Adaptive Thresholds --------------------

def calculate_adaptive_thresholds(performance_metrics: Dict) -> Tuple[float, float]:
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

# -------------------- Main AI Trading System --------------------

class AdvancedAITrader:
    def __init__(self):
        self.model = AdvancedTradingModel(FEATURE_DIM, SEQUENCE_LENGTH)
        self.scaler = RobustScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.last_training_time = None
        self.performance_history = []
        self.buy_threshold = BASE_BUY_THRESHOLD
        self.sell_threshold = BASE_SELL_THRESHOLD
        
        # Load existing model if available
        self.load_model()
        
    def load_model(self):
        """Load trained model and scaler."""
        try:
            self.model.load_state_dict(torch.load('advanced_ai_model.pth', map_location=self.device))
            self.scaler = joblib.load('advanced_ai_scaler.pkl')
            logging.info("Loaded existing AI model")
        except:
            logging.info("No existing model found, will train new one")
    
    def save_model(self):
        """Save trained model and scaler."""
        torch.save(self.model.state_dict(), 'advanced_ai_model.pth')
        joblib.dump(self.scaler, 'advanced_ai_scaler.pkl')
        logging.info("Saved AI model")
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if self.last_training_time is None:
            return True
        
        time_since_training = datetime.now() - self.last_training_time
        return time_since_training.total_seconds() > RETRAIN_INTERVAL_HOURS * 3600
    
    def retrain_model(self):
        """Retrain the model with new data."""
        logging.info("Starting model retraining...")
        
        # Load training data
        X, y, confidence = load_training_data(days_back=30)
        if X is None or len(X) < MIN_SAMPLES_FOR_TRAINING:
            logging.warning(f"Insufficient data for training: {len(X) if X is not None else 0} samples")
            return False
        
        # Scale features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Train model
        metrics = train_model(self.model, X_scaled, y, confidence)
        
        # Update performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'samples': len(X)
        })
        
        # Keep only recent history
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
        
        # Save model
        self.save_model()
        
        # Update adaptive thresholds
        self.update_thresholds()
        
        self.last_training_time = datetime.now()
        
        logging.info(f"Model retraining completed. Accuracy: {metrics['accuracy']:.3f}")
        return True
    
    def update_thresholds(self):
        """Update adaptive thresholds based on recent performance."""
        if self.performance_history:
            recent_metrics = self.performance_history[-1]['metrics']
            self.buy_threshold, self.sell_threshold = calculate_adaptive_thresholds(recent_metrics)
            logging.info(f"Updated thresholds: Buy={self.buy_threshold:.3f}, Sell={self.sell_threshold:.3f}")
    
    def get_latest_features(self) -> Optional[np.ndarray]:
        """Get latest features for prediction."""
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
                'price_momentum', 'volume_ratio', 'rsi_momentum', 'ob_momentum',
                'price_vs_sma10', 'sma_cross', 'rsi_oversold', 'rsi_overbought',
                'hour', 'day_of_week'
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
    
    def predict(self) -> Tuple[int, float, float]:
        """Make prediction using the AI model."""
        features = self.get_latest_features()
        if features is None:
            return 0, 0.5, 0.5  # Hold signal with medium confidence
        
        self.model.eval()
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
    
    def log_prediction(self, signal: int, score: float, confidence: float):
        """Log prediction to database."""
        client = get_clickhouse_client()
        
        ts = datetime.utcnow()
        row = {
            'ts': ts,
            'symbol': SYMBOL,
            'signal': signal,
            'score': score,
            'confidence': confidence,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold
        }
        
        client.insert(
            "executed_trades",
            [[row['ts'], row['symbol'], row['signal'], row['score']]],
            column_names=['ts', 'symbol', 'signal', 'score']
        )
        
        logging.info(f"[{ts}] AI Signal: {signal}, Score: {score:.3f}, Confidence: {confidence:.3f}")
    
    def run(self):
        """Main AI trading loop."""
        logging.info("Advanced AI Trading System started")
        
        while True:
            try:
                # Check if retraining is needed
                if self.should_retrain():
                    self.retrain_model()
                
                # Make prediction
                signal, score, confidence = self.predict()
                
                # Log prediction
                self.log_prediction(signal, score, confidence)
                
                # Update performance metrics periodically
                if len(self.performance_history) == 0 or \
                   (datetime.now() - self.performance_history[-1]['timestamp']).total_seconds() > 3600:
                    performance_metrics = calculate_performance_metrics()
                    if performance_metrics:
                        logging.info(f"Performance metrics: {performance_metrics}")
                
                time.sleep(10)  # Predict every 10 seconds
                
            except Exception as e:
                logging.error(f"Error in AI trading loop: {e}")
                time.sleep(30)

# -------------------- Main Execution --------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize and run AI trader
    ai_trader = AdvancedAITrader()
    ai_trader.run()
