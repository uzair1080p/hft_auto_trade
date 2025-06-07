# tests/test_strategy.py

import pytest
import pandas as pd
from model_runner import predict_trade_signal
from model_runner import LSTMModel
import lightgbm as lgb
import torch
from sklearn.preprocessing import RobustScaler

@pytest.fixture
def dummy_data():
    df = pd.DataFrame({
        'feature1': [0.1] * 20,
        'feature2': [0.2] * 20,
        'feature3': [0.3] * 20,
        'feature4': [0.4] * 20,
        'feature5': [0.5] * 20,
        'feature6': [0.6] * 20,
        'feature7': [0.7] * 20,
        'feature8': [0.8] * 20,
        'feature9': [0.9] * 20,
        'feature10': [1.0] * 20,
    })
    return df

@pytest.fixture
def dummy_model():
    model = LSTMModel(input_size=10)
    model.eval()
    return model

@pytest.fixture
def dummy_lgb():
    return lgb.Booster(model_str="<empty_model>")  # Placeholder

@pytest.fixture
def dummy_scaler():
    scaler = RobustScaler()
    scaler.fit([[i/10.0]*10 for i in range(1, 21)])
    return scaler

def test_signal_range(dummy_model, dummy_lgb, dummy_scaler, dummy_data):
    signal, score = predict_trade_signal(dummy_model, dummy_lgb, dummy_scaler, dummy_data)
    assert signal in [-1, 0, 1], "Signal must be -1, 0 or 1"
    assert 0.0 <= score <= 1.0, "Score must be between 0 and 1"
