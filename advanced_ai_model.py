#!/usr/bin/env python3

"""
Advanced AI Trading Model
LSTM + Transformer ensemble architecture for sophisticated trading decisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
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
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class AdvancedTradingModel(nn.Module):
    def __init__(self, input_dim=15, sequence_length=50, hidden_size=128):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_dim, hidden_size, 
            num_layers=2, 
            dropout=0.1, 
            batch_first=True,
            bidirectional=True
        )
        
        # Transformer for attention mechanisms
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size * 2, 8)
            for _ in range(4)
        ])
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output layers
        self.signal_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 3)  # Buy, Hold, Sell
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
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
