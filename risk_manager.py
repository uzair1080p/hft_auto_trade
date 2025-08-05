# risk_manager.py

"""
Risk management module for the HFT trading system.
Provides position monitoring, drawdown tracking, and safety checks.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from binance import Client
from clickhouse_connect import get_client
from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET, SYMBOL,
    MAX_DAILY_LOSS_PCT, MAX_DRAWDOWN_PCT, MIN_POSITION_SIZE_USDT,
    CLICKHOUSE_HOST, CLICKHOUSE_USER, CLICKHOUSE_PASS
)

# -------------------- Clients --------------------

binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
clickhouse_client = get_client(
    host=CLICKHOUSE_HOST,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASS,
    compress=True
)

# -------------------- Risk Manager --------------------

class RiskManager:
    def __init__(self):
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = 0.0
        self.daily_start_balance = 0.0
        self.last_reset_date = None
        self.position_history = []
        
    def initialize_daily_tracking(self):
        """Initialize daily PnL tracking."""
        try:
            account = binance_client.futures_account()
            total_balance = float(account['totalWalletBalance'])
            
            current_date = datetime.utcnow().date()
            
            if self.last_reset_date != current_date:
                self.daily_start_balance = total_balance
                self.peak_balance = total_balance
                self.daily_pnl = 0.0
                self.last_reset_date = current_date
                logging.info(f"Initialized daily tracking with balance: {total_balance}")
                
        except Exception as e:
            logging.error(f"Failed to initialize daily tracking: {e}")
    
    def get_account_balance(self) -> float:
        """Get current account balance."""
        try:
            account = binance_client.futures_account()
            return float(account['totalWalletBalance'])
        except Exception as e:
            logging.error(f"Failed to get account balance: {e}")
            return 0.0
    
    def update_daily_pnl(self):
        """Update daily PnL tracking."""
        try:
            current_balance = self.get_account_balance()
            self.daily_pnl = current_balance - self.daily_start_balance
            
            # Update peak balance and drawdown
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
            
            current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
                
        except Exception as e:
            logging.error(f"Failed to update daily PnL: {e}")
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been exceeded."""
        if self.daily_start_balance <= 0:
            return True
        
        daily_loss_pct = abs(self.daily_pnl) / self.daily_start_balance
        return daily_loss_pct <= MAX_DAILY_LOSS_PCT
    
    def check_drawdown_limit(self) -> bool:
        """Check if maximum drawdown limit has been exceeded."""
        return self.max_drawdown <= MAX_DRAWDOWN_PCT
    
    def check_position_size(self, position_value_usdt: float) -> bool:
        """Check if position size is within limits."""
        return position_value_usdt >= MIN_POSITION_SIZE_USDT
    
    def get_position_risk_metrics(self) -> Dict:
        """Get current position risk metrics."""
        try:
            positions = binance_client.futures_position_information(symbol=SYMBOL)
            total_exposure = 0.0
            unrealized_pnl = 0.0
            
            for position in positions:
                if position['symbol'] == SYMBOL:
                    size = float(position['positionAmt'])
                    if size != 0:
                        entry_price = float(position['entryPrice'])
                        mark_price = float(position['markPrice'])
                        unrealized_pnl += float(position['unRealizedProfit'])
                        
                        # Calculate exposure
                        if size > 0:  # Long position
                            total_exposure += size * mark_price
                        else:  # Short position
                            total_exposure += abs(size) * mark_price
            
            account_balance = self.get_account_balance()
            exposure_pct = (total_exposure / account_balance * 100) if account_balance > 0 else 0
            
            return {
                'total_exposure_usdt': total_exposure,
                'exposure_pct': exposure_pct,
                'unrealized_pnl': unrealized_pnl,
                'account_balance': account_balance
            }
            
        except Exception as e:
            logging.error(f"Failed to get position risk metrics: {e}")
            return {}
    
    def should_allow_trade(self, signal: int, position_value_usdt: float) -> Tuple[bool, str]:
        """Determine if a trade should be allowed based on risk checks."""
        self.initialize_daily_tracking()
        self.update_daily_pnl()
        
        # Check daily loss limit
        if not self.check_daily_loss_limit():
            return False, f"Daily loss limit exceeded: {self.daily_pnl:.2f} USDT"
        
        # Check drawdown limit
        if not self.check_drawdown_limit():
            return False, f"Maximum drawdown exceeded: {self.max_drawdown:.2%}"
        
        # Check position size
        if not self.check_position_size(position_value_usdt):
            return False, f"Position size too small: {position_value_usdt:.2f} USDT"
        
        # Get current risk metrics
        risk_metrics = self.get_position_risk_metrics()
        if not risk_metrics:
            return False, "Failed to get risk metrics"
        
        # Check exposure limits
        if risk_metrics['exposure_pct'] > 50:  # Max 50% exposure
            return False, f"Exposure too high: {risk_metrics['exposure_pct']:.1f}%"
        
        return True, "Trade allowed"
    
    def log_risk_check(self, signal: int, allowed: bool, reason: str, position_value: float):
        """Log risk check results."""
        try:
            risk_metrics = self.get_position_risk_metrics()
            
            row = {
                'ts': datetime.utcnow(),
                'symbol': SYMBOL,
                'signal': signal,
                'allowed': allowed,
                'reason': reason,
                'position_value': position_value,
                'daily_pnl': self.daily_pnl,
                'max_drawdown': self.max_drawdown,
                'exposure_pct': risk_metrics.get('exposure_pct', 0),
                'account_balance': risk_metrics.get('account_balance', 0)
            }
            
            clickhouse_client.insert(
                "INSERT INTO risk_checks (ts, symbol, signal, allowed, reason, position_value, daily_pnl, max_drawdown, exposure_pct, account_balance) VALUES",
                [row]
            )
            
            if not allowed:
                logging.warning(f"Trade blocked: {reason}")
            else:
                logging.info(f"Trade allowed: {reason}")
                
        except Exception as e:
            logging.error(f"Failed to log risk check: {e}")
    
    def get_risk_summary(self) -> Dict:
        """Get a summary of current risk metrics."""
        self.initialize_daily_tracking()
        self.update_daily_pnl()
        
        risk_metrics = self.get_position_risk_metrics()
        
        return {
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': (self.daily_pnl / self.daily_start_balance * 100) if self.daily_start_balance > 0 else 0,
            'max_drawdown': self.max_drawdown,
            'account_balance': risk_metrics.get('account_balance', 0),
            'total_exposure': risk_metrics.get('total_exposure_usdt', 0),
            'exposure_pct': risk_metrics.get('exposure_pct', 0),
            'unrealized_pnl': risk_metrics.get('unrealized_pnl', 0),
            'daily_loss_limit_ok': self.check_daily_loss_limit(),
            'drawdown_limit_ok': self.check_drawdown_limit()
        }

# -------------------- Risk Check Table Setup --------------------

def setup_risk_tables():
    """Setup ClickHouse tables for risk management."""
    try:
        # Create risk_checks table
        clickhouse_client.command("""
            CREATE TABLE IF NOT EXISTS risk_checks (
                ts DateTime64(3),
                symbol String,
                signal Int8,
                allowed UInt8,
                reason String,
                position_value Float64,
                daily_pnl Float64,
                max_drawdown Float64,
                exposure_pct Float64,
                account_balance Float64
            ) ENGINE = MergeTree()
            ORDER BY (symbol, ts)
            TTL ts + INTERVAL 30 DAY
        """)
        
        logging.info("Risk management tables setup complete")
        
    except Exception as e:
        logging.error(f"Failed to setup risk tables: {e}")

# -------------------- Main --------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_risk_tables()
    
    risk_manager = RiskManager()
    
    # Test risk manager
    summary = risk_manager.get_risk_summary()
    print("Risk Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}") 