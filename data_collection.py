import pandas as pd
import numpy as np
import talib
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
from datetime import datetime, timedelta
import time

# Initialize Binance client
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
client = Client(api_key, api_secret)

class CryptoDataCollector:
    def __init__(self, symbols=['BTCUSDT', 'ETHUSDT'], interval='1m'):
        self.symbols = symbols
        self.interval = interval
        self.order_books = {symbol: {'bids': [], 'asks': []} for symbol in symbols}
        self.trade_data = {symbol: [] for symbol in symbols}
        self.indicators = {symbol: pd.DataFrame() for symbol in symbols}
        self.twm = ThreadedWebsocketManager(api_key, api_secret)
        
    def start_data_streams(self):
        """Start WebSocket streams for real-time data"""
        self.twm.start()
        
        # Start order book streams
        for symbol in self.symbols:
            self.twm.start_depth_socket(
                symbol=symbol, 
                callback=self.handle_order_book, 
                depth=BinanceSocketManager.WEBSOCKET_DEPTH_20
            )
            
        # Start trade streams
        for symbol in self.symbols:
            self.twm.start_trade_socket(
                symbol=symbol, 
                callback=self.handle_trade
            )
            
        # Start kline streams
        for symbol in self.symbols:
            self.twm.start_kline_socket(
                symbol=symbol, 
                interval=self.interval,
                callback=self.handle_kline
            )
    
    def handle_order_book(self, msg):
        """Process real-time order book updates"""
        symbol = msg['s']
        self.order_books[symbol]['bids'] = [(float(price), float(qty)) for price, qty in msg['bids']]
        self.order_books[symbol]['asks'] = [(float(price), float(qty)) for price, qty in msg['asks']]
        self.calculate_order_book_features(symbol)
    
    def handle_trade(self, msg):
        """Process real-time trade updates"""
        symbol = msg['s']
        trade = {
            'timestamp': msg['T'],
            'price': float(msg['p']),
            'quantity': float(msg['q']),
            'is_buyer_maker': msg['m'],
            'direction': 'buy' if not msg['m'] else 'sell'
        }
        self.trade_data[symbol].append(trade)
        self.calculate_trade_features(symbol)
    
    def handle_kline(self, msg):
        """Process kline/candlestick updates"""
        kline = msg['k']
        symbol = msg['s']
        new_row = {
            'timestamp': kline['t'],
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'closed': kline['x']
        }
        
        # Update indicators DataFrame
        if self.indicators[symbol].empty:
            self.indicators[symbol] = pd.DataFrame([new_row])
        else:
            self.indicators[symbol] = pd.concat([
                self.indicators[symbol], 
                pd.DataFrame([new_row])
            ], ignore_index=True)
        
        # Keep only recent data
        self.indicators[symbol] = self.indicators[symbol].tail(500)
        
        # Calculate technical indicators when candle closes
        if new_row['closed']:
            self.calculate_technical_indicators(symbol)
    
    def calculate_order_book_features(self, symbol):
        """Calculate order book-based features"""
        bids = self.order_books[symbol]['bids']
        asks = self.order_books[symbol]['asks']
        
        if not bids or not asks:
            return
            
        # Order Book Imbalance
        bid_vol = sum(qty for _, qty in bids)
        ask_vol = sum(qty for _, qty in asks)
        ob_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
        
        # Weighted Mid Price
        bid_value = sum(price * qty for price, qty in bids)
        ask_value = sum(price * qty for price, qty in asks)
        wmid = (bid_value + ask_value) / (bid_vol + ask_vol)
        
        # Depth Features
        depth_diff = bid_vol - ask_vol
        top_bid = bids[0][0]
        top_ask = asks[0][0]
        spread = top_ask - top_bid
        
        # Store features
        self.store_feature(symbol, 'ob_imbalance', ob_imbalance)
        self.store_feature(symbol, 'wmid', wmid)
        self.store_feature(symbol, 'depth_diff', depth_diff)
        self.store_feature(symbol, 'spread', spread)
    
    def calculate_trade_features(self, symbol):
        """Calculate trade-based features"""
        trades = self.trade_data[symbol][-1000:]  # Last 1000 trades
        
        if not trades:
            return
            
        df = pd.DataFrame(trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Resample to 1-second intervals
        resampled = df.resample('1S').agg({
            'price': 'ohlc',
            'quantity': 'sum',
            'direction': lambda x: (x == 'buy').sum()
        })
        resampled.columns = ['open', 'high', 'low', 'close', 'volume', 'buy_count']
        resampled['sell_count'] = resampled['volume'].count() - resampled['buy_count']
        
        # Trade Imbalance
        resampled['trade_imbalance'] = (resampled['buy_count'] - resampled['sell_count']) / \
                                      (resampled['buy_count'] + resampled['sell_count'] + 1e-10)
        
        # Large Trade Detection
        large_trades = df[df['quantity'] > df['quantity'].quantile(0.95)]
        large_trade_ratio = len(large_trades) / len(df) if len(df) > 0 else 0
        
        # Store features
        self.store_feature(symbol, 'trade_imbalance', resampled['trade_imbalance'].iloc[-1])
        self.store_feature(symbol, 'large_trade_ratio', large_trade_ratio)
    
    def calculate_technical_indicators(self, symbol):
        """Calculate technical indicators"""
        df = self.indicators[symbol]
        if len(df) < 20:  # Ensure enough data
            return
            
        # Calculate indicators
        closes = df['close'].values
        volumes = df['volume'].values
        
        # Moving Averages
        df['ema_10'] = talib.EMA(closes, timeperiod=10)
        df['ema_20'] = talib.EMA(closes, timeperiod=20)
        
        # Oscillators
        df['rsi'] = talib.RSI(closes, timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'], df['low'], closes, 
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # Volatility
        df['atr'] = talib.ATR(df['high'], df['low'], closes, timeperiod=14)
        
        # Volume-based
        df['vwap'] = np.cumsum(df['volume'] * df['close']) / np.cumsum(df['volume'])
        df['obv'] = talib.OBV(closes, volumes)
        
        # Store latest values
        last_row = df.iloc[-1]
        for col in ['ema_10', 'ema_20', 'rsi', 'stoch_k', 'stoch_d', 'atr', 'vwap', 'obv']:
            self.store_feature(symbol, col, last_row[col])
    
    def get_fundamentals(self):
        """Fetch fundamental data (executed periodically)"""
        for symbol in self.symbols:
            # Get funding rate
            fr = client.futures_funding_rate(symbol=symbol)
            funding_rate = float(fr[0]['fundingRate'])
            self.store_feature(symbol, 'funding_rate', funding_rate)
            
            # Get open interest
            oi = client.futures_open_interest(symbol=symbol)
            open_interest = float(oi['openInterest'])
            self.store_feature(symbol, 'open_interest', open_interest)
            
            # Get liquidation data (from WebSocket)
            # Requires separate liquidation stream implementation
    
    def store_feature(self, symbol, feature_name, value):
        """Store feature in database or in-memory structure"""
        # Implement your storage solution here
        # Example: {symbol: {timestamp: {feature: value}}}
        timestamp = int(time.time() * 1000)
        print(f"{timestamp} | {symbol} | {feature_name}: {value}")
        
        # This would typically save to a database or file
        # database.save(symbol, timestamp, feature_name, value)
    
    def run(self):
        """Main execution loop"""
        self.start_data_streams()
        
        # Periodically fetch fundamental data
        while True:
            self.get_fundamentals()
            time.sleep(300)  # Every 5 minutes

# Example usage
if __name__ == "__main__":
    collector = CryptoDataCollector()
    collector.run()