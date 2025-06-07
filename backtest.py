# backtest.py

import backtrader as bt
import pandas as pd
from model_runner import load_model_and_scaler, predict_signal
from datetime import datetime

class SignalStrategy(bt.Strategy):
    def __init__(self, model, scaler, feature_cols):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols

    def next(self):
        features = {col: self.datas[0].lines._getline(col)[0] for col in self.feature_cols}
        df = pd.DataFrame([features])
        signal = predict_signal(self.model, self.scaler, df)
        if not self.position:
            if signal == 1:
                self.buy()
            elif signal == -1:
                self.sell()
        else:
            if signal == 0:
                self.close()

if __name__ == "__main__":
    df = pd.read_csv("data/ethusdt_backtest.csv", parse_dates=["ts"])
    df.set_index("ts", inplace=True)

    model, scaler, feature_cols = load_model_and_scaler()

    data = bt.feeds.PandasData(dataname=df, datetime=None, open=0, high=1, low=2, close=3, volume=4)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SignalStrategy, model=model, scaler=scaler, feature_cols=feature_cols)
    cerebro.adddata(data)
    cerebro.broker.set_cash(100000)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    result = cerebro.run()
    print("Final Portfolio Value:", cerebro.broker.getvalue())
    print("Sharpe Ratio:", result[0].analyzers.sharpe.get_analysis())
    cerebro.plot()