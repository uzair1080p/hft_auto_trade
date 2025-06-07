import streamlit as st
import pandas as pd
import plotly.express as px
from clickhouse_connect import get_client
import numpy as np

# -------------------- Config --------------------

CLICKHOUSE_HOST = 'clickhouse'
CLICKHOUSE_USER = 'default'
CLICKHOUSE_PASS = ''

client = get_client(
    host=CLICKHOUSE_HOST,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASS
)

st.set_page_config(layout="wide")
st.title("📈 Trading Agent Dashboard")

# -------------------- Helper Queries --------------------

def get_signals():
    q = """
    SELECT ts, symbol, signal, score FROM executed_trades
    ORDER BY ts DESC LIMIT 500
    """
    rows = client.query(q).result_rows
    df = pd.DataFrame(rows, columns=['ts', 'symbol', 'signal', 'score'])
    df['ts'] = pd.to_datetime(df['ts'])
    return df.sort_values('ts')

def get_liquidation():
    q = """
    SELECT timestamp, symbol, liquidation_cluster_size, liquidation_imbalance FROM liquidation_features
    ORDER BY timestamp DESC LIMIT 300
    """
    rows = client.query(q).result_rows
    df = pd.DataFrame(rows, columns=['ts', 'symbol', 'cluster_size', 'imbalance'])
    df['ts'] = pd.to_datetime(df['ts'])
    return df.sort_values('ts')

def get_indicators():
    q = """
    SELECT ts, features FROM futures_features
    ORDER BY ts DESC LIMIT 300
    """
    rows = client.query(q).result_rows
    ts_list = [r[0] for r in rows]
    df_feat = pd.DataFrame([eval(r[1]) for r in rows])
    df_feat['ts'] = pd.to_datetime(ts_list)
    return df_feat.sort_values('ts')

def get_backtest_prices():
    q = """
    SELECT ts, features FROM futures_features
    ORDER BY ts DESC LIMIT 500
    """
    rows = client.query(q).result_rows
    ts_list = [r[0] for r in rows]
    df_feat = pd.DataFrame([eval(r[1]) for r in rows])
    df_feat['ts'] = pd.to_datetime(ts_list)
    df_feat['close'] = df_feat.get('wmid', df_feat.get('price', None))  # fallback
    return df_feat[['ts', 'close']].sort_values('ts')

# -------------------- PnL Calculation --------------------

def calculate_pnl(signal_df, price_df):
    df = pd.merge_asof(signal_df.sort_values("ts"), price_df.sort_values("ts"), on="ts")
    df['position'] = df['signal'].shift(1).fillna(0)
    df['return'] = df['close'].pct_change().fillna(0)
    df['strategy_return'] = df['return'] * df['position']
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    df['rolling_max'] = df['cumulative_return'].cummax()
    df['drawdown'] = (df['cumulative_return'] - df['rolling_max']) / df['rolling_max']
    return df

def calculate_metrics(pnl_df):
    daily_return = pnl_df['strategy_return']
    sharpe = np.sqrt(252) * daily_return.mean() / (daily_return.std() + 1e-9)

    trades = pnl_df[pnl_df['position'].diff() != 0]
    trade_returns = trades['strategy_return'].dropna()
    wins = (trade_returns > 0).sum()
    losses = (trade_returns < 0).sum()

    win_rate = wins / (wins + losses + 1e-6)
    return {
        'Sharpe Ratio': round(sharpe, 2),
        'Win/Loss Ratio': f"{wins}:{losses}",
        'Total Trades': int((pnl_df['position'].diff() != 0).sum())
    }, trades[['ts', 'position', 'close', 'strategy_return']]

# -------------------- Layout --------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Signals")
    df_signals = get_signals()
    fig = px.step(df_signals, x='ts', y='signal', color='symbol', markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Signal Scores")
    fig2 = px.line(df_signals, x='ts', y='score', color='symbol')
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Liquidation Clusters")
    df_liq = get_liquidation()
    fig3 = px.line(df_liq, x='ts', y='cluster_size', color='symbol', title='Cluster Size')
    fig4 = px.line(df_liq, x='ts', y='imbalance', color='symbol', title='Imbalance')
    st.plotly_chart(fig3, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("Technical Indicators")
df_ind = get_indicators()
fig5 = px.line(df_ind, x='ts', y='rsi', title='RSI')
fig6 = px.line(df_ind, x='ts', y='atr', title='ATR')
st.plotly_chart(fig5, use_container_width=True)
st.plotly_chart(fig6, use_container_width=True)

# -------------------- PnL & Metrics --------------------

st.subheader("📊 Strategy PnL & Metrics")
price_df = get_backtest_prices()
if not df_signals.empty and not price_df.empty:
    pnl_df = calculate_pnl(df_signals, price_df)
    metrics, trades_log = calculate_metrics(pnl_df)

    fig7 = px.line(pnl_df, x='ts', y='cumulative_return', title='Cumulative Return')
    fig8 = px.area(pnl_df, x='ts', y='drawdown', title='Drawdown', range_y=[-1, 0])
    st.plotly_chart(fig7, use_container_width=True)
    st.plotly_chart(fig8, use_container_width=True)

    st.metric("Sharpe Ratio", metrics['Sharpe Ratio'])
    st.metric("Win/Loss", metrics['Win/Loss Ratio'])
    st.metric("Total Trades", metrics['Total Trades'])

    st.subheader("🔍 Trade-by-Trade Log")
    trades_log.rename(columns={'ts': 'Timestamp', 'position': 'Position', 'close': 'Price', 'strategy_return': 'Return'}, inplace=True)
    st.dataframe(trades_log.sort_values('Timestamp', ascending=False).reset_index(drop=True), use_container_width=True)
else:
    st.warning("Not enough data to calculate PnL or metrics.")