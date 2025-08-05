# ui_dashboard_fixed.py

"""
Fixed Streamlit dashboard for the HFT trading system.
Handles connection errors gracefully and doesn't rely on external services.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from clickhouse_connect import get_client
import numpy as np
import logging
from datetime import datetime, timedelta

# -------------------- Config --------------------

CLICKHOUSE_HOST = 'clickhouse'
CLICKHOUSE_USER = 'default'
CLICKHOUSE_PASS = ''

# -------------------- Error Handling --------------------

def safe_clickhouse_query(query, default_df=None):
    """Safely execute ClickHouse query with error handling."""
    try:
        client = get_client(
            host=CLICKHOUSE_HOST,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASS
        )
        rows = client.query(query).result_rows
        return rows
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        logging.error(f"ClickHouse query failed: {e}")
        return []

def safe_dataframe(rows, columns, default_df=None):
    """Safely create DataFrame with error handling."""
    try:
        if not rows:
            return default_df if default_df is not None else pd.DataFrame()
        
        df = pd.DataFrame(rows, columns=columns)
        if 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'])
        return df.sort_values('ts') if 'ts' in df.columns else df
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        logging.error(f"DataFrame creation failed: {e}")
        return default_df if default_df is not None else pd.DataFrame()

# -------------------- Page Config --------------------

st.set_page_config(
    page_title="HFT Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Sidebar --------------------

st.sidebar.title("🎛️ Trading Controls")

# Connection status
try:
    test_client = get_client(
        host=CLICKHOUSE_HOST,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASS
    )
    test_client.ping()
    st.sidebar.success("✅ Database Connected")
except Exception as e:
    st.sidebar.error("❌ Database Disconnected")
    st.sidebar.error(f"Error: {str(e)}")

# Time range selector
st.sidebar.subheader("📅 Time Range")
time_range = st.sidebar.selectbox(
    "Select time range:",
    ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"],
    index=0
)

# -------------------- Helper Queries --------------------

def get_signals():
    """Get trading signals with error handling."""
    q = """
    SELECT ts, symbol, signal, score 
    FROM executed_trades
    ORDER BY ts DESC 
    LIMIT 500
    """
    rows = safe_clickhouse_query(q)
    return safe_dataframe(rows, ['ts', 'symbol', 'signal', 'score'])

def get_trade_executions():
    """Get trade executions with error handling."""
    q = """
    SELECT ts, symbol, signal, score, executed, price, position_size 
    FROM trade_executions
    ORDER BY ts DESC 
    LIMIT 500
    """
    rows = safe_clickhouse_query(q)
    return safe_dataframe(rows, ['ts', 'symbol', 'signal', 'score', 'executed', 'price', 'position_size'])

def get_risk_checks():
    """Get risk management checks with error handling."""
    q = """
    SELECT ts, symbol, signal, allowed, reason, daily_pnl, max_drawdown, exposure_pct, account_balance
    FROM risk_checks
    ORDER BY ts DESC 
    LIMIT 500
    """
    rows = safe_clickhouse_query(q)
    return safe_dataframe(rows, ['ts', 'symbol', 'signal', 'allowed', 'reason', 'daily_pnl', 'max_drawdown', 'exposure_pct', 'account_balance'])

def get_liquidation():
    """Get liquidation events with error handling."""
    q = """
    SELECT ts, symbol, liquidation_cluster_size, liquidation_imbalance 
    FROM liquidation_events
    ORDER BY ts DESC 
    LIMIT 300
    """
    rows = safe_clickhouse_query(q)
    return safe_dataframe(rows, ['ts', 'symbol', 'cluster_size', 'imbalance'])

def get_indicators():
    """Get technical indicators with error handling."""
    q = """
    SELECT ts, features 
    FROM futures_features
    ORDER BY ts DESC 
    LIMIT 300
    """
    rows = safe_clickhouse_query(q)
    if not rows:
        return pd.DataFrame()
    
    try:
        ts_list = [r[0] for r in rows]
        features_list = []
        for r in rows:
            try:
                features = eval(r[1]) if isinstance(r[1], str) else r[1]
                features_list.append(features)
            except:
                features_list.append({})
        
        df_feat = pd.DataFrame(features_list)
        df_feat['ts'] = pd.to_datetime(ts_list)
        return df_feat.sort_values('ts')
    except Exception as e:
        st.error(f"Error processing indicators: {str(e)}")
        return pd.DataFrame()

# -------------------- Main Dashboard --------------------

st.title("📈 HFT Trading Dashboard")
st.markdown("---")

# -------------------- Key Metrics Row --------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    try:
        df_signals = get_signals()
        total_signals = len(df_signals)
        st.metric("Total Signals", total_signals)
    except:
        st.metric("Total Signals", "N/A")

with col2:
    try:
        df_executions = get_trade_executions()
        executed_trades = len(df_executions[df_executions['executed'] == 1]) if not df_executions.empty else 0
        st.metric("Executed Trades", executed_trades)
    except:
        st.metric("Executed Trades", "N/A")

with col3:
    try:
        df_risk = get_risk_checks()
        if not df_risk.empty:
            latest_risk = df_risk.iloc[0]
            account_balance = latest_risk.get('account_balance', 0)
            st.metric("Account Balance", f"${account_balance:,.2f}")
        else:
            st.metric("Account Balance", "N/A")
    except:
        st.metric("Account Balance", "N/A")

with col4:
    try:
        df_risk = get_risk_checks()
        if not df_risk.empty:
            latest_risk = df_risk.iloc[0]
            daily_pnl = latest_risk.get('daily_pnl', 0)
            color = "normal" if daily_pnl >= 0 else "inverse"
            st.metric("Daily PnL", f"${daily_pnl:,.2f}", delta=f"{daily_pnl:+.2f}", delta_color=color)
        else:
            st.metric("Daily PnL", "N/A")
    except:
        st.metric("Daily PnL", "N/A")

# -------------------- Charts Row --------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Trading Signals")
    df_signals = get_signals()
    if not df_signals.empty:
        fig = px.step(df_signals, x='ts', y='signal', color='symbol', 
                     markers=True, title="Model Trading Signals")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trading signals available")

    st.subheader("💰 Trade Executions")
    df_executions = get_trade_executions()
    if not df_executions.empty:
        fig_exec = px.scatter(df_executions, x='ts', y='price', 
                             color='executed', size='position_size',
                             title="Real Trade Executions")
        st.plotly_chart(fig_exec, use_container_width=True)
        
        # Execution statistics
        total_signals = len(df_executions)
        executed_trades = len(df_executions[df_executions['executed'] == 1])
        execution_rate = (executed_trades / total_signals * 100) if total_signals > 0 else 0
        
        st.metric("Execution Rate", f"{execution_rate:.1f}%")
    else:
        st.info("No trade executions recorded yet")

with col2:
    st.subheader("⚠️ Risk Management")
    df_risk = get_risk_checks()
    if not df_risk.empty:
        # Daily PnL chart
        fig_pnl = px.line(df_risk, x='ts', y='daily_pnl', 
                         title="Daily PnL Tracking")
        st.plotly_chart(fig_pnl, use_container_width=True)
        
        # Drawdown chart
        fig_dd = px.line(df_risk, x='ts', y='max_drawdown', 
                        title="Maximum Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)
        
        # Latest risk metrics
        latest = df_risk.iloc[0]
        st.metric("Max Drawdown", f"{latest.get('max_drawdown', 0):.2%}")
        st.metric("Exposure", f"{latest.get('exposure_pct', 0):.1f}%")
    else:
        st.info("No risk management data available")

    st.subheader("💥 Liquidation Events")
    df_liq = get_liquidation()
    if not df_liq.empty:
        fig_liq = px.line(df_liq, x='ts', y='cluster_size', 
                         color='symbol', title="Liquidation Cluster Size")
        st.plotly_chart(fig_liq, use_container_width=True)
    else:
        st.info("No liquidation events recorded")

# -------------------- Technical Analysis --------------------

st.subheader("📈 Technical Indicators")
df_ind = get_indicators()

if not df_ind.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        if 'rsi' in df_ind.columns:
            fig_rsi = px.line(df_ind, x='ts', y='rsi', title="RSI")
            st.plotly_chart(fig_rsi, use_container_width=True)
        else:
            st.info("RSI data not available")
    
    with col2:
        if 'atr' in df_ind.columns:
            fig_atr = px.line(df_ind, x='ts', y='atr', title="ATR")
            st.plotly_chart(fig_atr, use_container_width=True)
        else:
            st.info("ATR data not available")
else:
    st.info("No technical indicators available")

# -------------------- Recent Activity --------------------

st.subheader("📋 Recent Activity")

col1, col2 = st.columns(2)

with col1:
    st.write("**Latest Signals**")
    df_signals = get_signals()
    if not df_signals.empty:
        recent_signals = df_signals.head(10)[['ts', 'symbol', 'signal', 'score']]
        st.dataframe(recent_signals, use_container_width=True)
    else:
        st.info("No signals available")

with col2:
    st.write("**Latest Risk Checks**")
    df_risk = get_risk_checks()
    if not df_risk.empty:
        recent_risk = df_risk.head(10)[['ts', 'signal', 'allowed', 'reason']]
        st.dataframe(recent_risk, use_container_width=True)
    else:
        st.info("No risk checks available")

# -------------------- Footer --------------------

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>HFT Auto Trade Dashboard | Built with Streamlit</p>
    <p>Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True) 