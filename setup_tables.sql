-- setup_tables.sql
-- ClickHouse tables for the HFT trading system

-- Table for storing market features
CREATE TABLE IF NOT EXISTS futures_features (
    ts DateTime64(3),
    symbol String,
    features String,  -- JSON string of computed features
    raw_data String   -- JSON string of raw market data
) ENGINE = MergeTree()
ORDER BY (symbol, ts)
TTL ts + INTERVAL 30 DAY;

-- Table for storing model predictions/signals
CREATE TABLE IF NOT EXISTS executed_trades (
    ts DateTime64(3),
    symbol String,
    signal Int8,  -- 1=buy, -1=sell, 0=hold
    score Float64
) ENGINE = MergeTree()
ORDER BY (symbol, ts)
TTL ts + INTERVAL 30 DAY;

-- Table for storing actual trade executions
CREATE TABLE IF NOT EXISTS trade_executions (
    ts DateTime64(3),
    symbol String,
    signal Int8,  -- 1=buy, -1=sell, 0=hold
    score Float64,
    executed UInt8,  -- 1=executed, 0=not executed
    price Float64,
    position_size Float64
) ENGINE = MergeTree()
ORDER BY (symbol, ts)
TTL ts + INTERVAL 30 DAY;

-- Table for storing liquidation events
CREATE TABLE IF NOT EXISTS liquidation_events (
    ts DateTime64(3),
    symbol String,
    liquidation_cluster_size Float64,
    liquidation_imbalance Float64
) ENGINE = MergeTree()
ORDER BY (symbol, ts)
TTL ts + INTERVAL 30 DAY;

-- Table for storing risk management checks
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
TTL ts + INTERVAL 30 DAY;

-- Note: Secondary indexes are not strictly necessary here because
-- the ORDER BY (symbol, ts) already optimizes the common query pattern.
-- If needed, add skipping indexes with an explicit TYPE. 