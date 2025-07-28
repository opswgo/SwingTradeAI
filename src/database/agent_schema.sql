-- NEAT Options System - Agent Database Schema
-- Per-Stock Evolutionary Agent Architecture

-- Main agent storage table
CREATE TABLE IF NOT EXISTS stock_agents (
    agent_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    agent_version INTEGER NOT NULL,
    network_weights BLOB NOT NULL,
    network_config JSON NOT NULL,
    fitness_score REAL NOT NULL,
    accuracy REAL NOT NULL,
    total_return REAL NOT NULL,
    sharpe_ratio REAL NOT NULL,
    max_drawdown REAL NOT NULL,
    win_rate REAL NOT NULL,
    total_trades INTEGER NOT NULL,
    training_data_start DATE NOT NULL,
    training_data_end DATE NOT NULL,
    validation_accuracy REAL NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    is_champion BOOLEAN DEFAULT FALSE,
    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    training_generations INTEGER NOT NULL,
    overfitting_score REAL,
    notes TEXT
);

-- Performance tracking for each prediction
CREATE TABLE IF NOT EXISTS agent_performance (
    performance_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    prediction_date DATE NOT NULL,
    prediction_horizon_weeks INTEGER NOT NULL,
    predicted_price REAL NOT NULL,
    predicted_direction TEXT NOT NULL, -- 'UP', 'DOWN', 'SIDEWAYS'
    confidence_score REAL NOT NULL,
    actual_price REAL,
    actual_direction TEXT,
    prediction_accuracy REAL,
    price_error_pct REAL,
    direction_correct BOOLEAN,
    trade_executed BOOLEAN DEFAULT FALSE,
    trade_pnl REAL,
    options_strategy TEXT, -- 'SELL_PUT', 'SELL_CALL', 'IRON_CONDOR', etc.
    strike_price REAL,
    premium_collected REAL,
    assignment_occurred BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES stock_agents(agent_id)
);

-- Historical winners archive - NEVER DELETE
CREATE TABLE IF NOT EXISTS agent_history (
    history_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    agent_version INTEGER NOT NULL,
    network_weights BLOB NOT NULL,
    network_config JSON NOT NULL,
    performance_snapshot JSON NOT NULL, -- Full performance metrics at time of archival
    archive_reason TEXT NOT NULL, -- 'SUPERSEDED', 'REGIME_CHANGE', 'SCHEDULED_BACKUP'
    archive_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fitness_score REAL NOT NULL,
    accuracy REAL NOT NULL,
    total_return REAL NOT NULL,
    sharpe_ratio REAL NOT NULL,
    training_period TEXT NOT NULL,
    validation_period TEXT NOT NULL,
    can_restore BOOLEAN DEFAULT TRUE,
    notes TEXT
);

-- Market regime detection and labeling
CREATE TABLE IF NOT EXISTS market_regimes (
    regime_id TEXT PRIMARY KEY,
    start_date DATE NOT NULL,
    end_date DATE,
    regime_type TEXT NOT NULL, -- 'BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOL', 'LOW_VOL'
    vix_avg REAL,
    spy_return REAL,
    volatility REAL,
    confidence REAL NOT NULL,
    detected_by TEXT NOT NULL, -- 'MANUAL', 'AUTO_VIX', 'AUTO_TREND', 'AUTO_ML'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training sessions and results
CREATE TABLE IF NOT EXISTS training_sessions (
    session_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    population_size INTEGER NOT NULL,
    generations_completed INTEGER NOT NULL,
    best_fitness REAL,
    best_accuracy REAL,
    training_data_start DATE NOT NULL,
    training_data_end DATE NOT NULL,
    validation_data_start DATE NOT NULL,
    validation_data_end DATE NOT NULL,
    hyperparameters JSON NOT NULL,
    status TEXT NOT NULL, -- 'RUNNING', 'COMPLETED', 'FAILED', 'STOPPED'
    error_message TEXT,
    champion_agent_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (champion_agent_id) REFERENCES stock_agents(agent_id)
);

-- Real-time monitoring and alerts
CREATE TABLE IF NOT EXISTS agent_monitoring (
    monitor_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    check_time TIMESTAMP NOT NULL,
    accuracy_7d REAL,
    accuracy_30d REAL,
    recent_trades INTEGER,
    recent_pnl REAL,
    alert_triggered BOOLEAN DEFAULT FALSE,
    alert_type TEXT, -- 'ACCURACY_DROP', 'PERFORMANCE_DECLINE', 'OVERFITTING'
    alert_message TEXT,
    action_taken TEXT, -- 'RETRAIN_TRIGGERED', 'AGENT_DISABLED', 'MANUAL_REVIEW'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES stock_agents(agent_id)
);

-- Stock metadata and selection criteria
CREATE TABLE IF NOT EXISTS stock_metadata (
    symbol TEXT PRIMARY KEY,
    company_name TEXT,
    sector TEXT,
    market_cap REAL,
    avg_volume_30d REAL,
    options_volume_30d REAL,
    iv_rank REAL,
    beta REAL,
    last_earnings_date DATE,
    next_earnings_date DATE,
    is_tradeable BOOLEAN DEFAULT TRUE,
    priority_tier INTEGER, -- 1=highest, 5=lowest
    data_quality_score REAL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_stock_agents_symbol ON stock_agents(symbol);
CREATE INDEX IF NOT EXISTS idx_stock_agents_active ON stock_agents(is_active);
CREATE INDEX IF NOT EXISTS idx_stock_agents_champion ON stock_agents(is_champion);
CREATE INDEX IF NOT EXISTS idx_agent_performance_symbol_date ON agent_performance(symbol, prediction_date);
CREATE INDEX IF NOT EXISTS idx_agent_performance_agent_id ON agent_performance(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_history_symbol ON agent_history(symbol);
CREATE INDEX IF NOT EXISTS idx_training_sessions_symbol ON training_sessions(symbol);
CREATE INDEX IF NOT EXISTS idx_agent_monitoring_symbol ON agent_monitoring(symbol);

-- Views for common queries
CREATE VIEW IF NOT EXISTS active_agents AS
SELECT 
    sa.*,
    sm.company_name,
    sm.sector,
    sm.priority_tier
FROM stock_agents sa
JOIN stock_metadata sm ON sa.symbol = sm.symbol
WHERE sa.is_active = TRUE;

CREATE VIEW IF NOT EXISTS champion_agents AS
SELECT 
    sa.*,
    sm.company_name,
    sm.sector
FROM stock_agents sa
JOIN stock_metadata sm ON sa.symbol = sm.symbol
WHERE sa.is_champion = TRUE;

CREATE VIEW IF NOT EXISTS recent_performance AS
SELECT 
    ap.symbol,
    ap.agent_id,
    COUNT(*) as total_predictions,
    AVG(ap.prediction_accuracy) as avg_accuracy,
    AVG(ap.trade_pnl) as avg_pnl,
    SUM(ap.trade_pnl) as total_pnl,
    COUNT(CASE WHEN ap.direction_correct = TRUE THEN 1 END) * 100.0 / COUNT(*) as direction_accuracy
FROM agent_performance ap
WHERE ap.prediction_date >= date('now', '-30 days')
GROUP BY ap.symbol, ap.agent_id;
