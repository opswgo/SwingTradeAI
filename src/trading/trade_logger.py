#!/usr/bin/env python3
"""
SwingTradeAI - Comprehensive Trade Logging System
User Acceptance: User can see complete trade history with reasoning
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TradeType(Enum):
    CALL = "call"
    PUT = "put"
    STOCK = "stock"

class TradeStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class TradeAction(Enum):
    BUY_TO_OPEN = "buy_to_open"
    SELL_TO_CLOSE = "sell_to_close"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_OPEN = "sell_to_open"

@dataclass
class TradeEntry:
    """Complete trade record with all metadata"""
    trade_id: str
    symbol: str
    trade_type: TradeType
    action: TradeAction
    quantity: int
    entry_price: float
    entry_timestamp: datetime
    
    # AI Reasoning
    ai_confidence: float
    ai_reasoning: str
    market_regime: str
    key_indicators: Dict[str, float]
    
    # Risk Parameters
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size_pct: float
    max_risk_pct: float
    
    # Market Context
    underlying_price: float
    implied_volatility: Optional[float]
    days_to_expiration: Optional[int]
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    
    # Exit Details (populated when closed)
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None
    
    # Performance
    realized_pnl: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    total_return_pct: Optional[float] = None
    holding_period_hours: Optional[float] = None
    
    # Costs
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    slippage_cost: float = 0.0
    
    # Status
    status: TradeStatus = TradeStatus.OPEN
    notes: str = ""

class TradeLogger:
    """Production-ready trade logging with comprehensive tracking"""
    
    def __init__(self, db_path: str = "data/trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database with comprehensive schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    trade_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_timestamp TEXT NOT NULL,
                    
                    -- AI Reasoning
                    ai_confidence REAL NOT NULL,
                    ai_reasoning TEXT NOT NULL,
                    market_regime TEXT NOT NULL,
                    key_indicators TEXT NOT NULL,  -- JSON
                    
                    -- Risk Parameters
                    stop_loss REAL,
                    take_profit REAL,
                    position_size_pct REAL NOT NULL,
                    max_risk_pct REAL NOT NULL,
                    
                    -- Market Context
                    underlying_price REAL NOT NULL,
                    implied_volatility REAL,
                    days_to_expiration INTEGER,
                    delta REAL,
                    gamma REAL,
                    theta REAL,
                    vega REAL,
                    
                    -- Exit Details
                    exit_price REAL,
                    exit_timestamp TEXT,
                    exit_reason TEXT,
                    
                    -- Performance
                    realized_pnl REAL,
                    unrealized_pnl REAL,
                    total_return_pct REAL,
                    holding_period_hours REAL,
                    
                    -- Costs
                    entry_commission REAL DEFAULT 0.0,
                    exit_commission REAL DEFAULT 0.0,
                    slippage_cost REAL DEFAULT 0.0,
                    
                    -- Status
                    status TEXT NOT NULL DEFAULT 'open',
                    notes TEXT DEFAULT '',
                    
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON trades(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entry_timestamp ON trades(entry_timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_type ON trades(trade_type)")
            
            conn.commit()
            
    def log_trade_entry(self, trade: TradeEntry) -> str:
        """Log new trade entry with complete metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert trade to database format
                trade_data = {
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'trade_type': trade.trade_type.value,
                    'action': trade.action.value,
                    'quantity': trade.quantity,
                    'entry_price': trade.entry_price,
                    'entry_timestamp': trade.entry_timestamp.isoformat(),
                    'ai_confidence': trade.ai_confidence,
                    'ai_reasoning': trade.ai_reasoning,
                    'market_regime': trade.market_regime,
                    'key_indicators': json.dumps(trade.key_indicators),
                    'stop_loss': trade.stop_loss,
                    'take_profit': trade.take_profit,
                    'position_size_pct': trade.position_size_pct,
                    'max_risk_pct': trade.max_risk_pct,
                    'underlying_price': trade.underlying_price,
                    'implied_volatility': trade.implied_volatility,
                    'days_to_expiration': trade.days_to_expiration,
                    'delta': trade.delta,
                    'gamma': trade.gamma,
                    'theta': trade.theta,
                    'vega': trade.vega,
                    'entry_commission': trade.entry_commission,
                    'status': trade.status.value,
                    'notes': trade.notes
                }
                
                # Insert trade
                placeholders = ', '.join(['?' for _ in trade_data])
                columns = ', '.join(trade_data.keys())
                
                conn.execute(
                    f"INSERT INTO trades ({columns}) VALUES ({placeholders})",
                    list(trade_data.values())
                )
                conn.commit()
                
                logger.info(f"✅ Trade logged: {trade.trade_id} - {trade.symbol} {trade.trade_type.value}")
                return trade.trade_id
                
        except Exception as e:
            logger.error(f"❌ Failed to log trade: {str(e)}")
            raise
    
    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str, 
                   exit_commission: float = 0.0, slippage_cost: float = 0.0) -> Dict[str, Any]:
        """Close trade and calculate final P&L"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get trade details
                trade_data = conn.execute(
                    "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
                ).fetchone()
                
                if not trade_data:
                    raise ValueError(f"Trade {trade_id} not found")
                
                # Convert to dict for easier access
                columns = [desc[0] for desc in conn.description]
                trade_dict = dict(zip(columns, trade_data))
                
                # Calculate performance
                entry_price = trade_dict['entry_price']
                quantity = trade_dict['quantity']
                entry_commission = trade_dict['entry_commission']
                
                # Calculate P&L
                if trade_dict['action'] in ['buy_to_open', 'buy_to_close']:
                    # Long position
                    price_diff = exit_price - entry_price
                else:
                    # Short position
                    price_diff = entry_price - exit_price
                
                gross_pnl = price_diff * quantity
                total_commissions = entry_commission + exit_commission
                realized_pnl = gross_pnl - total_commissions - slippage_cost
                
                # Calculate return percentage
                total_cost = entry_price * quantity + entry_commission
                total_return_pct = (realized_pnl / total_cost) * 100 if total_cost > 0 else 0
                
                # Calculate holding period
                entry_time = datetime.fromisoformat(trade_dict['entry_timestamp'])
                exit_time = datetime.now(timezone.utc)
                holding_period_hours = (exit_time - entry_time).total_seconds() / 3600
                
                # Update trade record
                conn.execute("""
                    UPDATE trades SET
                        exit_price = ?,
                        exit_timestamp = ?,
                        exit_reason = ?,
                        exit_commission = ?,
                        slippage_cost = ?,
                        realized_pnl = ?,
                        total_return_pct = ?,
                        holding_period_hours = ?,
                        status = 'closed',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE trade_id = ?
                """, (
                    exit_price, exit_time.isoformat(), exit_reason,
                    exit_commission, slippage_cost, realized_pnl,
                    total_return_pct, holding_period_hours, trade_id
                ))
                conn.commit()
                
                result = {
                    'trade_id': trade_id,
                    'symbol': trade_dict['symbol'],
                    'realized_pnl': realized_pnl,
                    'total_return_pct': total_return_pct,
                    'holding_period_hours': holding_period_hours,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason
                }
                
                logger.info(f"✅ Trade closed: {trade_id} - P&L: ${realized_pnl:.2f} ({total_return_pct:.1f}%)")
                return result
                
        except Exception as e:
            logger.error(f"❌ Failed to close trade: {str(e)}")
            raise
    
    def get_trade_history(self, symbol: Optional[str] = None, 
                         status: Optional[TradeStatus] = None,
                         limit: int = 100) -> pd.DataFrame:
        """Get trade history with optional filtering"""
        try:
            query = "SELECT * FROM trades"
            params = []
            conditions = []
            
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if status:
                conditions.append("status = ?")
                params.append(status.value)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY entry_timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
                # Parse JSON fields
                if not df.empty:
                    df['key_indicators'] = df['key_indicators'].apply(json.loads)
                    df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'])
                    df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])
                
                return df
                
        except Exception as e:
            logger.error(f"❌ Failed to get trade history: {str(e)}")
            return pd.DataFrame()
    
    def get_performance_summary(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN status = 'closed' THEN 1 END) as closed_trades,
                        COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
                        COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as losing_trades,
                        SUM(realized_pnl) as total_pnl,
                        AVG(realized_pnl) as avg_pnl,
                        MAX(realized_pnl) as max_win,
                        MIN(realized_pnl) as max_loss,
                        AVG(total_return_pct) as avg_return_pct,
                        AVG(holding_period_hours) as avg_holding_hours,
                        SUM(entry_commission + exit_commission + slippage_cost) as total_costs
                    FROM trades
                """
                
                params = []
                if symbol:
                    query += " WHERE symbol = ?"
                    params.append(symbol)
                
                result = conn.execute(query, params).fetchone()
                
                if result:
                    columns = [desc[0] for desc in conn.description]
                    summary = dict(zip(columns, result))
                    
                    # Calculate additional metrics
                    if summary['closed_trades'] > 0:
                        summary['win_rate'] = (summary['winning_trades'] / summary['closed_trades']) * 100
                        summary['profit_factor'] = abs(summary['max_win'] / summary['max_loss']) if summary['max_loss'] < 0 else 0
                    else:
                        summary['win_rate'] = 0
                        summary['profit_factor'] = 0
                    
                    return summary
                
                return {}
                
        except Exception as e:
            logger.error(f"❌ Failed to get performance summary: {str(e)}")
            return {}
    
    def update_unrealized_pnl(self, current_prices: Dict[str, float]):
        """Update unrealized P&L for open positions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all open trades
                open_trades = conn.execute(
                    "SELECT trade_id, symbol, action, quantity, entry_price FROM trades WHERE status = 'open'"
                ).fetchall()
                
                for trade_id, symbol, action, quantity, entry_price in open_trades:
                    if symbol in current_prices:
                        current_price = current_prices[symbol]
                        
                        # Calculate unrealized P&L
                        if action in ['buy_to_open', 'buy_to_close']:
                            # Long position
                            price_diff = current_price - entry_price
                        else:
                            # Short position
                            price_diff = entry_price - current_price
                        
                        unrealized_pnl = price_diff * quantity
                        
                        # Update database
                        conn.execute(
                            "UPDATE trades SET unrealized_pnl = ?, updated_at = CURRENT_TIMESTAMP WHERE trade_id = ?",
                            (unrealized_pnl, trade_id)
                        )
                
                conn.commit()
                logger.info(f"✅ Updated unrealized P&L for {len(open_trades)} open positions")
                
        except Exception as e:
            logger.error(f"❌ Failed to update unrealized P&L: {str(e)}")

    def export_trades(self, filepath: str, format: str = 'csv'):
        """Export trade history to file"""
        try:
            df = self.get_trade_history(limit=None)
            
            if format.lower() == 'csv':
                df.to_csv(filepath, index=False)
            elif format.lower() == 'json':
                df.to_json(filepath, orient='records', date_format='iso')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"✅ Exported {len(df)} trades to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export trades: {str(e)}")
            raise
