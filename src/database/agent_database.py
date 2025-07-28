"""
Agent Database Manager - NEAT Options System
Handles all database operations for per-stock evolutionary agents.
"""

import sqlite3
import json
import uuid
import pickle
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

class AgentDatabase:
    """Database manager for evolutionary agents and performance tracking."""
    
    def __init__(self, db_path: str = "data/neat_agents.db"):
        """Initialize database connection and create tables."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize database with schema."""
        schema_path = Path(__file__).parent / "agent_schema.sql"
        
        with sqlite3.connect(self.db_path) as conn:
            # Read and execute schema
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            conn.executescript(schema_sql)
            conn.commit()
            
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def save_agent(self, symbol: str, network_weights: np.ndarray, 
                   network_config: Dict, performance_metrics: Dict,
                   training_info: Dict) -> str:
        """Save a trained agent to database."""
        
        agent_id = str(uuid.uuid4())
        
        # Serialize network weights
        weights_blob = pickle.dumps(network_weights)
        config_json = json.dumps(network_config)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO stock_agents (
                    agent_id, symbol, agent_version, network_weights, network_config,
                    fitness_score, accuracy, total_return, sharpe_ratio, max_drawdown,
                    win_rate, total_trades, training_data_start, training_data_end,
                    validation_accuracy, training_generations, overfitting_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_id, symbol, training_info.get('version', 1),
                weights_blob, config_json,
                performance_metrics['fitness_score'],
                performance_metrics['accuracy'],
                performance_metrics['total_return'],
                performance_metrics['sharpe_ratio'],
                performance_metrics['max_drawdown'],
                performance_metrics['win_rate'],
                performance_metrics['total_trades'],
                training_info['data_start'],
                training_info['data_end'],
                performance_metrics['validation_accuracy'],
                training_info['generations'],
                performance_metrics.get('overfitting_score', 0.0)
            ))
            
            conn.commit()
        
        self.logger.info(f"Saved agent {agent_id} for {symbol}")
        return agent_id
    
    def load_agent(self, symbol: str, agent_id: Optional[str] = None) -> Optional[Dict]:
        """Load agent for a symbol (champion if no agent_id specified)."""
        
        with sqlite3.connect(self.db_path) as conn:
            if agent_id:
                cursor = conn.execute("""
                    SELECT * FROM stock_agents WHERE agent_id = ?
                """, (agent_id,))
            else:
                # Load champion agent for symbol
                cursor = conn.execute("""
                    SELECT * FROM stock_agents 
                    WHERE symbol = ? AND is_champion = TRUE
                    ORDER BY fitness_score DESC LIMIT 1
                """, (symbol,))
            
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                agent_data = dict(zip(columns, row))
                
                # Deserialize network weights
                agent_data['network_weights'] = pickle.loads(agent_data['network_weights'])
                agent_data['network_config'] = json.loads(agent_data['network_config'])
                
                return agent_data
            
        return None
    
    def set_champion_agent(self, symbol: str, agent_id: str):
        """Set an agent as the champion for a symbol."""
        
        with sqlite3.connect(self.db_path) as conn:
            # Remove champion status from all agents for this symbol
            conn.execute("""
                UPDATE stock_agents 
                SET is_champion = FALSE, is_active = FALSE
                WHERE symbol = ?
            """, (symbol,))
            
            # Set new champion
            conn.execute("""
                UPDATE stock_agents 
                SET is_champion = TRUE, is_active = TRUE
                WHERE agent_id = ?
            """, (agent_id,))
            
            conn.commit()
        
        self.logger.info(f"Set agent {agent_id} as champion for {symbol}")
    
    def archive_agent(self, agent_id: str, reason: str = "SUPERSEDED"):
        """Archive an agent to historical winners."""
        
        with sqlite3.connect(self.db_path) as conn:
            # Get agent data
            cursor = conn.execute("""
                SELECT * FROM stock_agents WHERE agent_id = ?
            """, (agent_id,))
            
            agent_row = cursor.fetchone()
            if not agent_row:
                return
            
            columns = [desc[0] for desc in cursor.description]
            agent_data = dict(zip(columns, agent_row))
            
            # Create performance snapshot
            performance_snapshot = {
                'fitness_score': agent_data['fitness_score'],
                'accuracy': agent_data['accuracy'],
                'total_return': agent_data['total_return'],
                'sharpe_ratio': agent_data['sharpe_ratio'],
                'max_drawdown': agent_data['max_drawdown'],
                'win_rate': agent_data['win_rate'],
                'total_trades': agent_data['total_trades'],
                'validation_accuracy': agent_data['validation_accuracy']
            }
            
            # Archive to history
            history_id = str(uuid.uuid4())
            conn.execute("""
                INSERT INTO agent_history (
                    history_id, agent_id, symbol, agent_version,
                    network_weights, network_config, performance_snapshot,
                    archive_reason, fitness_score, accuracy, total_return,
                    sharpe_ratio, training_period, validation_period
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                history_id, agent_id, agent_data['symbol'], agent_data['agent_version'],
                agent_data['network_weights'], agent_data['network_config'],
                json.dumps(performance_snapshot), reason,
                agent_data['fitness_score'], agent_data['accuracy'],
                agent_data['total_return'], agent_data['sharpe_ratio'],
                f"{agent_data['training_data_start']} to {agent_data['training_data_end']}",
                f"Validation: {agent_data['validation_accuracy']:.3f}"
            ))
            
            conn.commit()
        
        self.logger.info(f"Archived agent {agent_id} with reason: {reason}")
    
    def log_performance(self, agent_id: str, symbol: str, performance_data: Dict):
        """Log performance data for an agent."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO agent_performance (
                    agent_id, symbol, date, fitness_score, accuracy,
                    total_return, sharpe_ratio, max_drawdown, win_rate, total_trades
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_id,
                symbol,
                performance_data['date'],
                performance_data['fitness_score'],
                performance_data['accuracy'],
                performance_data['total_return'],
                performance_data['sharpe_ratio'],
                performance_data.get('max_drawdown', 0.0),
                performance_data.get('win_rate', 0.0),
                performance_data['total_trades']
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Logged performance for {symbol} by agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to log performance: {str(e)}")
    
    def get_performance_history(self, symbol: str = None, days: int = 30, min_trades: int = 0) -> List[Dict]:
        """Get performance history for agents."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT agent_id, symbol, date, fitness_score, accuracy,
                       total_return, sharpe_ratio, max_drawdown, win_rate, total_trades
                FROM agent_performance
                WHERE date >= date('now', '-{} days')
            """.format(days)
            
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if min_trades > 0:
                query += " AND total_trades >= ?"
                params.append(min_trades)
            
            query += " ORDER BY date DESC"
            
            cursor.execute(query, params)
            
            performance_data = []
            for row in cursor.fetchall():
                performance_data.append({
                    'agent_id': row[0],
                    'symbol': row[1],
                    'date': row[2],
                    'fitness_score': row[3],
                    'accuracy': row[4],
                    'total_return': row[5],
                    'sharpe_ratio': row[6],
                    'max_drawdown': row[7],
                    'win_rate': row[8],
                    'total_trades': row[9]
                })
            
            conn.close()
            return performance_data
            
        except Exception as e:
            self.logger.error(f"Failed to get performance history: {str(e)}")
            return []
    
    def log_prediction(self, agent_id: str, symbol: str, prediction_data: Dict) -> str:
        """Log a prediction made by an agent."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions (
                    agent_id, symbol, prediction_date, horizon_weeks,
                    predicted_price, predicted_direction, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_id,
                symbol,
                prediction_data['date'],
                prediction_data['horizon_weeks'],
                prediction_data['predicted_price'],
                prediction_data['predicted_direction'],
                prediction_data['confidence']
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Logged prediction for {symbol} by agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to log prediction: {str(e)}")
    
    def update_prediction_outcome(self, performance_id: str, actual_price: float,
                                  actual_direction: str, trade_pnl: float = 0):
        """Update prediction with actual outcome."""
        
        with sqlite3.connect(self.db_path) as conn:
            # Get prediction data
            cursor = conn.execute("""
                SELECT predicted_price, predicted_direction FROM agent_performance
                WHERE performance_id = ?
            """, (performance_id,))
            
            row = cursor.fetchone()
            if not row:
                return
            
            predicted_price, predicted_direction = row
            
            # Calculate accuracy metrics
            price_error_pct = abs(actual_price - predicted_price) / predicted_price
            direction_correct = (predicted_direction == actual_direction)
            prediction_accuracy = 1.0 - min(price_error_pct, 1.0)  # Cap at 0
            
            # Update record
            conn.execute("""
                UPDATE agent_performance SET
                    actual_price = ?, actual_direction = ?,
                    prediction_accuracy = ?, price_error_pct = ?,
                    direction_correct = ?, trade_pnl = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE performance_id = ?
            """, (
                actual_price, actual_direction, prediction_accuracy,
                price_error_pct, direction_correct, trade_pnl, performance_id
            ))
            
            conn.commit()
    
    def get_agent_performance(self, symbol: str, days: int = 30) -> Dict:
        """Get recent performance metrics for a symbol's champion agent."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(prediction_accuracy) as avg_accuracy,
                    AVG(CASE WHEN direction_correct = 1 THEN 1.0 ELSE 0.0 END) as direction_accuracy,
                    AVG(trade_pnl) as avg_pnl,
                    SUM(trade_pnl) as total_pnl,
                    MIN(prediction_date) as first_prediction,
                    MAX(prediction_date) as last_prediction
                FROM agent_performance ap
                JOIN stock_agents sa ON ap.agent_id = sa.agent_id
                WHERE sa.symbol = ? AND sa.is_champion = TRUE
                    AND ap.prediction_date >= date('now', '-{} days')
            """.format(days), (symbol,))
            
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            
        return {}
    
    def check_retraining_needed(self, symbol: str, accuracy_threshold: float = 0.8) -> bool:
        """Check if agent needs retraining based on recent performance."""
        
        perf = self.get_agent_performance(symbol, days=14)  # 2 weeks
        
        if not perf or perf['total_predictions'] < 5:
            return False  # Not enough data
        
        return perf['avg_accuracy'] < accuracy_threshold
    
    def get_all_symbols_needing_retraining(self, accuracy_threshold: float = 0.8) -> List[str]:
        """Get all symbols that need retraining."""
        
        symbols_needing_retraining = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT symbol FROM stock_agents WHERE is_champion = TRUE
            """)
            
            symbols = [row[0] for row in cursor.fetchall()]
        
        for symbol in symbols:
            if self.check_retraining_needed(symbol, accuracy_threshold):
                symbols_needing_retraining.append(symbol)
        
        return symbols_needing_retraining
    
    def log_training_session(self, symbol: str, session_data: Dict) -> str:
        """Log a training session."""
        
        session_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO training_sessions (
                    session_id, symbol, start_time, population_size,
                    generations_completed, training_data_start, training_data_end,
                    validation_data_start, validation_data_end, hyperparameters, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, symbol, session_data['start_time'],
                session_data['population_size'], session_data.get('generations', 0),
                session_data['training_start'], session_data['training_end'],
                session_data['validation_start'], session_data['validation_end'],
                json.dumps(session_data.get('hyperparameters', {})),
                session_data.get('status', 'RUNNING')
            ))
            
            conn.commit()
        
        return session_id
    
    def update_training_session(self, session_id: str, updates: Dict):
        """Update training session with results."""
        
        with sqlite3.connect(self.db_path) as conn:
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                if key in ['end_time', 'generations_completed', 'best_fitness', 
                          'best_accuracy', 'status', 'champion_agent_id', 'error_message']:
                    set_clauses.append(f"{key} = ?")
                    values.append(value)
            
            if set_clauses:
                values.append(session_id)
                query = f"UPDATE training_sessions SET {', '.join(set_clauses)} WHERE session_id = ?"
                conn.execute(query, values)
                conn.commit()
    
    def add_stock_metadata(self, symbol: str, metadata: Dict):
        """Add or update stock metadata."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO stock_metadata (
                    symbol, company_name, sector, market_cap, avg_volume_30d,
                    options_volume_30d, iv_rank, beta, priority_tier,
                    data_quality_score, is_tradeable
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, metadata.get('company_name', ''),
                metadata.get('sector', ''), metadata.get('market_cap', 0),
                metadata.get('avg_volume_30d', 0), metadata.get('options_volume_30d', 0),
                metadata.get('iv_rank', 0), metadata.get('beta', 1.0),
                metadata.get('priority_tier', 3), metadata.get('data_quality_score', 0.5),
                metadata.get('is_tradeable', True)
            ))
            
            conn.commit()
    
    def get_top_stocks(self, limit: int = 20) -> List[Dict]:
        """Get top priority stocks for training."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM stock_metadata 
                WHERE is_tradeable = TRUE
                ORDER BY priority_tier ASC, market_cap DESC
                LIMIT ?
            """, (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Count agents
            cursor = conn.execute("SELECT COUNT(*) FROM stock_agents")
            stats['total_agents'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM stock_agents WHERE is_champion = TRUE")
            stats['champion_agents'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM agent_history")
            stats['archived_agents'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM agent_performance")
            stats['total_predictions'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(DISTINCT symbol) FROM stock_agents")
            stats['symbols_with_agents'] = cursor.fetchone()[0]
            
            # Recent activity
            cursor = conn.execute("""
                SELECT COUNT(*) FROM agent_performance 
                WHERE prediction_date >= date('now', '-7 days')
            """)
            stats['predictions_last_7_days'] = cursor.fetchone()[0]
            
        return stats
