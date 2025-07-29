#!/usr/bin/env python3
"""
Comprehensive Agent Evaluation System
Implements proper backtesting, forward testing, and dual output tables for NEAT agents.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import neat
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

@dataclass
class TradingMetrics:
    """Comprehensive trading performance metrics."""
    # Core Performance
    total_return: float
    buy_hold_return: float
    excess_return: float
    excess_return_pct: float
    
    # Risk Metrics
    max_drawdown: float
    current_drawdown: float
    volatility: float
    sharpe_ratio: float
    
    # Trading Metrics
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Time Horizon Metrics
    lookahead_days: int
    lookahead_weeks: float
    
    # Accuracy Metrics
    direction_accuracy: float
    price_accuracy: float
    
    # Fitness Score
    fitness_score: float
    fitness_interpretation: str

@dataclass
class HorizonResults:
    """Results for specific time horizon."""
    horizon_days: int
    horizon_weeks: float
    direction_accuracy: float
    price_mae: float  # Mean Absolute Error
    price_mape: float  # Mean Absolute Percentage Error
    trades: int
    win_rate: float
    avg_return: float

class ComprehensiveAgentEvaluator:
    """Comprehensive evaluation system for NEAT agents with proper backtesting."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Multi-horizon setup (1-8 weeks)
        self.horizons = [5, 10, 15, 20, 25, 30, 35, 40]  # 1-8 weeks
        self.horizon_weights = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]  # Longer horizons weighted more
        
        # Performance thresholds
        self.min_outperformance = 0.50  # 50% minimum outperformance vs buy-and-hold
        self.target_win_rate = 0.70  # 70% target win rate for options
        self.max_acceptable_drawdown = 0.15  # 15% max drawdown
        
    def evaluate_agent_comprehensive(self, genome, config, training_data: Dict) -> Tuple[TradingMetrics, Dict[int, HorizonResults]]:
        """Comprehensive evaluation with walk-forward backtesting and forward testing."""
        
        # Create neural network
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Get data
        raw_data = training_data['raw_data']
        features = training_data['test']['features']
        
        # Calculate buy-and-hold benchmark
        buy_hold_return = self._calculate_buy_hold_return(raw_data)
        
        # Walk-forward backtesting
        backtest_results = self._walk_forward_backtest(net, features, raw_data)
        
        # Forward testing on most recent data
        forward_results = self._forward_test(net, features, raw_data)
        
        # Multi-horizon analysis
        horizon_results = self._analyze_multi_horizon_performance(net, features, raw_data)
        
        # Calculate comprehensive metrics
        trading_metrics = self._calculate_comprehensive_metrics(
            backtest_results, forward_results, buy_hold_return, horizon_results
        )
        
        return trading_metrics, horizon_results
    
    def _calculate_buy_hold_return(self, data: pd.DataFrame) -> float:
        """Calculate buy-and-hold return for benchmark."""
        if len(data) < 2:
            return 0.0
        
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        return (end_price / start_price) - 1.0
    
    def _walk_forward_backtest(self, net, features: pd.DataFrame, raw_data: pd.DataFrame) -> Dict:
        """Walk-forward backtesting with expanding window."""
        results = {
            'returns': [],
            'trades': [],
            'positions': [],
            'drawdowns': []
        }
        
        # Use expanding window for walk-forward
        min_train_size = 252  # 1 year minimum
        
        for i in range(min_train_size, len(features) - max(self.horizons)):
            # Current features and future prices
            current_features = features.iloc[i].values
            
            try:
                # Get agent prediction
                output = net.activate(current_features)
                
                # Multi-horizon predictions
                direction_pred = np.argmax(output[:5]) - 2  # -2 to +2 scale
                price_pred = output[5] if len(output) > 5 else 0  # Price change prediction
                
                # Calculate actual returns for all horizons
                horizon_returns = []
                for horizon in self.horizons:
                    if i + horizon < len(raw_data):
                        current_price = raw_data['Close'].iloc[i]
                        future_price = raw_data['Close'].iloc[i + horizon]
                        horizon_return = (future_price / current_price) - 1
                        horizon_returns.append(horizon_return)
                
                if horizon_returns:
                    # Weighted average of horizon returns
                    weights = self.horizon_weights[:len(horizon_returns)]
                    actual_return = np.average(horizon_returns, weights=weights)
                    
                    # Trading decision based on prediction
                    if abs(direction_pred) >= 1:  # Only trade if confident
                        position_size = min(abs(direction_pred) / 2.0, 1.0)  # Scale position by confidence
                        trade_return = direction_pred * actual_return * position_size
                        
                        results['returns'].append(trade_return)
                        results['trades'].append({
                            'date': raw_data.index[i],
                            'direction': direction_pred,
                            'predicted_return': price_pred,
                            'actual_return': actual_return,
                            'trade_return': trade_return,
                            'position_size': position_size
                        })
                        results['positions'].append(direction_pred * position_size)
                    else:
                        results['returns'].append(0)
                        results['positions'].append(0)
                
            except Exception as e:
                self.logger.warning(f"Backtest error at step {i}: {e}")
                results['returns'].append(0)
                results['positions'].append(0)
        
        # Calculate drawdowns
        if results['returns']:
            cumulative_returns = np.cumsum(results['returns'])
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / (running_max + 1e-8)
            results['drawdowns'] = drawdowns.tolist()
        
        return results
    
    def _forward_test(self, net, features: pd.DataFrame, raw_data: pd.DataFrame) -> Dict:
        """Forward testing on most recent unseen data."""
        # Use last 20% of data for forward testing
        forward_start = int(len(features) * 0.8)
        forward_features = features.iloc[forward_start:]
        forward_data = raw_data.iloc[forward_start:]
        
        results = {
            'returns': [],
            'accuracy': [],
            'trades': 0,
            'wins': 0
        }
        
        for i in range(len(forward_features) - max(self.horizons)):
            try:
                current_features = forward_features.iloc[i].values
                output = net.activate(current_features)
                
                direction_pred = np.argmax(output[:5]) - 2
                
                # Check accuracy across horizons
                horizon_accuracies = []
                for horizon in self.horizons:
                    if i + horizon < len(forward_data):
                        current_price = forward_data['Close'].iloc[i]
                        future_price = forward_data['Close'].iloc[i + horizon]
                        actual_direction = 1 if future_price > current_price else -1
                        
                        if direction_pred != 0:
                            accuracy = 1 if np.sign(direction_pred) == np.sign(actual_direction) else 0
                            horizon_accuracies.append(accuracy)
                
                if horizon_accuracies:
                    avg_accuracy = np.mean(horizon_accuracies)
                    results['accuracy'].append(avg_accuracy)
                    
                    if direction_pred != 0:
                        results['trades'] += 1
                        if avg_accuracy > 0.5:
                            results['wins'] += 1
                
            except Exception as e:
                self.logger.warning(f"Forward test error at step {i}: {e}")
        
        return results
    
    def _analyze_multi_horizon_performance(self, net, features: pd.DataFrame, raw_data: pd.DataFrame) -> Dict[int, HorizonResults]:
        """Analyze performance across different time horizons."""
        horizon_results = {}
        
        for horizon in self.horizons:
            direction_correct = 0
            price_errors = []
            trades = 0
            wins = 0
            returns = []
            
            for i in range(len(features) - horizon):
                try:
                    current_features = features.iloc[i].values
                    output = net.activate(current_features)
                    
                    direction_pred = np.argmax(output[:5]) - 2
                    price_pred = output[5] if len(output) > 5 else 0
                    
                    # Actual values
                    current_price = raw_data['Close'].iloc[i]
                    future_price = raw_data['Close'].iloc[i + horizon]
                    actual_return = (future_price / current_price) - 1
                    actual_direction = 1 if actual_return > 0 else -1
                    
                    # Direction accuracy
                    if direction_pred != 0:
                        if np.sign(direction_pred) == np.sign(actual_direction):
                            direction_correct += 1
                        trades += 1
                        
                        # Trading simulation
                        trade_return = direction_pred * actual_return * 0.5  # 50% position size
                        returns.append(trade_return)
                        
                        if trade_return > 0:
                            wins += 1
                    
                    # Price accuracy
                    if price_pred != 0:
                        price_error = abs(price_pred - actual_return)
                        price_errors.append(price_error)
                
                except Exception:
                    continue
            
            # Calculate horizon metrics
            direction_accuracy = direction_correct / trades if trades > 0 else 0
            price_mae = np.mean(price_errors) if price_errors else 0
            price_mape = np.mean([abs(e) for e in price_errors]) if price_errors else 0
            win_rate = wins / trades if trades > 0 else 0
            avg_return = np.mean(returns) if returns else 0
            
            horizon_results[horizon] = HorizonResults(
                horizon_days=horizon,
                horizon_weeks=horizon / 5.0,
                direction_accuracy=direction_accuracy,
                price_mae=price_mae,
                price_mape=price_mape,
                trades=trades,
                win_rate=win_rate,
                avg_return=avg_return
            )
        
        return horizon_results
    
    def _calculate_comprehensive_metrics(self, backtest_results: Dict, forward_results: Dict, 
                                       buy_hold_return: float, horizon_results: Dict) -> TradingMetrics:
        """Calculate comprehensive trading metrics."""
        
        # Core performance
        total_return = sum(backtest_results['returns']) if backtest_results['returns'] else 0
        excess_return = total_return - buy_hold_return
        excess_return_pct = (excess_return / abs(buy_hold_return)) if buy_hold_return != 0 else 0
        
        # Risk metrics
        max_drawdown = min(backtest_results['drawdowns']) if backtest_results['drawdowns'] else 0
        current_drawdown = backtest_results['drawdowns'][-1] if backtest_results['drawdowns'] else 0
        volatility = np.std(backtest_results['returns']) if backtest_results['returns'] else 0
        sharpe_ratio = (total_return / volatility) if volatility > 0 else 0
        
        # Trading metrics
        trades = [t for t in backtest_results['trades'] if t['trade_return'] != 0]
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['trade_return'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [t['trade_return'] for t in trades if t['trade_return'] > 0]
        losses = [t['trade_return'] for t in trades if t['trade_return'] < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else 0
        
        # Multi-horizon averages
        avg_direction_accuracy = np.mean([h.direction_accuracy for h in horizon_results.values()])
        avg_price_accuracy = 1 - np.mean([h.price_mae for h in horizon_results.values()])  # Convert MAE to accuracy
        
        # Weighted average lookahead
        total_trades_by_horizon = sum([h.trades for h in horizon_results.values()])
        if total_trades_by_horizon > 0:
            weighted_lookahead = sum([h.horizon_days * h.trades for h in horizon_results.values()]) / total_trades_by_horizon
        else:
            weighted_lookahead = np.mean(self.horizons)
        
        # Fitness calculation (interpretable)
        fitness_score = (
            win_rate * 100 +  # Win rate is most important (0-100)
            excess_return_pct * 50 +  # Excess return vs buy-hold (scaled)
            avg_direction_accuracy * 25 +  # Direction accuracy bonus
            max(0, -max_drawdown * 100)  # Penalty for drawdown
        )
        
        # Fitness interpretation
        if fitness_score >= 150:
            fitness_interpretation = "Excellent (>150)"
        elif fitness_score >= 100:
            fitness_interpretation = "Good (100-150)"
        elif fitness_score >= 50:
            fitness_interpretation = "Fair (50-100)"
        else:
            fitness_interpretation = "Poor (<50)"
        
        return TradingMetrics(
            total_return=total_return,
            buy_hold_return=buy_hold_return,
            excess_return=excess_return,
            excess_return_pct=excess_return_pct,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            lookahead_days=int(weighted_lookahead),
            lookahead_weeks=weighted_lookahead / 5.0,
            direction_accuracy=avg_direction_accuracy,
            price_accuracy=avg_price_accuracy,
            fitness_score=fitness_score,
            fitness_interpretation=fitness_interpretation
        )
    
    def meets_performance_threshold(self, metrics: TradingMetrics) -> bool:
        """Check if agent meets minimum performance requirements."""
        return (
            metrics.excess_return_pct >= self.min_outperformance and
            metrics.win_rate >= 0.6 and  # Minimum 60% win rate
            metrics.max_drawdown >= -self.max_acceptable_drawdown
        )
    
    def generate_performance_report(self, symbol: str, metrics: TradingMetrics, 
                                  horizon_results: Dict[int, HorizonResults]) -> str:
        """Generate comprehensive performance report."""
        
        report = f"""
🎯 COMPREHENSIVE AGENT PERFORMANCE REPORT - {symbol}
{'='*60}

📊 CORE PERFORMANCE METRICS
  Total Return:           {metrics.total_return:>8.1%}
  Buy & Hold Return:      {metrics.buy_hold_return:>8.1%}
  Excess Return:          {metrics.excess_return:>8.1%} ({metrics.excess_return_pct:+.1%})
  
🎲 RISK METRICS
  Max Drawdown:           {metrics.max_drawdown:>8.1%}
  Current Drawdown:       {metrics.current_drawdown:>8.1%}
  Volatility:             {metrics.volatility:>8.1%}
  Sharpe Ratio:           {metrics.sharpe_ratio:>8.2f}

🏆 TRADING PERFORMANCE
  Win Rate:               {metrics.win_rate:>8.1%} ⭐
  Total Trades:           {metrics.total_trades:>8d}
  Winning Trades:         {metrics.winning_trades:>8d}
  Losing Trades:          {metrics.losing_trades:>8d}
  Average Win:            {metrics.avg_win:>8.1%}
  Average Loss:           {metrics.avg_loss:>8.1%}
  Profit Factor:          {metrics.profit_factor:>8.2f}

⏰ TIME HORIZON ANALYSIS
  Avg Lookahead:          {metrics.lookahead_days:>8d} days ({metrics.lookahead_weeks:.1f} weeks)
  Direction Accuracy:     {metrics.direction_accuracy:>8.1%}
  Price Accuracy:         {metrics.price_accuracy:>8.1%}

🔥 FITNESS SCORE
  Score:                  {metrics.fitness_score:>8.1f}
  Interpretation:         {metrics.fitness_interpretation}
  
✅ PERFORMANCE THRESHOLD: {'PASSED' if self.meets_performance_threshold(metrics) else 'FAILED'}
   (Requires: 50%+ excess return, 60%+ win rate, <15% max drawdown)

📈 HORIZON BREAKDOWN:
"""
        
        for horizon, result in horizon_results.items():
            report += f"  {result.horizon_weeks:>4.1f}w ({horizon:>2d}d): "
            report += f"Dir={result.direction_accuracy:>5.1%} "
            report += f"Win={result.win_rate:>5.1%} "
            report += f"Trades={result.trades:>3d} "
            report += f"Ret={result.avg_return:>6.1%}\n"
        
        return report
