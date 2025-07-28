#!/usr/bin/env python3
"""
SwingTradeAI - Real-time P&L Tracking System
User Acceptance: User sees accurate, real-time performance data
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from .trade_logger import TradeLogger, TradeStatus
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Basic Metrics
    total_pnl: float
    total_return_pct: float
    win_rate: float
    profit_factor: float
    
    # Risk Metrics
    sharpe_ratio: float
    max_drawdown_pct: float
    current_drawdown_pct: float
    volatility: float
    
    # Trade Metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_holding_period_hours: float
    
    # Recent Performance
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    
    # Portfolio Metrics
    current_capital: float
    total_exposure: float
    available_cash: float
    open_positions: int

class PnLTracker:
    """Production-ready P&L tracking with real-time updates"""
    
    def __init__(self, trade_logger: TradeLogger, risk_manager: RiskManager):
        self.trade_logger = trade_logger
        self.risk_manager = risk_manager
        self.performance_cache = {}
        self.last_update = None
        
    async def update_real_time_pnl(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Update real-time P&L for all open positions"""
        try:
            # Update unrealized P&L in trade logger
            self.trade_logger.update_unrealized_pnl(current_prices)
            
            # Get updated open positions
            open_trades = self.trade_logger.get_trade_history(status=TradeStatus.OPEN)
            
            total_unrealized_pnl = 0.0
            position_details = []
            
            if not open_trades.empty:
                for _, trade in open_trades.iterrows():
                    symbol = trade['symbol']
                    current_price = current_prices.get(symbol, trade['entry_price'])
                    
                    # Calculate real-time P&L
                    entry_price = trade['entry_price']
                    quantity = trade['quantity']
                    
                    if trade['action'] in ['buy_to_open', 'buy_to_close']:
                        # Long position
                        price_diff = current_price - entry_price
                    else:
                        # Short position
                        price_diff = entry_price - current_price
                    
                    unrealized_pnl = price_diff * quantity
                    total_unrealized_pnl += unrealized_pnl
                    
                    # Calculate return percentage
                    position_value = entry_price * quantity
                    return_pct = (unrealized_pnl / position_value) * 100 if position_value > 0 else 0
                    
                    position_details.append({
                        'trade_id': trade['trade_id'],
                        'symbol': symbol,
                        'trade_type': trade['trade_type'],
                        'quantity': quantity,
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'return_pct': return_pct,
                        'position_value': position_value,
                        'current_value': current_price * quantity,
                        'entry_timestamp': trade['entry_timestamp']
                    })
            
            # Update capital in risk manager
            realized_pnl = self.trade_logger.get_performance_summary().get('total_pnl', 0)
            new_capital = self.risk_manager.initial_capital + realized_pnl + total_unrealized_pnl
            self.risk_manager.update_capital(new_capital)
            
            result = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_realized_pnl': realized_pnl,
                'total_pnl': realized_pnl + total_unrealized_pnl,
                'current_capital': new_capital,
                'open_positions_count': len(position_details),
                'position_details': position_details
            }
            
            self.last_update = datetime.now(timezone.utc)
            logger.info(f"✅ P&L updated: Total ${result['total_pnl']:,.2f} (Realized: ${realized_pnl:,.2f}, Unrealized: ${total_unrealized_pnl:,.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to update real-time P&L: {str(e)}")
            return {}
    
    def calculate_performance_metrics(self, lookback_days: int = 30) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            # Get trade history
            all_trades = self.trade_logger.get_trade_history(limit=None)
            performance_summary = self.trade_logger.get_performance_summary()
            
            if all_trades.empty:
                return self._empty_metrics()
            
            # Filter trades by lookback period
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            recent_trades = all_trades[
                pd.to_datetime(all_trades['entry_timestamp']) >= cutoff_date
            ]
            
            # Basic metrics
            total_pnl = performance_summary.get('total_pnl', 0) or 0
            initial_capital = self.risk_manager.initial_capital
            total_return_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0
            
            win_rate = performance_summary.get('win_rate', 0) or 0
            profit_factor = performance_summary.get('profit_factor', 0) or 0
            
            # Calculate Sharpe ratio
            closed_trades = all_trades[all_trades['status'] == 'closed']
            if len(closed_trades) > 1:
                returns = closed_trades['total_return_pct'].dropna()
                if len(returns) > 1:
                    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
                else:
                    sharpe_ratio = 0
                volatility = returns.std() if len(returns) > 1 else 0
            else:
                sharpe_ratio = 0
                volatility = 0
            
            # Calculate drawdown
            if not closed_trades.empty:
                closed_trades = closed_trades.sort_values('exit_timestamp')
                cumulative_pnl = closed_trades['realized_pnl'].cumsum()
                running_max = cumulative_pnl.expanding().max()
                drawdown = cumulative_pnl - running_max
                max_drawdown_pct = (drawdown.min() / initial_capital) * 100 if initial_capital > 0 else 0
                current_drawdown_pct = (drawdown.iloc[-1] / initial_capital) * 100 if len(drawdown) > 0 and initial_capital > 0 else 0
            else:
                max_drawdown_pct = 0
                current_drawdown_pct = 0
            
            # Time-based P&L
            now = datetime.now(timezone.utc)
            daily_pnl = self._calculate_period_pnl(all_trades, now - timedelta(days=1))
            weekly_pnl = self._calculate_period_pnl(all_trades, now - timedelta(days=7))
            monthly_pnl = self._calculate_period_pnl(all_trades, now - timedelta(days=30))
            
            # Portfolio metrics
            portfolio_summary = self.risk_manager.get_portfolio_summary()
            
            return PerformanceMetrics(
                total_pnl=total_pnl,
                total_return_pct=total_return_pct,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown_pct=abs(max_drawdown_pct),
                current_drawdown_pct=abs(current_drawdown_pct),
                volatility=volatility,
                total_trades=performance_summary.get('total_trades', 0) or 0,
                winning_trades=performance_summary.get('winning_trades', 0) or 0,
                losing_trades=performance_summary.get('losing_trades', 0) or 0,
                avg_win=performance_summary.get('max_win', 0) or 0,
                avg_loss=abs(performance_summary.get('max_loss', 0) or 0),
                avg_holding_period_hours=performance_summary.get('avg_holding_hours', 0) or 0,
                daily_pnl=daily_pnl,
                weekly_pnl=weekly_pnl,
                monthly_pnl=monthly_pnl,
                current_capital=portfolio_summary.get('current_capital', 0) or 0,
                total_exposure=portfolio_summary.get('total_exposure', 0) or 0,
                available_cash=portfolio_summary.get('available_buying_power', 0) or 0,
                open_positions=portfolio_summary.get('open_positions', 0) or 0
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate performance metrics: {str(e)}")
            return self._empty_metrics()
    
    def _calculate_period_pnl(self, trades_df: pd.DataFrame, start_date: datetime) -> float:
        """Calculate P&L for a specific time period"""
        try:
            if trades_df.empty:
                return 0.0
            
            # Filter trades closed in the period
            period_trades = trades_df[
                (trades_df['status'] == 'closed') &
                (pd.to_datetime(trades_df['exit_timestamp']) >= start_date)
            ]
            
            return period_trades['realized_pnl'].sum() if not period_trades.empty else 0.0
            
        except Exception:
            return 0.0
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics"""
        return PerformanceMetrics(
            total_pnl=0.0, total_return_pct=0.0, win_rate=0.0, profit_factor=0.0,
            sharpe_ratio=0.0, max_drawdown_pct=0.0, current_drawdown_pct=0.0, volatility=0.0,
            total_trades=0, winning_trades=0, losing_trades=0, avg_win=0.0, avg_loss=0.0,
            avg_holding_period_hours=0.0, daily_pnl=0.0, weekly_pnl=0.0, monthly_pnl=0.0,
            current_capital=0.0, total_exposure=0.0, available_cash=0.0, open_positions=0
        )
    
    def generate_performance_report(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            metrics = self.calculate_performance_metrics(lookback_days)
            
            # Performance grades
            def grade_performance(value: float, thresholds: List[float]) -> str:
                """Grade performance based on thresholds [A, B, C, D]"""
                if value >= thresholds[0]:
                    return "A"
                elif value >= thresholds[1]:
                    return "B"
                elif value >= thresholds[2]:
                    return "C"
                else:
                    return "D"
            
            # Define grading thresholds
            return_grade = grade_performance(metrics.total_return_pct, [20, 10, 5, 0])
            win_rate_grade = grade_performance(metrics.win_rate, [70, 60, 50, 40])
            sharpe_grade = grade_performance(metrics.sharpe_ratio, [2.0, 1.5, 1.0, 0.5])
            drawdown_grade = grade_performance(-metrics.max_drawdown_pct, [-5, -10, -15, -20])
            
            # Risk assessment
            risk_level = "LOW"
            if metrics.max_drawdown_pct > 15 or metrics.volatility > 30:
                risk_level = "HIGH"
            elif metrics.max_drawdown_pct > 10 or metrics.volatility > 20:
                risk_level = "MEDIUM"
            
            # Trading frequency analysis
            if metrics.total_trades > 0:
                trades_per_day = metrics.total_trades / lookback_days
                if trades_per_day > 2:
                    frequency = "HIGH"
                elif trades_per_day > 0.5:
                    frequency = "MEDIUM"
                else:
                    frequency = "LOW"
            else:
                frequency = "NONE"
            
            report = {
                'report_date': datetime.now(timezone.utc).isoformat(),
                'lookback_days': lookback_days,
                'performance_summary': {
                    'total_pnl': metrics.total_pnl,
                    'total_return_pct': metrics.total_return_pct,
                    'return_grade': return_grade,
                    'current_capital': metrics.current_capital,
                    'available_cash': metrics.available_cash
                },
                'trading_metrics': {
                    'total_trades': metrics.total_trades,
                    'win_rate': metrics.win_rate,
                    'win_rate_grade': win_rate_grade,
                    'profit_factor': metrics.profit_factor,
                    'avg_win': metrics.avg_win,
                    'avg_loss': metrics.avg_loss,
                    'trading_frequency': frequency
                },
                'risk_metrics': {
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'sharpe_grade': sharpe_grade,
                    'max_drawdown_pct': metrics.max_drawdown_pct,
                    'drawdown_grade': drawdown_grade,
                    'current_drawdown_pct': metrics.current_drawdown_pct,
                    'volatility': metrics.volatility,
                    'risk_level': risk_level
                },
                'recent_performance': {
                    'daily_pnl': metrics.daily_pnl,
                    'weekly_pnl': metrics.weekly_pnl,
                    'monthly_pnl': metrics.monthly_pnl
                },
                'portfolio_status': {
                    'open_positions': metrics.open_positions,
                    'total_exposure': metrics.total_exposure,
                    'exposure_pct': (metrics.total_exposure / metrics.current_capital) * 100 if metrics.current_capital > 0 else 0
                },
                'recommendations': self._generate_recommendations(metrics)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate performance report: {str(e)}")
            return {}
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate actionable recommendations based on performance"""
        recommendations = []
        
        # Return-based recommendations
        if metrics.total_return_pct < 0:
            recommendations.append("📉 Consider reviewing strategy - negative returns detected")
        elif metrics.total_return_pct > 50:
            recommendations.append("🎉 Excellent returns! Consider taking some profits")
        
        # Win rate recommendations
        if metrics.win_rate < 40:
            recommendations.append("🎯 Low win rate - review entry criteria and market timing")
        elif metrics.win_rate > 80:
            recommendations.append("⚠️ Very high win rate - ensure you're not avoiding necessary losses")
        
        # Risk recommendations
        if metrics.max_drawdown_pct > 20:
            recommendations.append("🛑 High drawdown - consider reducing position sizes")
        if metrics.current_drawdown_pct > 10:
            recommendations.append("📊 Currently in drawdown - consider defensive positioning")
        
        # Sharpe ratio recommendations
        if metrics.sharpe_ratio < 1.0:
            recommendations.append("📈 Low risk-adjusted returns - optimize risk management")
        
        # Trading frequency recommendations
        if metrics.total_trades > 100:
            recommendations.append("⚡ High trading frequency - monitor transaction costs")
        elif metrics.total_trades < 5:
            recommendations.append("🐌 Low trading activity - consider more opportunities")
        
        # Portfolio recommendations
        exposure_pct = (metrics.total_exposure / metrics.current_capital) * 100 if metrics.current_capital > 0 else 0
        if exposure_pct > 80:
            recommendations.append("⚠️ High portfolio exposure - consider reducing risk")
        elif exposure_pct < 20:
            recommendations.append("💡 Low portfolio utilization - consider increasing allocation")
        
        return recommendations
    
    def export_performance_data(self, filepath: str, format: str = 'json'):
        """Export performance data for external analysis"""
        try:
            # Get comprehensive data
            trades_df = self.trade_logger.get_trade_history(limit=None)
            metrics = self.calculate_performance_metrics()
            report = self.generate_performance_report()
            
            export_data = {
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'performance_metrics': metrics.__dict__,
                'performance_report': report,
                'trade_count': len(trades_df),
                'summary': {
                    'total_pnl': metrics.total_pnl,
                    'total_return_pct': metrics.total_return_pct,
                    'win_rate': metrics.win_rate,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown_pct': metrics.max_drawdown_pct
                }
            }
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"✅ Performance data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export performance data: {str(e)}")
            raise
