#!/usr/bin/env python3
"""
Enhanced Output Generator for NEAT Agent Training Results
Generates dual output tables (price metrics and direction metrics) with comprehensive analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class AgentSummary:
    """Summary of agent performance for output tables."""
    symbol: str
    fitness_score: float
    fitness_interpretation: str
    
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
    profit_factor: float
    
    # Accuracy Metrics
    direction_accuracy: float
    price_accuracy: float
    
    # Time Horizon
    lookahead_days: int
    lookahead_weeks: float
    
    # Threshold Check
    meets_threshold: bool
    
    # Horizon Results
    horizon_results: Dict

class EnhancedOutputGenerator:
    """Generate comprehensive output tables for NEAT agent training results."""
    
    def __init__(self):
        self.min_outperformance = 0.50  # 50% minimum outperformance
        
    def generate_dual_output_tables(self, training_results: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate dual output tables: Direction Metrics and Price Metrics."""
        
        # Convert results to AgentSummary objects
        agent_summaries = []
        for result in training_results:
            if result.success and result.performance_metrics:
                metrics = result.performance_metrics
                agent_summaries.append(AgentSummary(
                    symbol=result.symbol,
                    fitness_score=metrics.get('fitness_score', 0),
                    fitness_interpretation=metrics.get('fitness_interpretation', 'Unknown'),
                    total_return=metrics.get('total_return', 0),
                    buy_hold_return=metrics.get('buy_hold_return', 0),
                    excess_return=metrics.get('excess_return', 0),
                    excess_return_pct=metrics.get('excess_return_pct', 0),
                    max_drawdown=metrics.get('max_drawdown', 0),
                    current_drawdown=metrics.get('current_drawdown', 0),
                    volatility=metrics.get('volatility', 0),
                    sharpe_ratio=metrics.get('sharpe_ratio', 0),
                    win_rate=metrics.get('win_rate', 0),
                    total_trades=metrics.get('total_trades', 0),
                    winning_trades=metrics.get('winning_trades', 0),
                    losing_trades=metrics.get('losing_trades', 0),
                    profit_factor=metrics.get('profit_factor', 0),
                    direction_accuracy=metrics.get('direction_accuracy', 0),
                    price_accuracy=metrics.get('price_accuracy', 0),
                    lookahead_days=metrics.get('lookahead_days', 0),
                    lookahead_weeks=metrics.get('lookahead_weeks', 0),
                    meets_threshold=metrics.get('meets_threshold', False),
                    horizon_results=metrics.get('horizon_results', {})
                ))
        
        # Generate Direction Metrics Table
        direction_table = self._generate_direction_metrics_table(agent_summaries)
        
        # Generate Price Metrics Table  
        price_table = self._generate_price_metrics_table(agent_summaries)
        
        return direction_table, price_table
    
    def _generate_direction_metrics_table(self, summaries: List[AgentSummary]) -> pd.DataFrame:
        """Generate Direction Metrics Table (Up/Down Classification Performance)."""
        
        direction_data = []
        
        for summary in summaries:
            # Determine status based on performance threshold
            status = "✅ PASSED" if summary.meets_threshold else "❌ FAILED"
            
            # Calculate additional direction metrics
            avg_trade_return = summary.total_return / summary.total_trades if summary.total_trades > 0 else 0
            
            direction_data.append({
                'Symbol': summary.symbol,
                'Direction Accuracy': f"{summary.direction_accuracy:.1%}",
                'Win Rate': f"{summary.win_rate:.1%}",
                'Total Trades': summary.total_trades,
                'Winning Trades': summary.winning_trades,
                'Losing Trades': summary.losing_trades,
                'Avg Trade Return': f"{avg_trade_return:.2%}",
                'Profit Factor': f"{summary.profit_factor:.2f}",
                'Lookahead Period': f"{summary.lookahead_weeks:.1f}w ({summary.lookahead_days}d)",
                'Fitness Score': f"{summary.fitness_score:.1f}",
                'Fitness Grade': summary.fitness_interpretation,
                'Status': status
            })
        
        # Sort by win rate (most important for options)
        df = pd.DataFrame(direction_data)
        if not df.empty:
            df = df.sort_values('Win Rate', ascending=False)
        
        return df
    
    def _generate_price_metrics_table(self, summaries: List[AgentSummary]) -> pd.DataFrame:
        """Generate Price Metrics Table (Price Target Performance)."""
        
        price_data = []
        
        for summary in summaries:
            # Determine status based on performance threshold
            status = "✅ PASSED" if summary.meets_threshold else "❌ FAILED"
            
            # Calculate risk-adjusted metrics
            risk_adjusted_return = summary.total_return / max(abs(summary.max_drawdown), 0.01)
            
            price_data.append({
                'Symbol': summary.symbol,
                'AI Total Return': f"{summary.total_return:.1%}",
                'Buy & Hold Return': f"{summary.buy_hold_return:.1%}",
                'Excess Return': f"{summary.excess_return:.1%}",
                'Outperformance': f"{summary.excess_return_pct:.1%}",
                'Price Accuracy': f"{summary.price_accuracy:.1%}",
                'Max Drawdown': f"{summary.max_drawdown:.1%}",
                'Current Drawdown': f"{summary.current_drawdown:.1%}",
                'Volatility': f"{summary.volatility:.1%}",
                'Sharpe Ratio': f"{summary.sharpe_ratio:.2f}",
                'Risk-Adj Return': f"{risk_adjusted_return:.2f}",
                'Lookahead Period': f"{summary.lookahead_weeks:.1f}w ({summary.lookahead_days}d)",
                'Status': status
            })
        
        # Sort by excess return percentage (outperformance)
        df = pd.DataFrame(price_data)
        if not df.empty:
            df = df.sort_values('Outperformance', ascending=False)
        
        return df
    
    def generate_horizon_breakdown_table(self, summaries: List[AgentSummary]) -> pd.DataFrame:
        """Generate detailed horizon breakdown table."""
        
        horizon_data = []
        
        for summary in summaries:
            if summary.horizon_results:
                for horizon_days, horizon_result in summary.horizon_results.items():
                    horizon_data.append({
                        'Symbol': summary.symbol,
                        'Horizon (Days)': horizon_days,
                        'Horizon (Weeks)': f"{horizon_result.get('horizon_weeks', 0):.1f}",
                        'Direction Accuracy': f"{horizon_result.get('direction_accuracy', 0):.1%}",
                        'Price MAE': f"{horizon_result.get('price_mae', 0):.3f}",
                        'Price MAPE': f"{horizon_result.get('price_mape', 0):.1%}",
                        'Trades': horizon_result.get('trades', 0),
                        'Win Rate': f"{horizon_result.get('win_rate', 0):.1%}",
                        'Avg Return': f"{horizon_result.get('avg_return', 0):.2%}"
                    })
        
        df = pd.DataFrame(horizon_data)
        if not df.empty:
            df = df.sort_values(['Symbol', 'Horizon (Days)'])
        
        return df
    
    def generate_summary_statistics(self, summaries: List[AgentSummary]) -> Dict:
        """Generate overall summary statistics."""
        
        if not summaries:
            return {}
        
        # Calculate aggregate statistics
        total_agents = len(summaries)
        passed_agents = sum(1 for s in summaries if s.meets_threshold)
        failed_agents = total_agents - passed_agents
        
        avg_outperformance = np.mean([s.excess_return_pct for s in summaries])
        avg_win_rate = np.mean([s.win_rate for s in summaries])
        avg_fitness = np.mean([s.fitness_score for s in summaries])
        
        # Top performers
        top_by_outperformance = sorted(summaries, key=lambda x: x.excess_return_pct, reverse=True)[:5]
        top_by_win_rate = sorted(summaries, key=lambda x: x.win_rate, reverse=True)[:5]
        
        return {
            'total_agents': total_agents,
            'passed_agents': passed_agents,
            'failed_agents': failed_agents,
            'success_rate': passed_agents / total_agents if total_agents > 0 else 0,
            'avg_outperformance': avg_outperformance,
            'avg_win_rate': avg_win_rate,
            'avg_fitness': avg_fitness,
            'top_by_outperformance': [(s.symbol, s.excess_return_pct) for s in top_by_outperformance],
            'top_by_win_rate': [(s.symbol, s.win_rate) for s in top_by_win_rate],
            'min_outperformance_threshold': self.min_outperformance
        }
    
    def print_comprehensive_results(self, training_results: List[Dict]) -> None:
        """Print comprehensive results with dual tables and analysis."""
        
        # Generate dual tables
        direction_table, price_table = self.generate_dual_output_tables(training_results)
        
        # Generate agent summaries for statistics
        agent_summaries = []
        for result in training_results:
            if result.success and result.performance_metrics:
                metrics = result.performance_metrics
                agent_summaries.append(AgentSummary(
                    symbol=result.symbol,
                    fitness_score=metrics.get('fitness_score', 0),
                    fitness_interpretation=metrics.get('fitness_interpretation', 'Unknown'),
                    total_return=metrics.get('total_return', 0),
                    buy_hold_return=metrics.get('buy_hold_return', 0),
                    excess_return=metrics.get('excess_return', 0),
                    excess_return_pct=metrics.get('excess_return_pct', 0),
                    max_drawdown=metrics.get('max_drawdown', 0),
                    current_drawdown=metrics.get('current_drawdown', 0),
                    volatility=metrics.get('volatility', 0),
                    sharpe_ratio=metrics.get('sharpe_ratio', 0),
                    win_rate=metrics.get('win_rate', 0),
                    total_trades=metrics.get('total_trades', 0),
                    winning_trades=metrics.get('winning_trades', 0),
                    losing_trades=metrics.get('losing_trades', 0),
                    profit_factor=metrics.get('profit_factor', 0),
                    direction_accuracy=metrics.get('direction_accuracy', 0),
                    price_accuracy=metrics.get('price_accuracy', 0),
                    lookahead_days=metrics.get('lookahead_days', 0),
                    lookahead_weeks=metrics.get('lookahead_weeks', 0),
                    meets_threshold=metrics.get('meets_threshold', False),
                    horizon_results=metrics.get('horizon_results', {})
                ))
        
        # Generate summary statistics
        summary_stats = self.generate_summary_statistics(agent_summaries)
        
        # Generate horizon breakdown
        horizon_table = self.generate_horizon_breakdown_table(agent_summaries)
        
        # Print results
        print("\n" + "="*100)
        print("🎯 COMPREHENSIVE NEAT AGENT TRAINING RESULTS")
        print("="*100)
        
        # Summary Statistics
        print(f"\n📊 SUMMARY STATISTICS")
        print(f"  Total Agents Trained:     {summary_stats.get('total_agents', 0)}")
        print(f"  Agents Meeting Threshold: {summary_stats.get('passed_agents', 0)} ✅")
        print(f"  Agents Below Threshold:   {summary_stats.get('failed_agents', 0)} ❌")
        print(f"  Success Rate:             {summary_stats.get('success_rate', 0):.1%}")
        print(f"  Avg Outperformance:       {summary_stats.get('avg_outperformance', 0):.1%}")
        print(f"  Avg Win Rate:             {summary_stats.get('avg_win_rate', 0):.1%}")
        print(f"  Avg Fitness Score:        {summary_stats.get('avg_fitness', 0):.1f}")
        print(f"  Required Threshold:       {summary_stats.get('min_outperformance_threshold', 0):.0%}+ outperformance")
        
        # Direction Metrics Table
        print(f"\n🎲 DIRECTION METRICS TABLE (Up/Down Classification)")
        print("="*100)
        if not direction_table.empty:
            print(direction_table.to_string(index=False))
        else:
            print("No direction metrics available.")
        
        # Price Metrics Table
        print(f"\n💰 PRICE METRICS TABLE (Price Target Performance)")
        print("="*100)
        if not price_table.empty:
            print(price_table.to_string(index=False))
        else:
            print("No price metrics available.")
        
        # Horizon Breakdown (if available)
        if not horizon_table.empty:
            print(f"\n⏰ MULTI-HORIZON BREAKDOWN (1-8 Week Analysis)")
            print("="*100)
            print(horizon_table.to_string(index=False))
        
        # Top Performers
        if summary_stats.get('top_by_outperformance'):
            print(f"\n🏆 TOP PERFORMERS BY OUTPERFORMANCE")
            for i, (symbol, perf) in enumerate(summary_stats['top_by_outperformance'], 1):
                print(f"  {i}. {symbol}: {perf:.1%}")
        
        if summary_stats.get('top_by_win_rate'):
            print(f"\n🎯 TOP PERFORMERS BY WIN RATE")
            for i, (symbol, win_rate) in enumerate(summary_stats['top_by_win_rate'], 1):
                print(f"  {i}. {symbol}: {win_rate:.1%}")
        
        print("\n" + "="*100)
        print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*100)
    
    def save_results_to_json(self, training_results: List[Dict], filename: str) -> None:
        """Save comprehensive results to JSON file."""
        
        # Generate all tables and statistics
        direction_table, price_table = self.generate_dual_output_tables(training_results)
        
        # Convert to serializable format
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'direction_metrics': direction_table.to_dict('records') if not direction_table.empty else [],
            'price_metrics': price_table.to_dict('records') if not price_table.empty else [],
            'raw_results': [
                {
                    'symbol': result.symbol,
                    'success': result.success,
                    'performance_metrics': result.performance_metrics,
                    'training_time': result.training_time,
                    'generations_completed': result.generations_completed
                }
                for result in training_results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"📄 Results saved to: {filename}")
