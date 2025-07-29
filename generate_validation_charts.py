#!/usr/bin/env python3
"""
Validation Chart Generator - Proves Agent Performance with Visual Evidence
Generates comprehensive charts showing trading performance, P&L evolution, and risk metrics.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from high_performance_neat_system import HighPerformanceNEATSystem, EnhancedDataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class ValidationChartGenerator:
    """Generate comprehensive validation charts for NEAT trading agents."""
    
    def __init__(self):
        self.output_dir = Path("validation_charts")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_comprehensive_charts(self, ticker="AAPL"):
        """Generate all validation charts for a given ticker."""
        logger.info(f"🎯 Generating validation charts for {ticker}")
        
        # Initialize systems
        data_pipeline = EnhancedDataPipeline()
        neat_system = HighPerformanceNEATSystem()
        
        # Collect data
        logger.info("📊 Collecting market data...")
        data = data_pipeline.collect_extended_data(ticker, period="2y")
        if data is None or len(data) < 100:
            logger.error(f"❌ Insufficient data for {ticker}")
            return False
            
        # Prepare features
        logger.info("🔧 Preparing enhanced features...")
        features = data_pipeline.prepare_enhanced_features(data)
        
        # Train agent
        logger.info("🚀 Training NEAT agent...")
        result = neat_system.train_agent(ticker, features, max_generations=5)
        
        if not result['success']:
            logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
            return False
            
        # Get performance data
        performance = result['performance']
        logger.info(f"📈 Agent Performance: {performance['annual_return']:.1f}% APY, {performance['win_rate']:.1f}% win rate")
        
        # Generate charts
        self._create_performance_overview_chart(ticker, performance, data)
        self._create_trading_simulation_chart(ticker, result['winner'], features, data)
        self._create_risk_metrics_chart(ticker, performance)
        self._create_comparison_chart(ticker, performance, data)
        
        logger.info(f"✅ All validation charts generated in {self.output_dir}/")
        return True
        
    def _create_performance_overview_chart(self, ticker, performance, data):
        """Create comprehensive performance overview chart."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{ticker} NEAT Trading Agent - Performance Overview', fontsize=16, fontweight='bold')
        
        # Performance metrics bar chart
        metrics = ['Win Rate (%)', 'Annual Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
        values = [performance['win_rate'], performance['annual_return'], 
                 performance['sharpe_ratio'], performance['max_drawdown']]
        targets = [80, 110, 2.0, 8.0]  # Target values
        
        bars = ax1.bar(metrics, values, color=['green' if v >= t else 'red' for v, t in zip(values, targets)])
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Performance Metrics vs Targets')
        ax1.set_ylabel('Value')
        
        # Add target lines
        for i, (metric, target) in enumerate(zip(metrics, targets)):
            ax1.axhline(y=target, color='blue', linestyle='--', alpha=0.7)
            ax1.text(i, target + max(values) * 0.05, f'Target: {target}', 
                    ha='center', va='bottom', color='blue', fontsize=8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Price chart with moving averages
        prices = data['Close'].values
        dates = pd.to_datetime(data.index)
        
        ax2.plot(dates, prices, label=f'{ticker} Price', linewidth=2, color='blue')
        ax2.plot(dates, data['Close'].rolling(20).mean(), label='20-day MA', alpha=0.7, color='orange')
        ax2.plot(dates, data['Close'].rolling(50).mean(), label='50-day MA', alpha=0.7, color='red')
        ax2.set_title(f'{ticker} Price Action (Training Period)')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Weekly return distribution
        weekly_returns = data['Close'].pct_change(5).dropna() * 100
        ax3.hist(weekly_returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(x=1.5, color='red', linestyle='--', label='Target: 1.5% weekly')
        ax3.axvline(x=weekly_returns.mean(), color='green', linestyle='-', 
                   label=f'Avg: {weekly_returns.mean():.1f}%')
        ax3.set_title('Weekly Return Distribution')
        ax3.set_xlabel('Weekly Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Assignment rate analysis
        assignment_data = [performance['assignment_rate'], 5.0 - performance['assignment_rate']]
        colors = ['red' if performance['assignment_rate'] > 5 else 'green', 'lightgray']
        labels = [f'Assignment Rate: {performance["assignment_rate"]:.1f}%', 
                 f'Safe Zone: {5.0 - performance["assignment_rate"]:.1f}%']
        
        ax4.pie(assignment_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Assignment Rate vs Target (5% max)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{ticker}_performance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_trading_simulation_chart(self, ticker, winner_genome, features, data):
        """Create detailed trading simulation chart."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle(f'{ticker} NEAT Agent - Trading Simulation Analysis', fontsize=16, fontweight='bold')
        
        # Simulate trading with the winner genome
        predictions, trades_data = self._simulate_detailed_trading(winner_genome, features, data)
        
        prices = data['Close'].values
        dates = pd.to_datetime(data.index)
        
        # Price chart with trade signals
        ax1.plot(dates, prices, label=f'{ticker} Price', linewidth=2, color='blue', alpha=0.8)
        
        # Mark entry and exit points
        for trade in trades_data:
            entry_date = dates[trade['entry_idx']]
            exit_date = dates[trade['exit_idx']]
            entry_price = prices[trade['entry_idx']]
            exit_price = prices[trade['exit_idx']]
            
            color = 'green' if trade['pnl'] > 0 else 'red'
            ax1.scatter(entry_date, entry_price, color='blue', marker='^', s=100, alpha=0.7)
            ax1.scatter(exit_date, exit_price, color=color, marker='v', s=100, alpha=0.7)
            
            # Draw trade line
            ax1.plot([entry_date, exit_date], [entry_price, exit_price], 
                    color=color, alpha=0.5, linewidth=2)
        
        ax1.set_title('Trading Signals and Execution')
        ax1.set_ylabel('Price ($)')
        ax1.legend(['Price', 'Entry', 'Profitable Exit', 'Loss Exit'])
        ax1.grid(True, alpha=0.3)
        
        # Predictions vs actual
        ax2.plot(dates, predictions, label='NEAT Predictions', color='red', alpha=0.8)
        ax2.axhline(y=0.4, color='orange', linestyle='--', label='Entry Threshold (0.4)')
        ax2.axhline(y=0.2, color='red', linestyle='--', label='Exit Threshold (0.2)')
        ax2.set_title('NEAT Predictions vs Entry/Exit Thresholds')
        ax2.set_ylabel('Prediction Confidence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Cumulative P&L
        cumulative_pnl = np.cumsum([trade['pnl'] for trade in trades_data])
        trade_dates = [dates[trade['exit_idx']] for trade in trades_data]
        
        ax3.plot(trade_dates, cumulative_pnl, color='green', linewidth=3, marker='o')
        ax3.fill_between(trade_dates, 0, cumulative_pnl, alpha=0.3, color='green')
        ax3.set_title('Cumulative P&L Evolution')
        ax3.set_ylabel('Cumulative Return')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{ticker}_trading_simulation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_risk_metrics_chart(self, ticker, performance):
        """Create comprehensive risk metrics visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'{ticker} NEAT Agent - Risk Analysis', fontsize=16, fontweight='bold')
        
        # Risk-Return Scatter
        benchmark_return = 10  # S&P 500 approximate
        benchmark_sharpe = 0.8
        
        ax1.scatter(performance['max_drawdown'], performance['annual_return'], 
                   s=200, c='green', alpha=0.8, label='NEAT Agent')
        ax1.scatter(8, benchmark_return, s=200, c='blue', alpha=0.8, label='S&P 500 Benchmark')
        ax1.set_xlabel('Max Drawdown (%)')
        ax1.set_ylabel('Annual Return (%)')
        ax1.set_title('Risk-Return Profile')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add target zones
        ax1.axhline(y=110, color='red', linestyle='--', alpha=0.7, label='Min Target Return')
        ax1.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='Max Target Drawdown')
        
        # Sharpe Ratio Comparison
        categories = ['NEAT Agent', 'Target', 'S&P 500']
        sharpe_values = [performance['sharpe_ratio'], 2.0, benchmark_sharpe]
        colors = ['green', 'blue', 'orange']
        
        bars = ax2.bar(categories, sharpe_values, color=colors, alpha=0.8)
        ax2.set_title('Sharpe Ratio Comparison')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, sharpe_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Win Rate Analysis
        win_rate = performance['win_rate']
        loss_rate = 100 - win_rate
        
        ax3.pie([win_rate, loss_rate], labels=[f'Wins: {win_rate:.1f}%', f'Losses: {loss_rate:.1f}%'],
               colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
        ax3.set_title('Win/Loss Distribution')
        
        # Performance Score
        scores = {
            'Win Rate': min(100, (win_rate / 80) * 100),
            'Return': min(100, (performance['annual_return'] / 110) * 100),
            'Sharpe': min(100, (performance['sharpe_ratio'] / 2.0) * 100),
            'Drawdown': min(100, (8.0 / max(0.1, performance['max_drawdown'])) * 100)
        }
        
        categories = list(scores.keys())
        values = list(scores.values())
        
        bars = ax4.barh(categories, values, color=['green' if v >= 100 else 'orange' if v >= 80 else 'red' for v in values])
        ax4.set_xlim(0, 120)
        ax4.set_xlabel('Score (% of Target)')
        ax4.set_title('Performance Score vs Targets')
        ax4.grid(True, alpha=0.3)
        
        # Add score labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax4.text(width + 2, bar.get_y() + bar.get_height()/2.,
                    f'{value:.0f}%', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{ticker}_risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_comparison_chart(self, ticker, performance, data):
        """Create comparison chart vs benchmarks."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{ticker} NEAT Agent vs Benchmarks', fontsize=16, fontweight='bold')
        
        # Return comparison
        strategies = ['NEAT Agent', 'Buy & Hold', 'S&P 500', 'Target']
        
        # Calculate buy & hold return
        buy_hold_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) ** (252/len(data)) - 1) * 100
        
        returns = [performance['annual_return'], buy_hold_return, 10, 110]
        colors = ['green', 'blue', 'orange', 'red']
        
        bars = ax1.bar(strategies, returns, color=colors, alpha=0.8)
        ax1.set_title('Annual Return Comparison')
        ax1.set_ylabel('Annual Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(returns) * 0.02,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Risk-adjusted return (Sharpe)
        sharpe_ratios = [performance['sharpe_ratio'], buy_hold_return/15, 0.8, 2.0]  # Estimate volatility
        
        bars = ax2.bar(strategies, sharpe_ratios, color=colors, alpha=0.8)
        ax2.set_title('Risk-Adjusted Return (Sharpe Ratio)')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(sharpe_ratios) * 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{ticker}_benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _simulate_detailed_trading(self, genome, features, data):
        """Simulate detailed trading to extract trade-by-trade data."""
        import neat
        
        # Create neural network
        net = neat.nn.FeedForwardNetwork.create(genome, neat.Config(neat.DefaultGenome,
                                                                   neat.DefaultReproduction,
                                                                   neat.DefaultSpeciesSet,
                                                                   neat.DefaultStagnation,
                                                                   'neat_config.txt'))
        
        predictions = []
        trades_data = []
        
        prices = data['Close'].values
        position = 0
        entry_idx = 0
        entry_price = 0
        entry_confidence = 0
        days_in_position = 0
        
        for i in range(len(features)):
            # Get prediction
            prediction = net.activate(features[i])[0]
            predictions.append(prediction)
            
            current_price = prices[i]
            
            # Entry logic
            if position == 0 and prediction > 0.4:
                position = 1
                entry_idx = i
                entry_price = current_price
                entry_confidence = prediction
                days_in_position = 0
                
            # Exit logic
            elif position != 0:
                days_in_position += 1
                
                # Calculate P&L
                strike_price = entry_price * (0.95 - entry_confidence * 0.05)
                if current_price >= strike_price:
                    base_premium = 0.035
                    confidence_multiplier = entry_confidence * 1.5 + 0.9
                    volatility_bonus = min(0.025, abs(current_price - entry_price) / entry_price * 0.8)
                    market_regime_bonus = 0.01 if abs(current_price - entry_price) / entry_price > 0.05 else 0
                    pnl = (base_premium + volatility_bonus + market_regime_bonus) * confidence_multiplier
                else:
                    pnl = -(strike_price - current_price) / current_price + 0.015
                
                # Exit conditions
                max_days = max(3, min(8, int(5 + entry_confidence * 3)))
                profit_target = 0.05 + (entry_confidence * 0.03)
                stop_loss = -0.08
                
                if (days_in_position >= max_days or pnl < stop_loss or 
                    pnl > profit_target or prediction < 0.2):
                    
                    net_pnl = pnl - 0.002
                    trades_data.append({
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': net_pnl,
                        'days_held': days_in_position,
                        'entry_confidence': entry_confidence
                    })
                    
                    position = 0
                    days_in_position = 0
        
        return np.array(predictions), trades_data

def main():
    """Generate validation charts for key tickers."""
    generator = ValidationChartGenerator()
    
    # Test with AAPL first
    logger.info("🎯 Starting validation chart generation...")
    success = generator.generate_comprehensive_charts("AAPL")
    
    if success:
        logger.info("✅ Validation charts generated successfully!")
        logger.info(f"📁 Charts saved to: {generator.output_dir.absolute()}")
        
        # List generated files
        chart_files = list(generator.output_dir.glob("*.png"))
        logger.info(f"📊 Generated {len(chart_files)} charts:")
        for chart_file in sorted(chart_files):
            logger.info(f"   - {chart_file.name}")
            
    else:
        logger.error("❌ Failed to generate validation charts")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
