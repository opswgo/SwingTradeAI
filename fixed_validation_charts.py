#!/usr/bin/env python3
"""
FIXED VALIDATION CHARTS - Uses ACTUAL NEAT agent data, NO SIMULATION
Captures real NEAT predictions and generates proper validation charts.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from high_performance_neat_system import HighPerformanceNEATSystem
import neat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def capture_real_neat_data():
    """Capture REAL NEAT agent predictions and trading data."""
    
    logger.info("🎯 Capturing REAL NEAT Agent Data (NO SIMULATION)")
    
    # Initialize REAL NEAT system
    neat_system = HighPerformanceNEATSystem()
    
    try:
        # Get real market data for AAPL
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="2y")
        
        if data is None or len(data) < 100:
            logger.error("❌ Insufficient real data for AAPL")
            return None
            
        logger.info(f"📊 Real market data: {len(data)} days")
        
        # Train REAL NEAT agent and capture predictions
        from enhanced_options_features import EnhancedOptionsFeatureEngine
        feature_engine = EnhancedOptionsFeatureEngine()
        
        # Prepare features
        features_df = feature_engine.engineer_enhanced_features("AAPL", data)
        features = features_df.values
        
        logger.info(f"📈 Features prepared: {features.shape}")
        
        # Train REAL agent
        result = neat_system.high_performance_train("AAPL")
        
        if not result['success']:
            logger.error(f"❌ Training failed: {result.get('error', 'Unknown')}")
            return None
            
        # Get REAL trained genome
        winner_genome = result['winner']
        
        # Create REAL neural network
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           'neat_config.txt')
        net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
        
        # Generate REAL predictions
        real_predictions = []
        for i in range(len(features)):
            prediction = net.activate(features[i])[0]
            real_predictions.append(prediction)
        
        logger.info(f"🔮 Generated {len(real_predictions)} REAL predictions")
        logger.info(f"   Prediction range: {min(real_predictions):.3f} - {max(real_predictions):.3f}")
        
        # Return REAL data
        return {
            'data': data,
            'features': features,
            'predictions': real_predictions,
            'genome': winner_genome,
            'performance': result['performance']
        }
        
    except Exception as e:
        logger.error(f"❌ Error capturing REAL data: {str(e)}")
        return None

def generate_real_validation_charts(real_data):
    """Generate validation charts using REAL NEAT agent data."""
    
    if real_data is None:
        logger.error("❌ No real data available")
        return False
        
    data = real_data['data']
    predictions = real_data['predictions']
    performance = real_data['performance']
    
    prices = data['Close'].values
    dates = data.index
    
    logger.info("📊 Generating REAL validation charts...")
    
    # Create comprehensive validation charts
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('REAL NEAT Agent Validation - Actual Performance Data', fontsize=16, fontweight='bold')
    
    # Chart 1: REAL Predictions vs Actual Price Changes
    actual_changes = []
    for i in range(len(prices)-1):
        change = (prices[i+1] - prices[i]) / prices[i] * 100
        actual_changes.append(change)
    
    prediction_signals = [(pred - 0.5) * 10 for pred in predictions[:-1]]
    dates_subset = dates[:-1]
    
    ax1.plot(dates_subset, actual_changes, color='#4682B4', linewidth=2, 
            label='Actual Next-Day Change (%)', alpha=0.8)
    ax1.plot(dates_subset, prediction_signals, color='red', linewidth=2, 
            label='REAL NEAT Predictions', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title('REAL NEAT Predictions vs Actual Price Changes')
    ax1.set_ylabel('Price Change (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: REAL Performance Metrics
    metrics = ['Win Rate (%)', 'Annual Return (%)', 'Sharpe Ratio', 'Assignment Rate (%)']
    values = [
        performance.get('win_rate', 0),
        performance.get('annual_return', 0), 
        performance.get('sharpe_ratio', 0),
        performance.get('assignment_rate', 0)
    ]
    targets = [80, 110, 2.0, 5.0]
    
    colors = ['green' if v >= t else 'red' for v, t in zip(values, targets)]
    bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
    ax2.set_title('REAL Performance Metrics vs Targets')
    ax2.set_ylabel('Value')
    
    # Add target lines and value labels
    for i, (metric, value, target) in enumerate(zip(metrics, values, targets)):
        ax2.axhline(y=target, color='orange', linestyle='--', alpha=0.7)
        ax2.text(i, value + max(max(values), max(targets)) * 0.05, f'{value:.1f}', 
                ha='center', va='bottom', fontweight='bold')
        ax2.text(i, target + max(max(values), max(targets)) * 0.02, f'Target: {target}', 
                ha='center', va='bottom', color='orange', fontsize=8)
    
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: Price Action with REAL Trading Signals
    ax3.plot(dates, prices, color='#4682B4', linewidth=2, label='AAPL Price', alpha=0.6)
    ax3.plot(dates, pd.Series(prices).rolling(20).mean(), 'orange', alpha=0.7, label='20-day MA')
    
    # Mark REAL entry signals
    entry_threshold = 0.6
    exit_threshold = 0.3
    
    for i, (date, pred) in enumerate(zip(dates, predictions)):
        if pred > entry_threshold:  # REAL buy signal
            ax3.scatter(date, prices[i], color='green', marker='^', s=100, alpha=0.8, zorder=4)
        elif pred < exit_threshold:  # REAL sell signal
            ax3.scatter(date, prices[i], color='red', marker='v', s=100, alpha=0.8, zorder=4)
    
    ax3.set_title('AAPL Price with REAL NEAT Trading Signals')
    ax3.set_ylabel('Price ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Chart 4: REAL Prediction Confidence Over Time
    ax4.plot(dates, predictions, color='purple', linewidth=2, label='REAL NEAT Confidence')
    ax4.axhline(y=entry_threshold, color='green', linestyle='--', alpha=0.7, label=f'Buy Threshold ({entry_threshold})')
    ax4.axhline(y=exit_threshold, color='red', linestyle='--', alpha=0.7, label=f'Sell Threshold ({exit_threshold})')
    ax4.fill_between(dates, exit_threshold, entry_threshold, alpha=0.2, color='yellow', label='Hold Zone')
    ax4.set_title('REAL NEAT Prediction Confidence Over Time')
    ax4.set_ylabel('Confidence (0-1)')
    ax4.set_ylim(0, 1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save REAL validation charts
    chart_file = 'REAL_NEAT_validation_fixed.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    logger.info(f"💾 REAL validation charts saved to: {chart_file}")
    
    plt.show()
    
    # Print REAL performance summary
    logger.info("📊 REAL NEAT Agent Performance Summary:")
    logger.info(f"   Win Rate: {performance.get('win_rate', 0):.1f}%")
    logger.info(f"   Annual Return: {performance.get('annual_return', 0):.1f}%")
    logger.info(f"   Assignment Rate: {performance.get('assignment_rate', 0):.1f}%")
    logger.info(f"   Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
    logger.info(f"   Max Drawdown: {performance.get('max_drawdown', 0):.1f}%")
    
    return True

def main():
    """Generate REAL NEAT validation charts."""
    logger.info("🎯 Starting REAL NEAT Validation Chart Generation")
    
    # Capture REAL NEAT data
    real_data = capture_real_neat_data()
    
    if real_data is None:
        logger.error("❌ Failed to capture REAL NEAT data")
        return 1
    
    # Generate REAL validation charts
    success = generate_real_validation_charts(real_data)
    
    if success:
        logger.info("✅ REAL NEAT validation charts generated successfully!")
        logger.info("📊 Charts show ACTUAL trained NEAT agent performance")
        logger.info("🚫 NO SIMULATION - All data is from real NEAT agent")
    else:
        logger.error("❌ Failed to generate REAL validation charts")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
