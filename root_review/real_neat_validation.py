#!/usr/bin/env python3
"""
REAL NEAT Validation Charts - Uses ACTUAL trained NEAT agent, NO SIMULATION
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from high_performance_neat_system import HighPerformanceNEATSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def generate_real_neat_validation():
    """Generate validation charts using ACTUAL NEAT agent - NO SIMULATION."""
    
    logger.info("🎯 Generating REAL NEAT Validation Charts (NO SIMULATION)")
    
    # Initialize REAL NEAT system
    neat_system = HighPerformanceNEATSystem()
    
    # Train ACTUAL NEAT agent on AAPL
    logger.info("🚀 Training REAL NEAT agent on AAPL...")
    
    try:
        # Get real data and train real agent
        # Use the data pipeline that's already in the NEAT system
        data_pipeline = neat_system.data_pipeline
        
        # Collect REAL market data
        data = data_pipeline.collect_extended_data("AAPL", period="2y")
        if data is None or len(data) < 100:
            logger.error("❌ Insufficient real data for AAPL")
            return False
            
        # Prepare REAL features
        features = data_pipeline.prepare_enhanced_features(data)
        
        # Train REAL NEAT agent
        result = neat_system.train_agent("AAPL", features, max_generations=10)
        
        if not result['success']:
            logger.error(f"❌ REAL training failed: {result.get('error', 'Unknown error')}")
            return False
            
        # Get REAL performance from REAL agent
        performance = result['performance']
        winner_genome = result['winner']
        
        logger.info("📊 REAL NEAT Agent Performance:")
        logger.info(f"   Win Rate: {performance['win_rate']:.1f}%")
        logger.info(f"   Annual Return: {performance['annual_return']:.1f}%")
        logger.info(f"   Assignment Rate: {performance['assignment_rate']:.1f}%")
        logger.info(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        logger.info(f"   Max Drawdown: {performance['max_drawdown']:.1f}%")
        
        # Generate REAL predictions using trained agent
        logger.info("🔮 Generating REAL predictions from trained NEAT agent...")
        
        # Use the actual trained neural network
        import neat
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           'neat_config.txt')
        net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
        
        # Get REAL predictions from REAL agent
        real_predictions = []
        for i in range(len(features)):
            prediction = net.activate(features[i])[0]
            real_predictions.append(prediction)
        
        prices = data['Close'].values
        dates = data.index
        
        # Create validation charts with REAL data
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('REAL NEAT Agent Validation - NO SIMULATION', fontsize=16, fontweight='bold')
        
        # Chart 1: REAL Predictions vs Actual Price Changes
        actual_changes = [(prices[i+1] - prices[i]) / prices[i] * 100 for i in range(len(prices)-1)]
        prediction_signals = [(pred - 0.5) * 8 for pred in real_predictions[:-1]]
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
        values = [performance['win_rate'], performance['annual_return'], 
                 performance['sharpe_ratio'], performance['assignment_rate']]
        targets = [80, 110, 2.0, 5.0]
        
        bars = ax2.bar(metrics, values, color=['green' if v >= t else 'orange' for v, t in zip(values, targets)])
        ax2.set_title('REAL Performance Metrics vs Targets')
        ax2.set_ylabel('Value')
        
        # Add target lines and value labels
        for i, (metric, value, target) in enumerate(zip(metrics, values, targets)):
            ax2.axhline(y=target, color='red', linestyle='--', alpha=0.7)
            ax2.text(i, value + max(values) * 0.05, f'{value:.1f}', 
                    ha='center', va='bottom', fontweight='bold')
            ax2.text(i, target + max(values) * 0.02, f'Goal: {target}', 
                    ha='center', va='bottom', color='red', fontsize=8)
        
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Price Action with REAL Trading Signals
        ax3.plot(dates, prices, color='#4682B4', linewidth=2, label='AAPL Price', alpha=0.6)
        ax3.plot(dates, pd.Series(prices).rolling(20).mean(), 'orange', alpha=0.7, label='20-day MA')
        
        # Mark REAL entry signals where prediction > 0.6
        for i, (date, pred) in enumerate(zip(dates, real_predictions)):
            if pred > 0.6:  # REAL entry threshold
                ax3.scatter(date, prices[i], color='#00FF00', marker='^', s=120, alpha=0.8, zorder=4)
        
        ax3.set_title('AAPL Price with REAL NEAT Trading Signals')
        ax3.set_ylabel('Price ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: REAL Prediction Confidence
        ax4.plot(dates, real_predictions, color='purple', linewidth=2, label='REAL NEAT Confidence')
        ax4.axhline(y=0.6, color='green', linestyle='--', alpha=0.7, label='Buy Threshold (0.6)')
        ax4.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Sell Threshold (0.3)')
        ax4.set_title('REAL NEAT Prediction Confidence Over Time')
        ax4.set_ylabel('Confidence (0-1)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save REAL validation charts
        chart_file = 'REAL_NEAT_validation.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        logger.info(f"💾 REAL validation charts saved to: {chart_file}")
        
        plt.show()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in REAL NEAT validation: {str(e)}")
        return False

def main():
    """Generate REAL NEAT validation charts."""
    logger.info("🎯 Starting REAL NEAT Validation (NO SIMULATION)")
    
    success = generate_real_neat_validation()
    
    if success:
        logger.info("✅ REAL NEAT validation charts generated successfully!")
        logger.info("📊 Charts show ACTUAL trained NEAT agent performance")
    else:
        logger.error("❌ Failed to generate REAL validation charts")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
