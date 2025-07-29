#!/usr/bin/env python3
"""
Test the realistic fitness calculations and ticker-based model organization
"""

import os
import sys
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from high_performance_neat_system import HighPerformanceNEATSystem

def main():
    """Test realistic fitness and ticker organization."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing Realistic Fitness Calculations and Ticker Organization")
    
    # Initialize NEAT system with small population for quick test
    neat_system = HighPerformanceNEATSystem(use_gpu=True)
    
    # Override settings for quick test
    neat_system.population_size = 10  # Small population for quick test
    neat_system.max_generations = 3   # Just a few generations
    
    logger.info(f"📊 Test Configuration:")
    logger.info(f"   Population: {neat_system.population_size}")
    logger.info(f"   Generations: {neat_system.max_generations}")
    logger.info(f"   Target Win Rate: {neat_system.target_win_rate:.1%}")
    logger.info(f"   Target Annual Return: {neat_system.target_annual_return:.1%}")
    logger.info(f"   Target Sharpe: {neat_system.target_sharpe:.1f}")
    logger.info(f"   Max Assignment Rate: {neat_system.target_assignment_rate:.1%}")
    
    # Test with AAPL
    symbol = "AAPL"
    logger.info(f"🍎 Training {symbol} with realistic bounds...")
    
    try:
        result = neat_system.high_performance_train(symbol, horizon_weeks=4)
        
        if result['success']:
            logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"   Exceptional Model: {result.get('exceptional', False)}")
            logger.info(f"   Training Time: {result.get('training_time', 0):.1f}s")
            logger.info(f"   Generations: {result.get('generations', 0)}")
            
            # Check performance metrics for realism
            perf = result.get('performance', {})
            logger.info(f"📈 PERFORMANCE METRICS:")
            logger.info(f"   Win Rate: {perf.get('win_rate', 0):.1%}")
            logger.info(f"   Annual Return: {perf.get('annual_return', 0):.1%}")
            logger.info(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            logger.info(f"   Assignment Rate: {perf.get('assignment_rate', 0):.1%}")
            logger.info(f"   Max Drawdown: {perf.get('max_drawdown', 0):.1%}")
            
            # Validate realistic bounds
            annual_return = perf.get('annual_return', 0)
            sharpe_ratio = perf.get('sharpe_ratio', 0)
            
            realistic = True
            if annual_return > 2.0:  # > 200%
                logger.warning(f"⚠️  Annual return {annual_return:.1%} exceeds realistic bounds (200%)")
                realistic = False
            
            if sharpe_ratio > 5.0:  # > 5.0
                logger.warning(f"⚠️  Sharpe ratio {sharpe_ratio:.2f} exceeds realistic bounds (5.0)")
                realistic = False
            
            if realistic:
                logger.info("✅ ALL METRICS WITHIN REALISTIC BOUNDS!")
            else:
                logger.error("❌ METRICS STILL UNREALISTIC - FURTHER FIXES NEEDED")
            
            # Check ticker-based organization
            ticker_dir = f'models/{symbol}'
            if os.path.exists(ticker_dir):
                logger.info(f"✅ TICKER-BASED ORGANIZATION WORKING:")
                logger.info(f"   Directory: {ticker_dir}")
                
                if os.path.exists(f'{ticker_dir}/current_model.pkl'):
                    logger.info(f"   ✅ Current model saved")
                
                if os.path.exists(f'{ticker_dir}/metadata.json'):
                    logger.info(f"   ✅ Metadata saved")
                    
                    import json
                    with open(f'{ticker_dir}/metadata.json', 'r') as f:
                        metadata = json.load(f)
                    logger.info(f"   📊 Fitness Score: {metadata.get('fitness_score', 'Unknown')}")
            else:
                logger.error(f"❌ TICKER DIRECTORY NOT CREATED: {ticker_dir}")
        
        else:
            # Handle both rejection and actual errors
            if 'reason' in result:
                logger.warning(f"⚠️ Model rejected: {result['reason']}")
                logger.info("📊 This is expected behavior - model didn't meet exceptional standards")
            else:
                logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
