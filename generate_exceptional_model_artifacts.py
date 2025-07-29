#!/usr/bin/env python3
"""
Generate missing charts and database entries for the exceptional AAPL model
that was just trained with fitness score 100.0
"""

import pickle
import os
import sys
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from high_performance_neat_system import HighPerformanceNEATSystem

def main():
    """Generate comprehensive charts and database entries for exceptional model."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load the exceptional model that was just saved
    model_path = 'models/AAPL_high_performance_model.pkl'
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    logger.info(f"📂 Loading exceptional model from {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        logger.info(f"✅ Model loaded successfully:")
        logger.info(f"   Symbol: {model_data['symbol']}")
        logger.info(f"   Fitness: {model_data.get('fitness_score', 'Unknown')}")
        logger.info(f"   Training Time: {model_data['training_time']:.1f}s")
        
        # Initialize the NEAT system
        neat_system = HighPerformanceNEATSystem(use_gpu=True)
        
        # Load training data for chart generation
        from enhanced_data_pipeline import EnhancedDataPipeline
        
        logger.info("📊 Loading training data for chart generation...")
        data_pipeline = EnhancedDataPipeline()
        symbol = model_data['symbol']
        
        # Collect data (same as used in training)
        raw_data = data_pipeline.collect_extended_data(symbol)
        if raw_data is None or raw_data.empty:
            logger.error(f"Failed to load data for {symbol}")
            return
        
        # Prepare training data format
        features = data_pipeline.create_features(raw_data)
        labels = neat_system.create_options_labels(raw_data, horizon_weeks=4)
        
        # Align features and labels
        min_length = min(len(features), len(labels))
        features = features[:min_length]
        labels = labels[:min_length]
        
        training_data = {
            'features': features.values,
            'labels': labels,
            'prices': raw_data['Close'].values[:min_length],
            'dates': raw_data.index[:min_length]
        }
        
        logger.info(f"📈 Training data prepared: {len(features)} samples")
        
        # Generate comprehensive validation charts
        logger.info("🎨 Generating comprehensive validation charts...")
        chart_paths = neat_system.generate_validation_charts(
            model_data['genome'], 
            model_data['config'], 
            training_data, 
            symbol, 
            model_data['performance']
        )
        
        logger.info(f"✅ Generated {len(chart_paths)} validation charts:")
        for path in chart_paths:
            logger.info(f"   📊 {path}")
        
        # Save to database
        logger.info("💾 Saving to database...")
        neat_system.save_model_to_database(symbol, model_data, model_data['performance'])
        
        # Check database
        import sqlite3
        db_path = 'models/exceptional_models.db'
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM exceptional_models WHERE symbol = ?", (symbol,))
            count = cursor.fetchone()[0]
            conn.close()
            logger.info(f"✅ Database updated: {count} records for {symbol}")
        
        logger.info("🎉 EXCEPTIONAL MODEL ARTIFACTS COMPLETE!")
        logger.info("   ✅ Comprehensive charts generated")
        logger.info("   ✅ Database entries created")
        logger.info("   ✅ Model validation complete")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate artifacts: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
