#!/usr/bin/env python3
"""
Main entry point for NEAT Stock Price Prediction System.
Runs the NEAT algorithm to predict stock prices for a given ticker.
"""
import os
import argparse
import logging
import sys
from datetime import datetime
from neat_stock_predictor import NEATStockPredictor

def setup_logging():
    """Set up logging configuration."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"{log_dir}/neat_stock_predictor_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def main():
    """Main function to run the NEAT stock predictor."""
    # Set up logging
    logger = setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NEAT Stock Price Predictor')
    parser.add_argument('--ticker', type=str, default='AAPL',
                        help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--generations', type=int, default=50,
                        help='Number of generations for NEAT (default: 50)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from saved state')
    parser.add_argument('--predict-only', action='store_true',
                        help='Skip training and only generate predictions')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model on validation data')
    args = parser.parse_args()
    
    logger.info(f"Starting NEAT Stock Price Predictor for {args.ticker}")
    logger.info(f"Configuration: generations={args.generations}, resume={args.resume}, "
               f"predict_only={args.predict_only}, evaluate={args.evaluate}")
    
    try:
        # Initialize predictor
        predictor = NEATStockPredictor(ticker=args.ticker)
        
        # Train model (if not predict-only)
        if not args.predict_only:
            logger.info("Starting training")
            train_results = predictor.train(generations=args.generations, resume=args.resume)
            
            if not train_results.get('success'):
                logger.error(f"Training failed: {train_results.get('error')}")
                return
            
            logger.info(f"Training completed in {train_results.get('training_time', 0):.2f} seconds")
            logger.info(f"Best fitness: {train_results.get('best_genome', {}).fitness if train_results.get('best_genome') else 'N/A'}")
        
        # Generate predictions
        logger.info("Generating predictions")
        prediction_results = predictor.predict()
        
        if not prediction_results.get('success'):
            logger.error(f"Prediction failed: {prediction_results.get('error')}")
            return
        
        # Generate charts
        logger.info("Generating charts")
        chart_results = predictor.generate_charts(prediction_results)
        
        if not chart_results.get('success'):
            logger.error(f"Chart generation failed: {chart_results.get('error')}")
            return
        
        logger.info(f"Charts saved to {predictor.charts_dir}/")
        
        # Evaluate model if requested
        if args.evaluate:
            logger.info("Evaluating model")
            eval_results = predictor.evaluate()
            
            if not eval_results.get('success'):
                logger.error(f"Evaluation failed: {eval_results.get('error')}")
                return
            
            logger.info(f"Average direction accuracy: {eval_results.get('avg_direction_accuracy', 0):.4f}")
            logger.info(f"Average price MSE: {eval_results.get('avg_price_mse', 0):.4f}")
        
        # Print key prediction highlights
        highlights = prediction_results.get('highlights', {})
        logger.info("\n" + "=" * 60)
        logger.info(f"PREDICTION HIGHLIGHTS FOR {args.ticker}")
        logger.info("=" * 60)
        
        # Current price
        current_price = prediction_results.get('current_price', 0)
        if hasattr(current_price, 'iloc'):
            current_price = float(current_price.iloc[0]) if len(current_price) > 0 else float(current_price)
        else:
            current_price = float(current_price)
        current_date = prediction_results.get('current_date', 'N/A')
        logger.info(f"Current price (as of {current_date}): ${current_price:.2f}")
        
        # Key predictions
        for day_key, highlight in sorted(highlights.items()):
            # Convert all potential pandas Series to native Python types
            price = highlight.get('Price', 0)
            if hasattr(price, 'iloc'):
                price = float(price.iloc[0]) if len(price) > 0 else float(price)
            else:
                price = float(price)
                
            direction = str(highlight.get('Direction', 'Unknown'))
            
            confidence = highlight.get('Confidence', 0)
            if hasattr(confidence, 'iloc'):
                confidence = float(confidence.iloc[0]) if len(confidence) > 0 else float(confidence)
            else:
                confidence = float(confidence)
                
            date = str(highlight.get('Date', 'Unknown'))
            
            # Calculate change
            change = ((price / current_price) - 1) * 100 if current_price else 0
            
            logger.info(f"{day_key}: ${price:.2f} ({direction}, {change:+.2f}%) on {date} - Confidence: {confidence:.2f}")
        
        logger.info("=" * 60)
        logger.info(f"Detailed predictions saved to: {prediction_results.get('predictions_file')}")
        logger.info(f"Charts saved to: {predictor.charts_dir}/")
        
    except Exception as e:
        logger.exception(f"Error running NEAT Stock Predictor: {e}")
        return

if __name__ == "__main__":
    main()
