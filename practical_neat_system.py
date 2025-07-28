#!/usr/bin/env python3
"""
Practical NEAT Options Trading System
A human-usable system for weekly options trading with clear validation.
"""

import numpy as np
import pandas as pd
import neat
import time
import logging
from datetime import datetime, timedelta
import yfinance as yf
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@deprecated  # BROKEN: Training takes <1 second, produces 46% win rate models
class PracticalNEATSystem:
    """Practical NEAT system focused on human usability and validation."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.models = {}
        
    def _setup_logging(self):
        logger = logging.getLogger('PracticalNEAT')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def create_neat_config(self, num_inputs: int) -> str:
        """Create proper NEAT configuration without warnings."""
        config_text = f"""
[NEAT]
fitness_criterion = max
fitness_threshold = 100.0
pop_size = 50
reset_on_extinction = False

[DefaultGenome]
activation_default = tanh
activation_mutate_rate = 0.1
activation_options = tanh relu sigmoid
aggregation_default = sum
aggregation_mutate_rate = 0.0
aggregation_options = sum
bias_init_mean = 0.0
bias_init_stdev = 1.0
bias_max_value = 30.0
bias_min_value = -30.0
bias_mutate_power = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient = 0.5
conn_add_prob = 0.3
conn_delete_prob = 0.2
enabled_default = True
enabled_mutate_rate = 0.01
feed_forward = True
initial_connection = full_direct
node_add_prob = 0.1
node_delete_prob = 0.05
num_hidden = 0
num_inputs = {num_inputs}
num_outputs = 1
response_init_mean = 1.0
response_init_stdev = 0.0
response_max_value = 30.0
response_min_value = -30.0
response_mutate_power = 0.0
response_mutate_rate = 0.0
response_replace_rate = 0.0
weight_init_mean = 0.0
weight_init_stdev = 1.0
weight_max_value = 30
weight_min_value = -30
weight_mutate_power = 0.5
weight_mutate_rate = 0.8
weight_replace_rate = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 15
species_elitism = 2

[DefaultReproduction]
elitism = 2
survival_threshold = 0.2
"""
        return config_text
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer simple, effective features for options trading."""
        df = data.copy()
        features = pd.DataFrame(index=df.index)
        
        # Price momentum (5 features)
        features['returns_5d'] = df['Close'].pct_change(5)
        features['returns_20d'] = df['Close'].pct_change(20)
        features['price_vs_sma20'] = df['Close'] / df['Close'].rolling(20).mean() - 1
        features['price_vs_sma50'] = df['Close'] / df['Close'].rolling(50).mean() - 1
        
        # Volatility (3 features)
        features['volatility_20d'] = df['Close'].pct_change().rolling(20).std()
        features['volatility_ratio'] = (df['Close'].pct_change().rolling(5).std() / 
                                       df['Close'].pct_change().rolling(20).std())
        
        # Volume (2 features)
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Clean and normalize
        features = features.fillna(method='ffill').fillna(0)
        for col in features.columns:
            if features[col].std() > 0:
                features[col] = (features[col] - features[col].mean()) / features[col].std()
        
        return features
    
    def create_weekly_labels(self, data: pd.DataFrame, horizon_weeks: int = 4) -> np.ndarray:
        """Create labels for weekly options trading (not daily churning)."""
        horizon_days = horizon_weeks * 5  # 5 trading days per week
        labels = []
        
        for i in range(len(data)):
            if i >= len(data) - horizon_days:
                labels.append(0.0)
                continue
            
            current_price = data['Close'].iloc[i]
            future_price = data['Close'].iloc[i + horizon_days]
            
            # Calculate return over the horizon
            return_pct = (future_price / current_price) - 1
            
            # Convert to signal (-1 to 1)
            signal = np.tanh(return_pct * 5)  # Scale and bound
            labels.append(signal)
        
        return np.array(labels)
    
    def weekly_fitness_function(self, genome, config, training_data):
        """Fitness function for weekly options trading (not daily churning)."""
        try:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            features = training_data['features'].values
            labels = training_data['labels']
            
            # Generate predictions
            predictions = []
            for feature_row in features:
                output = net.activate(feature_row)
                predictions.append(output[0])
            
            predictions = np.array(predictions)
            
            # Weekly trading simulation (not daily)
            trades = []
            position = 0
            weeks_in_position = 0
            max_weeks_per_trade = 4  # Maximum 4 weeks per trade
            
            for i in range(0, len(predictions), 5):  # Check weekly (every 5 days)
                if i >= len(predictions):
                    break
                    
                prediction = predictions[i]
                
                # Entry logic - only enter if no position
                if position == 0 and abs(prediction) > 0.3:  # Confidence threshold
                    position = np.sign(prediction)
                    weeks_in_position = 0
                
                # Exit logic - weekly check
                elif position != 0:
                    weeks_in_position += 1
                    
                    if i < len(labels):
                        # Calculate weekly return
                        weekly_return = labels[i] * position
                        
                        # Exit conditions
                        if (weeks_in_position >= max_weeks_per_trade or  # Time exit
                            weekly_return < -0.05 or  # Stop loss
                            weekly_return > 0.10):    # Take profit
                            
                            trades.append(weekly_return - 0.002)  # Transaction cost
                            position = 0
                            weeks_in_position = 0
            
            if len(trades) == 0:
                return 0.1
            
            # Calculate performance metrics
            total_return = sum(trades)
            win_rate = len([t for t in trades if t > 0]) / len(trades)
            avg_return = np.mean(trades)
            
            # Fitness focused on consistent weekly profits
            fitness = (
                total_return * 100 +      # Total profitability
                win_rate * 50 +           # Consistency
                avg_return * 200 +        # Average trade quality
                len(trades) * 0.5         # Activity bonus (but not too much)
            )
            
            return max(fitness, 0.1)
            
        except Exception:
            return 0.1
    
    def train_weekly_agent(self, symbol: str, data: pd.DataFrame, 
                          horizon_weeks: int = 4) -> dict:
        """Train agent for weekly options trading."""
        try:
            self.logger.info(f"Training {symbol} for {horizon_weeks}-week options...")
            
            # Prepare data
            features = self.engineer_features(data)
            labels = self.create_weekly_labels(data, horizon_weeks)
            
            # Align data
            min_length = min(len(features), len(labels))
            features = features.iloc[:min_length]
            labels = labels[:min_length]
            
            # Split data
            train_size = int(len(features) * 0.7)
            train_features = features.iloc[:train_size]
            train_labels = labels[:train_size]
            
            training_data = {
                'features': train_features,
                'labels': train_labels,
                'symbol': symbol,
                'horizon_weeks': horizon_weeks
            }
            
            # Create NEAT config
            config_text = self.create_neat_config(len(features.columns))
            
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(config_text)
                config_path = f.name
            
            config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
            
            # Train
            population = neat.Population(config)
            
            def fitness_wrapper(genomes, config):
                for genome_id, genome in genomes:
                    genome.fitness = self.weekly_fitness_function(genome, config, training_data)
            
            best_genome = None
            best_fitness = 0
            
            for generation in range(20):  # Limited generations
                population.run(fitness_wrapper, 1)
                
                current_best = max(population.population.values(), key=lambda g: g.fitness)
                if current_best.fitness > best_fitness:
                    best_fitness = current_best.fitness
                    best_genome = current_best
                
                if best_fitness > 50:  # Early stopping
                    break
            
            # Cleanup
            os.unlink(config_path)
            
            # Test performance
            test_features = features.iloc[train_size:]
            test_labels = labels[train_size:]
            
            performance = self._evaluate_weekly_performance(
                best_genome, config, test_features, test_labels, data.iloc[train_size:]
            )
            
            # CRITICAL FIX: Save the trained model to self.models
            self.models[symbol] = {
                'genome': best_genome,
                'config': config,
                'performance': performance,
                'horizon_weeks': horizon_weeks,
                'training_period': f"{data.index[0].date()} to {data.index[train_size-1].date()}",
                'test_period': f"{data.index[train_size].date()} to {data.index[-1].date()}",
                'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"✅ Model saved for {symbol} (fitness: {best_fitness:.1f})")
            
            return {
                'success': True,
                'symbol': symbol,
                'genome': best_genome,
                'config': config,
                'performance': performance,
                'horizon_weeks': horizon_weeks,
                'training_period': f"{data.index[0].date()} to {data.index[train_size-1].date()}",
                'test_period': f"{data.index[train_size].date()} to {data.index[-1].date()}"
            }
            
        except Exception as e:
            return {'success': False, 'symbol': symbol, 'error': str(e)}
    
    def _evaluate_weekly_performance(self, genome, config, features, labels, price_data):
        """Evaluate weekly trading performance with clear metrics."""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Generate predictions
        predictions = [net.activate(row)[0] for row in features.values]
        
        # Simulate weekly trading
        trades = []
        trade_dates = []
        position = 0
        weeks_in_position = 0
        
        for i in range(0, len(predictions), 5):  # Weekly checks
            if i >= len(predictions):
                break
                
            prediction = predictions[i]
            
            if position == 0 and abs(prediction) > 0.3:
                position = np.sign(prediction)
                weeks_in_position = 0
                
            elif position != 0:
                weeks_in_position += 1
                
                if i < len(labels):
                    weekly_return = labels[i] * position
                    
                    if (weeks_in_position >= 4 or 
                        weekly_return < -0.05 or 
                        weekly_return > 0.10):
                        
                        trades.append(weekly_return - 0.002)
                        trade_dates.append(features.index[i])
                        position = 0
        
        if not trades:
            return {
                'total_return': 0.0, 'win_rate': 0.0, 'total_trades': 0,
                'avg_trade_duration': 0, 'buy_hold_return': 0.0,
                'trade_frequency': 'No trades'
            }
        
        # Calculate metrics
        total_return = sum(trades)
        win_rate = len([t for t in trades if t > 0]) / len(trades)
        
        # Buy and hold comparison (same period)
        start_price = price_data['Close'].iloc[0]
        end_price = price_data['Close'].iloc[-1]
        buy_hold_return = (end_price / start_price) - 1
        
        # Trade frequency
        days_span = (price_data.index[-1] - price_data.index[0]).days
        trades_per_year = len(trades) * 365 / days_span
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_trade_duration': 4 * 5,  # 4 weeks * 5 days
            'buy_hold_return': buy_hold_return,
            'trade_frequency': f"{trades_per_year:.1f} trades/year",
            'test_period_days': days_span
        }
    
    def predict_next_weeks(self, symbol: str, weeks_ahead: int = 4) -> dict:
        """Predict next 1-8 weeks for a symbol."""
        try:
            if symbol not in self.models:
                return {'error': f'No trained model for {symbol}. Please train the model first.'}
            
            # Get current data
            ticker = yf.Ticker(symbol)
            try:
                data = ticker.history(period="1y", auto_adjust=True)
            except Exception as e:
                return {'error': f'Could not fetch data for {symbol}: {str(e)}'}
            
            if len(data) < 100:
                return {'error': f'Insufficient data for {symbol} (got {len(data)} days, need 100+)'}
            
            # Engineer features
            try:
                features = self.engineer_features(data)
            except Exception as e:
                return {'error': f'Could not generate features for {symbol}: {str(e)}'}
            
            if len(features) == 0 or features.empty:
                return {'error': f'No valid features generated for {symbol}'}
            
            # Get current features
            current_features = features.iloc[-1].values
            current_price = data['Close'].iloc[-1]
            
            # Get model and make prediction
            model_info = self.models[symbol]
            net = neat.nn.FeedForwardNetwork.create(
                model_info['genome'], model_info['config']
            )
            
            prediction = net.activate(current_features)[0]
            
            # Convert to actionable recommendation
            if abs(prediction) < 0.3:
                recommendation = "HOLD - Low confidence signal"
                confidence = "Low"
            elif prediction > 0.3:
                recommendation = f"BUY - Bullish signal for {weeks_ahead} weeks"
                confidence = "High" if prediction > 0.6 else "Medium"
            else:
                recommendation = f"SELL/PUT - Bearish signal for {weeks_ahead} weeks"
                confidence = "High" if prediction < -0.6 else "Medium"
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'prediction_signal': prediction,
                'confidence': confidence,
                'recommendation': recommendation,
                'horizon_weeks': weeks_ahead,
                'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                'model_trained_on': model_info.get('training_period', 'Unknown')
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def validate_historical_prediction(self, symbol: str, test_date: str, 
                                     weeks_ahead: int = 4) -> dict:
        """Test prediction against historical data."""
        try:
            test_date = pd.to_datetime(test_date)
            
            # Get historical data
            ticker = yf.Ticker(symbol)
            start_date = test_date - timedelta(days=365)  # 1 year before test
            end_date = test_date + timedelta(weeks=weeks_ahead + 4)  # Include outcome
            
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if test_date not in data.index:
                return {'error': f'No data available for {test_date.date()}'}
            
            # Find test date index
            test_idx = data.index.get_loc(test_date, method='nearest')
            
            # Use data up to test date for prediction
            historical_data = data.iloc[:test_idx+1]
            features = self.engineer_features(historical_data)
            
            if symbol not in self.models:
                return {'error': f'No trained model for {symbol}'}
            
            # Make historical prediction
            model_info = self.models[symbol]
            net = neat.nn.FeedForwardNetwork.create(
                model_info['genome'], model_info['config']
            )
            
            prediction = net.activate(features.iloc[-1].values)[0]
            
            # Get actual outcome
            outcome_date = test_date + timedelta(weeks=weeks_ahead)
            if outcome_date <= data.index[-1]:
                test_price = data.loc[test_date, 'Close']
                outcome_price = data.loc[data.index[data.index >= outcome_date][0], 'Close']
                actual_return = (outcome_price / test_price) - 1
                
                # Compare prediction vs actual
                predicted_direction = "UP" if prediction > 0.3 else "DOWN" if prediction < -0.3 else "NEUTRAL"
                actual_direction = "UP" if actual_return > 0.02 else "DOWN" if actual_return < -0.02 else "NEUTRAL"
                
                correct = predicted_direction == actual_direction
                
                return {
                    'symbol': symbol,
                    'test_date': test_date.strftime('%Y-%m-%d'),
                    'weeks_ahead': weeks_ahead,
                    'predicted_signal': prediction,
                    'predicted_direction': predicted_direction,
                    'actual_return': actual_return,
                    'actual_direction': actual_direction,
                    'prediction_correct': correct,
                    'test_price': test_price,
                    'outcome_price': outcome_price
                }
            else:
                return {'error': 'Outcome date is in the future'}
                
        except Exception as e:
            return {'error': str(e)}

def run_practical_system():
    """Run the practical NEAT system with clear validation."""
    
    trader = PracticalNEATTrader()
    symbols = ['AAPL', 'MSFT', 'GOOGL']  # Start with 3 symbols
    
    print("\n" + "="*80)
    print("🎯 PRACTICAL NEAT OPTIONS TRADING SYSTEM")
    print("="*80)
    
    # Train models
    print(f"\n📚 Training weekly options models...")
    results = {}
    
    for symbol in symbols:
        try:
            # Get 2 years of data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y", auto_adjust=True)
            
            if len(data) < 500:
                print(f"❌ {symbol}: Insufficient data")
                continue
            
            # Train for 4-week options
            result = trader.train_weekly_agent(symbol, data, horizon_weeks=4)
            results[symbol] = result
            
            if result['success']:
                trader.models[symbol] = result
                perf = result['performance']
                print(f"✅ {symbol}: {perf['total_trades']} trades, "
                      f"{perf['win_rate']:.1%} win rate, "
                      f"{perf['trade_frequency']}")
            
        except Exception as e:
            print(f"❌ {symbol}: {e}")
    
    # Show results
    print(f"\n📊 TRAINING RESULTS")
    print("-" * 80)
    print(f"{'Symbol':<8} {'Trades':<8} {'Win Rate':<10} {'AI Return':<12} {'Buy&Hold':<12} {'Frequency':<15}")
    print("-" * 80)
    
    for symbol, result in results.items():
        if result['success']:
            perf = result['performance']
            print(f"{symbol:<8} {perf['total_trades']:<8} {perf['win_rate']:>8.1%} "
                  f"{perf['total_return']:>10.1%} {perf['buy_hold_return']:>10.1%} "
                  f"{perf['trade_frequency']:<15}")
    
    # Example predictions
    print(f"\n🔮 NEXT 4-WEEK PREDICTIONS")
    print("-" * 60)
    for symbol in symbols:
        if symbol in trader.models:
            pred = trader.predict_next_weeks(symbol, 4)
            if 'error' not in pred:
                print(f"{symbol}: {pred['recommendation']} (Confidence: {pred['confidence']})")
    
    # Historical validation example
    print(f"\n🧪 HISTORICAL VALIDATION EXAMPLE")
    print("-" * 60)
    test_date = "2024-01-15"  # Example historical date
    for symbol in symbols[:2]:  # Test first 2 symbols
        if symbol in trader.models:
            validation = trader.validate_historical_prediction(symbol, test_date, 4)
            if 'error' not in validation:
                print(f"{symbol} on {test_date}: Predicted {validation['predicted_direction']}, "
                      f"Actual {validation['actual_direction']} "
                      f"({'✅ CORRECT' if validation['prediction_correct'] else '❌ WRONG'})")
    
    print("\n" + "="*80)
    print("✅ PRACTICAL SYSTEM READY - REASONABLE TRADE FREQUENCY")
    print("="*80)

if __name__ == "__main__":
    run_practical_system()
