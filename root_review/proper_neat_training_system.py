#!/usr/bin/env python3
"""
PROPER NEAT TRAINING SYSTEM - Sprint 2
Fix broken NEAT training to actually produce profitable models.

FIXES:
1. Training takes 5-15 minutes (not 1 second!)
2. Sophisticated fitness function optimized for profitability
3. Real options trading strategy (cash-secured puts)
4. Only save models meeting exceptional standards (80%+ win rate)
"""

import neat
import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
import pickle
import os
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from bulletproof_data_fetcher import BulletproofDataFetcher
from comprehensive_feature_engine import ComprehensiveFeatureEngine

class ProperNEATTrainingSystem:
    """Proper NEAT training system that takes appropriate time and produces profitable models."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.data_fetcher = BulletproofDataFetcher()
        self.feature_engine = ComprehensiveFeatureEngine()
        self.models = {}
        
        # PROPER TRAINING PARAMETERS (not broken 1-second training)
        self.min_training_time = 300  # 5 minutes minimum
        self.max_training_time = 900  # 15 minutes maximum
        self.min_generations = 50     # 50+ generations minimum
        self.population_size = 100    # Large population for diversity
        self.max_stagnation = 15      # Allow time for evolution
        
        # EXCEPTIONAL FITNESS TARGETS FOR PROFITABILITY
        self.target_win_rate = 0.80      # 80% win rate
        self.target_annual_return = 1.0  # 100% annual return
        self.target_sharpe = 2.0         # 2.0 Sharpe ratio
        self.max_drawdown = 0.08         # 8% max drawdown
        self.target_assignment_rate = 0.05  # 5% max assignment rate
        
    def _setup_logging(self):
        """Setup comprehensive logging."""
        logger = logging.getLogger('ProperNEAT')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def create_options_labels(self, data: pd.DataFrame, horizon_weeks: int = 4) -> np.ndarray:
        """Create sophisticated labels for cash-secured put strategy."""
        horizon_days = horizon_weeks * 5  # 5 trading days per week
        labels = []
        
        for i in range(len(data)):
            if i >= len(data) - horizon_days:
                labels.append(0.0)
                continue
            
            current_price = data['Close'].iloc[i]
            
            # Look ahead to see actual price movement
            future_prices = data['Close'].iloc[i:i+horizon_days+1]
            
            if len(future_prices) < horizon_days:
                labels.append(0.0)
                continue
            
            # Calculate various returns
            final_return = (future_prices.iloc[-1] / current_price) - 1
            max_return = (future_prices.max() / current_price) - 1
            min_return = (future_prices.min() / current_price) - 1
            
            # CASH-SECURED PUT STRATEGY LOGIC
            # Want stocks going UP to avoid assignment
            # Signal strength based on probability of profit
            
            if final_return > 0.08:  # Strong upward movement (8%+)
                signal = 0.9
            elif final_return > 0.04:  # Moderate upward movement (4-8%)
                signal = 0.7
            elif final_return > 0.01:  # Small upward movement (1-4%)
                signal = 0.5
            elif final_return > -0.02:  # Sideways (good for theta decay)
                signal = 0.3
            elif final_return > -0.05:  # Small decline (manageable, 2-5%)
                signal = -0.2
            elif final_return > -0.10:  # Moderate decline (5-10%)
                signal = -0.6
            else:  # Large decline (>10% - avoid!)
                signal = -0.9
            
            # Adjust for volatility (high vol = higher premiums but more risk)
            volatility = abs(max_return - min_return)
            vol_adjustment = min(volatility, 0.3) / 0.3  # Normalize to 0-1
            
            # Higher volatility increases premium but also assignment risk
            if signal > 0:  # Bullish signals
                signal *= (1 + vol_adjustment * 0.2)  # Slight boost for vol
            else:  # Bearish signals
                signal *= (1 + vol_adjustment * 0.5)  # Penalty for vol
            
            # Bound signal
            signal = np.clip(signal, -1.0, 1.0)
            labels.append(signal)
        
        return np.array(labels)
    
    def sophisticated_fitness_function(self, genome, config, training_data) -> float:
        """Sophisticated fitness function optimized for profitable options trading."""
        try:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            features = training_data['features'].values
            labels = training_data['labels']
            prices = training_data['prices']
            
            # Generate predictions
            predictions = []
            for feature_row in features:
                output = net.activate(feature_row)
                predictions.append(output[0])
            
            predictions = np.array(predictions)
            
            # SOPHISTICATED CASH-SECURED PUT SIMULATION
            trades = []
            daily_returns = []
            position = 0
            entry_price = 0
            entry_confidence = 0
            days_in_position = 0
            max_position_days = 20  # 4 weeks max
            
            # Track comprehensive performance metrics
            wins = 0
            losses = 0
            assignments = 0
            total_trades = 0
            total_return = 0
            max_drawdown = 0
            peak_value = 1.0
            current_value = 1.0
            
            for i in range(len(predictions)):
                if i >= len(prices) - 1:
                    break
                
                prediction = predictions[i]
                current_price = prices[i]
                
                # ENTRY LOGIC - Cash-secured put strategy
                if position == 0 and prediction > 0.4:  # High confidence bullish signal
                    position = 1  # Long position (expecting stock to go up)
                    entry_price = current_price
                    entry_confidence = prediction
                    days_in_position = 0
                    
                    # Calculate strike price based on confidence
                    # Higher confidence = closer to current price (more premium)
                    strike_distance = 0.95 - (entry_confidence * 0.05)  # 90-95% of current price
                    strike_price = current_price * strike_distance
                
                # POSITION MANAGEMENT
                elif position != 0:
                    days_in_position += 1
                    
                    # Calculate current P&L for cash-secured put
                    # If stock goes up, we keep the premium (profit)
                    # If stock goes down below strike, we get assigned (loss)
                    
                    if current_price >= strike_price:
                        # Stock above strike - we keep premium
                        pnl = 0.015 * (entry_confidence * 0.5 + 0.5)  # 1-2% premium based on confidence
                    else:
                        # Stock below strike - assignment risk
                        assignment_loss = (strike_price - current_price) / current_price
                        pnl = -assignment_loss + 0.015  # Loss minus premium collected
                        assignments += 1
                    
                    # EXIT LOGIC - Multiple conditions
                    should_exit = (
                        days_in_position >= max_position_days or  # Time decay (theta)
                        pnl < -0.10 or  # Stop loss (10%)
                        pnl > 0.03 or   # Take profit (3%)
                        prediction < 0.2  # Confidence dropped significantly
                    )
                    
                    if should_exit:
                        # Account for transaction costs and slippage
                        net_pnl = pnl - 0.002  # 0.2% total transaction cost
                        trades.append(net_pnl)
                        total_trades += 1
                        
                        # Update performance tracking
                        current_value *= (1 + net_pnl)
                        total_return += net_pnl
                        
                        if net_pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                        
                        # Track drawdown
                        if current_value > peak_value:
                            peak_value = current_value
                        
                        drawdown = (peak_value - current_value) / peak_value
                        max_drawdown = max(max_drawdown, drawdown)
                        
                        # Reset position
                        position = 0
                        days_in_position = 0
                
                # Track daily returns for Sharpe calculation
                if i > 0 and len(trades) > 0:
                    daily_return = trades[-1] / max_position_days if trades else 0
                    daily_returns.append(daily_return)
            
            # COMPREHENSIVE FITNESS CALCULATION FOR PROFITABILITY
            if total_trades < 5:  # Need minimum trades for statistical significance
                return -1000
            
            # Calculate key profitability metrics
            win_rate = wins / total_trades if total_trades > 0 else 0
            assignment_rate = assignments / total_trades if total_trades > 0 else 0
            avg_return_per_trade = np.mean(trades) if trades else 0
            annual_return = avg_return_per_trade * 252 / max_position_days  # Annualized
            
            # Sharpe ratio (risk-adjusted returns)
            if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Calmar ratio (return / max drawdown)
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
            
            # MULTI-OBJECTIVE FITNESS OPTIMIZED FOR PROFITABILITY
            fitness_components = {
                # PRIMARY: Profitability metrics (heavily weighted)
                'win_rate_bonus': (win_rate - self.target_win_rate) * 2000,  # Massive weight
                'annual_return_bonus': (annual_return - self.target_annual_return) * 1000,
                'assignment_penalty': -(assignment_rate - self.target_assignment_rate) * 3000 if assignment_rate > self.target_assignment_rate else 0,
                
                # SECONDARY: Risk-adjusted metrics
                'sharpe_bonus': (sharpe_ratio - self.target_sharpe) * 500,
                'drawdown_penalty': -(max_drawdown - self.max_drawdown) * 1500 if max_drawdown > self.max_drawdown else 0,
                'calmar_bonus': calmar_ratio * 200,
                
                # TERTIARY: Trading activity and consistency
                'trade_count_bonus': min(total_trades - 5, 20) * 10,  # Bonus for sufficient trades
                'consistency_bonus': (1 - np.std(trades)) * 100 if trades else 0,
                'avg_trade_bonus': avg_return_per_trade * 1000
            }
            
            # Calculate total fitness
            total_fitness = sum(fitness_components.values())
            
            # STRICT PENALTIES for poor performance (ensure only profitable models)
            if (win_rate < 0.6 or 
                annual_return < 0.3 or 
                assignment_rate > 0.10 or 
                max_drawdown > 0.15):
                total_fitness -= 1000  # Heavy penalty
            
            # EXCEPTIONAL BONUS for meeting all targets
            if (win_rate >= self.target_win_rate and
                annual_return >= self.target_annual_return and
                assignment_rate <= self.target_assignment_rate and
                sharpe_ratio >= self.target_sharpe and
                max_drawdown <= self.max_drawdown):
                total_fitness += 500  # Exceptional bonus
            
            return total_fitness
            
        except Exception as e:
            self.logger.error(f"Fitness evaluation error: {e}")
            return -1000
    
    def create_neat_config(self, num_inputs: int) -> neat.Config:
        """Create NEAT configuration for proper training."""
        config_text = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1500
pop_size              = {self.population_size}
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh sigmoid relu

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.3
conn_delete_prob        = 0.3

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full_direct

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 3
num_inputs              = {num_inputs}
num_outputs             = 1

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = {self.max_stagnation}
species_elitism      = 2

[DefaultReproduction]
elitism            = 3
survival_threshold = 0.2
"""
        
        # Write config to temporary file
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
            f.write(config_text)
            config_path = f.name
        
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path)
        
        # Clean up temp file
        os.unlink(config_path)
        
        return config
    
    def train_profitable_agent(self, symbol: str, horizon_weeks: int = 4) -> Dict:
        """Train profitable NEAT agent with proper time and rigor."""
        start_time = time.time()
        self.logger.info(f"🎯 Training PROFITABLE agent for {symbol} (minimum {self.min_training_time/60:.1f} minutes)")
        
        try:
            # Fetch comprehensive data
            data = self.data_fetcher.fetch_data(symbol, "2y")
            features = self.feature_engine.engineer_all_features(data, symbol)
            labels = self.create_options_labels(data, horizon_weeks)
            prices = data['Close'].values
            
            # Ensure sufficient data
            min_samples = max(252, horizon_weeks * 5 * 10)  # At least 1 year
            if len(features) < min_samples:
                return {'success': False, 'error': f'Insufficient data: {len(features)} < {min_samples}'}
            
            # Align data
            min_length = min(len(features), len(labels), len(prices))
            features = features.iloc[:min_length]
            labels = labels[:min_length]
            prices = prices[:min_length]
            
            training_data = {
                'features': features,
                'labels': labels,
                'prices': prices
            }
            
            # Create NEAT configuration
            config = self.create_neat_config(len(features.columns))
            
            # Create population
            population = neat.Population(config)
            
            # Add reporters for monitoring
            population.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            population.add_reporter(stats)
            
            # Define fitness function with training data
            def fitness_func(genomes, config):
                for genome_id, genome in genomes:
                    genome.fitness = self.sophisticated_fitness_function(genome, config, training_data)
            
            # PROPER TRAINING - Run for sufficient time and generations
            self.logger.info(f"🚀 Starting evolution: {self.population_size} population, minimum {self.min_generations} generations")
            
            generation = 0
            best_fitness = -float('inf')
            stagnation_count = 0
            
            while True:
                # Run one generation
                winner = population.run(fitness_func, 1)
                generation += 1
                
                # Check progress
                current_best = max([g.fitness for g in population.population.values() if g.fitness is not None])
                if current_best > best_fitness:
                    best_fitness = current_best
                    stagnation_count = 0
                    self.logger.info(f"🎯 New best fitness: {best_fitness:.1f} (Generation {generation})")
                else:
                    stagnation_count += 1
                
                elapsed_time = time.time() - start_time
                
                # Log progress every 10 generations
                if generation % 10 == 0:
                    self.logger.info(f"📊 Generation {generation}: Best={best_fitness:.1f}, Time={elapsed_time:.1f}s, Stagnation={stagnation_count}")
                
                # Check termination conditions
                should_terminate = (
                    elapsed_time >= self.min_training_time and  # Minimum time
                    generation >= self.min_generations and     # Minimum generations
                    (best_fitness > 1200 or                    # Excellent fitness achieved
                     stagnation_count > self.max_stagnation or # No improvement
                     elapsed_time > self.max_training_time)    # Maximum time
                )
                
                if should_terminate:
                    break
            
            training_time = time.time() - start_time
            self.logger.info(f"⏱️ Training completed: {generation} generations in {training_time:.1f}s")
            
            # Evaluate final model
            if winner:
                final_fitness = self.sophisticated_fitness_function(winner, config, training_data)
                
                # Create comprehensive performance evaluation
                performance = self.evaluate_model_performance(winner, config, training_data, symbol)
                
                # CHECK EXCEPTIONAL STANDARDS - Only save if meets criteria
                meets_standards = (
                    performance['win_rate'] >= self.target_win_rate and
                    performance['annual_return'] >= self.target_annual_return and
                    performance['assignment_rate'] <= self.target_assignment_rate and
                    performance['sharpe_ratio'] >= self.target_sharpe and
                    performance['max_drawdown'] <= self.max_drawdown
                )
                
                if meets_standards:
                    # Save exceptional model
                    model_data = {
                        'genome': winner,
                        'config': config,
                        'performance': performance,
                        'training_time': training_time,
                        'generations': generation,
                        'symbol': symbol,
                        'horizon_weeks': horizon_weeks,
                        'saved_at': datetime.now().isoformat()
                    }
                    
                    self.models[symbol] = model_data
                    
                    # Save to disk
                    os.makedirs('models', exist_ok=True)
                    with open(f'models/{symbol}_profitable_model.pkl', 'wb') as f:
                        pickle.dump(model_data, f)
                    
                    self.logger.info(f"🏆 EXCEPTIONAL MODEL SAVED for {symbol}:")
                    self.logger.info(f"   Win Rate: {performance['win_rate']:.1%}")
                    self.logger.info(f"   Annual Return: {performance['annual_return']:.1%}")
                    self.logger.info(f"   Assignment Rate: {performance['assignment_rate']:.1%}")
                    self.logger.info(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
                    
                    return {
                        'success': True,
                        'exceptional': True,
                        'fitness': final_fitness,
                        'performance': performance,
                        'training_time': training_time,
                        'generations': generation
                    }
                else:
                    self.logger.warning(f"❌ MODEL REJECTED for {symbol} - Does not meet exceptional standards:")
                    self.logger.warning(f"   Win Rate: {performance['win_rate']:.1%} (need {self.target_win_rate:.1%})")
                    self.logger.warning(f"   Annual Return: {performance['annual_return']:.1%} (need {self.target_annual_return:.1%})")
                    self.logger.warning(f"   Assignment Rate: {performance['assignment_rate']:.1%} (max {self.target_assignment_rate:.1%})")
                    
                    return {
                        'success': False,
                        'exceptional': False,
                        'reason': 'Does not meet exceptional standards',
                        'performance': performance,
                        'training_time': training_time,
                        'generations': generation
                    }
            else:
                return {'success': False, 'error': 'No winner found'}
                
        except Exception as e:
            self.logger.error(f"❌ Training failed for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def evaluate_model_performance(self, genome, config, training_data, symbol: str) -> Dict:
        """Comprehensive model performance evaluation."""
        try:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            features = training_data['features'].values
            labels = training_data['labels']
            prices = training_data['prices']
            
            # Generate predictions
            predictions = []
            for feature_row in features:
                output = net.activate(feature_row)
                predictions.append(output[0])
            
            predictions = np.array(predictions)
            
            # Simulate comprehensive trading
            trades = []
            wins = 0
            losses = 0
            assignments = 0
            total_trades = 0
            max_drawdown = 0
            peak_value = 1.0
            current_value = 1.0
            
            position = 0
            entry_price = 0
            entry_confidence = 0
            days_in_position = 0
            
            for i in range(len(predictions)):
                if i >= len(prices) - 1:
                    break
                
                prediction = predictions[i]
                current_price = prices[i]
                
                if position == 0 and prediction > 0.4:
                    position = 1
                    entry_price = current_price
                    entry_confidence = prediction
                    days_in_position = 0
                    
                    strike_distance = 0.95 - (entry_confidence * 0.05)
                    strike_price = current_price * strike_distance
                
                elif position != 0:
                    days_in_position += 1
                    
                    if current_price >= strike_price:
                        pnl = 0.015 * (entry_confidence * 0.5 + 0.5)
                    else:
                        assignment_loss = (strike_price - current_price) / current_price
                        pnl = -assignment_loss + 0.015
                        assignments += 1
                    
                    if (days_in_position >= 20 or pnl < -0.10 or pnl > 0.03 or prediction < 0.2):
                        net_pnl = pnl - 0.002
                        trades.append(net_pnl)
                        total_trades += 1
                        
                        current_value *= (1 + net_pnl)
                        
                        if net_pnl > 0:
                            wins += 1
                        else:
                            losses += 1
                        
                        if current_value > peak_value:
                            peak_value = current_value
                        
                        drawdown = (peak_value - current_value) / peak_value
                        max_drawdown = max(max_drawdown, drawdown)
                        
                        position = 0
                        days_in_position = 0
            
            # Calculate comprehensive metrics
            win_rate = wins / total_trades if total_trades > 0 else 0
            assignment_rate = assignments / total_trades if total_trades > 0 else 0
            total_return = current_value - 1
            avg_trade = np.mean(trades) if trades else 0
            annual_return = avg_trade * 252 / 20 if trades else 0
            
            # Sharpe ratio
            if len(trades) > 1 and np.std(trades) > 0:
                sharpe_ratio = np.mean(trades) / np.std(trades) * np.sqrt(252/20)
            else:
                sharpe_ratio = 0
            
            # Buy and hold comparison
            buy_hold_return = (prices[-1] / prices[0]) - 1
            outperformance = annual_return - buy_hold_return
            
            return {
                'symbol': symbol,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'assignment_rate': assignment_rate,
                'total_return': total_return,
                'annual_return': annual_return,
                'avg_trade_return': avg_trade,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'buy_hold_return': buy_hold_return,
                'outperformance': outperformance,
                'calmar_ratio': annual_return / max_drawdown if max_drawdown > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Performance evaluation error: {e}")
            return {}


if __name__ == "__main__":
    # Test proper NEAT training system
    print("🔥 TESTING PROPER NEAT TRAINING SYSTEM")
    print("=" * 60)
    
    trainer = ProperNEATTrainingSystem()
    
    # Test with one ticker first
    test_ticker = 'AAPL'
    
    print(f"\n🎯 Training profitable model for {test_ticker}...")
    result = trainer.train_profitable_agent(test_ticker)
    
    if result['success']:
        print(f"✅ SUCCESS: {test_ticker} model trained successfully!")
        if result.get('exceptional'):
            print("🏆 EXCEPTIONAL MODEL - Meets all profitability standards!")
        else:
            print("⚠️ Model trained but rejected for not meeting standards")
    else:
        print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
    
    print(f"\n🏆 PROPER NEAT TRAINING TEST COMPLETE")
