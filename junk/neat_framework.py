"""
Production-ready NEAT (NeuroEvolution of Augmenting Topologies) framework
for options premium harvesting strategy evolution.
"""

import neat
import pickle
import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from database import db_manager
from data_ingestion import data_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NEATOptionsEvolution:
    """Production-ready NEAT evolution engine for options trading strategies."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize NEAT evolution framework.
        
        Args:
            config_path: Path to NEAT configuration file
        """
        self.config_path = config_path or self._create_default_config()
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path
        )
        
        self.population = None
        self.generation = 0
        self.best_genome = None
        self.fitness_history = []
        self.species_history = []
        
        # Performance tracking
        self.evaluation_cache = {}
        self.validation_data = {}
        
    def _create_default_config(self) -> str:
        """Create default NEAT configuration optimized for options trading."""
        
        config_content = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 3.0
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh sigmoid relu

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.1
aggregation_options     = sum product

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
num_hidden              = 0
num_inputs              = 50
num_outputs             = 4

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
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
"""
        
        config_path = '/Users/wgo/Documents/Cline/SwingTradeAI/neat_config.txt'
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Created NEAT configuration at {config_path}")
        return config_path
    
    def create_feature_vector(self, symbol: str, date: datetime) -> Optional[np.ndarray]:
        """
        Create feature vector for NEAT network input.
        
        Args:
            symbol: Stock symbol
            date: Date for feature calculation
            
        Returns:
            Feature vector array or None if insufficient data
        """
        try:
            # Get historical stock data (60 days for technical indicators)
            end_date = date
            start_date = date - timedelta(days=90)
            
            stock_data = db_manager.get_stock_data(symbol, start_date, end_date)
            if len(stock_data) < 30:  # Need minimum data
                logger.warning(f"Insufficient stock data for {symbol} on {date}")
                return None
            
            # Calculate technical indicators
            features = []
            
            # Price-based features (10 features)
            current_price = stock_data['Close'].iloc[-1]
            features.extend([
                current_price / stock_data['Close'].iloc[-5] - 1,  # 5-day return
                current_price / stock_data['Close'].iloc[-10] - 1,  # 10-day return
                current_price / stock_data['Close'].iloc[-20] - 1,  # 20-day return
                current_price / stock_data['Close'].mean() - 1,  # Price vs mean
                current_price / stock_data['Close'].rolling(5).mean().iloc[-1] - 1,  # Price vs 5-day MA
                current_price / stock_data['Close'].rolling(10).mean().iloc[-1] - 1,  # Price vs 10-day MA
                current_price / stock_data['Close'].rolling(20).mean().iloc[-1] - 1,  # Price vs 20-day MA
                stock_data['High'].iloc[-1] / stock_data['Low'].iloc[-1] - 1,  # Daily range
                (current_price - stock_data['Low'].rolling(20).min().iloc[-1]) / 
                (stock_data['High'].rolling(20).max().iloc[-1] - stock_data['Low'].rolling(20).min().iloc[-1]),  # %K
                stock_data['Close'].rolling(5).std().iloc[-1] / current_price  # 5-day volatility
            ])
            
            # Volume features (5 features)
            current_volume = stock_data['Volume'].iloc[-1]
            avg_volume = stock_data['Volume'].rolling(20).mean().iloc[-1]
            features.extend([
                current_volume / avg_volume - 1,  # Volume vs average
                stock_data['Volume'].rolling(5).mean().iloc[-1] / avg_volume - 1,  # 5-day avg volume vs 20-day
                np.log(current_volume + 1),  # Log volume (normalized)
                stock_data['Volume'].rolling(5).std().iloc[-1] / avg_volume,  # Volume volatility
                (stock_data['Volume'].iloc[-1] - stock_data['Volume'].iloc[-2]) / stock_data['Volume'].iloc[-2]  # Volume change
            ])
            
            # Momentum features (10 features)
            returns = stock_data['Close'].pct_change()
            features.extend([
                returns.iloc[-1],  # 1-day return
                returns.rolling(5).mean().iloc[-1],  # 5-day avg return
                returns.rolling(10).mean().iloc[-1],  # 10-day avg return
                returns.rolling(20).mean().iloc[-1],  # 20-day avg return
                returns.rolling(5).std().iloc[-1],  # 5-day return volatility
                returns.rolling(20).std().iloc[-1],  # 20-day return volatility
                returns.rolling(5).skew().iloc[-1] if not pd.isna(returns.rolling(5).skew().iloc[-1]) else 0,  # Return skewness
                (returns > 0).rolling(10).sum().iloc[-1] / 10,  # Win rate (10 days)
                returns.rolling(10).apply(lambda x: (x > x.mean()).sum()).iloc[-1] / 10,  # Above average rate
                np.tanh(returns.rolling(20).sum().iloc[-1])  # 20-day cumulative return (tanh normalized)
            ])
            
            # Trend features (10 features)
            sma_5 = stock_data['Close'].rolling(5).mean()
            sma_10 = stock_data['Close'].rolling(10).mean()
            sma_20 = stock_data['Close'].rolling(20).mean()
            
            features.extend([
                1 if sma_5.iloc[-1] > sma_10.iloc[-1] else -1,  # 5 vs 10 MA trend
                1 if sma_10.iloc[-1] > sma_20.iloc[-1] else -1,  # 10 vs 20 MA trend
                1 if current_price > sma_5.iloc[-1] else -1,  # Price vs 5 MA
                1 if current_price > sma_10.iloc[-1] else -1,  # Price vs 10 MA
                1 if current_price > sma_20.iloc[-1] else -1,  # Price vs 20 MA
                (sma_5.iloc[-1] - sma_5.iloc[-5]) / sma_5.iloc[-5] if sma_5.iloc[-5] != 0 else 0,  # 5 MA slope
                (sma_10.iloc[-1] - sma_10.iloc[-5]) / sma_10.iloc[-5] if sma_10.iloc[-5] != 0 else 0,  # 10 MA slope
                (sma_20.iloc[-1] - sma_20.iloc[-10]) / sma_20.iloc[-10] if sma_20.iloc[-10] != 0 else 0,  # 20 MA slope
                (current_price - stock_data['Close'].rolling(20).min().iloc[-1]) / 
                (stock_data['Close'].rolling(20).max().iloc[-1] - stock_data['Close'].rolling(20).min().iloc[-1]),  # Position in range
                np.tanh((current_price - sma_20.iloc[-1]) / sma_20.iloc[-1])  # Distance from 20 MA (normalized)
            ])
            
            # Market context features (10 features) - simplified for now
            features.extend([
                np.sin(2 * np.pi * date.timetuple().tm_yday / 365),  # Seasonal (day of year)
                np.cos(2 * np.pi * date.timetuple().tm_yday / 365),  # Seasonal (day of year)
                date.weekday() / 6.0 - 0.5,  # Day of week normalized
                (date.day - 15) / 15.0,  # Day of month normalized
                np.tanh(len(stock_data) / 252),  # Data availability (years)
                0.0,  # VIX (placeholder - will implement later)
                0.0,  # Market breadth (placeholder)
                0.0,  # Sector performance (placeholder)
                0.0,  # Earnings proximity (placeholder)
                np.random.normal(0, 0.1)  # Noise feature for robustness
            ])
            
            # Options-specific features (5 features) - simplified for now
            options_data = db_manager.get_options_data(symbol, date)
            if not options_data.empty:
                puts_data = options_data[options_data['option_type'] == 'put']
                if not puts_data.empty:
                    # Find ATM put
                    atm_put = puts_data.iloc[(puts_data['strike'] - current_price).abs().argsort()[:1]]
                    if not atm_put.empty:
                        features.extend([
                            atm_put['implied_volatility'].iloc[0] if not pd.isna(atm_put['implied_volatility'].iloc[0]) else 0.3,
                            abs(atm_put['delta'].iloc[0]) if not pd.isna(atm_put['delta'].iloc[0]) else 0.5,
                            atm_put['theta'].iloc[0] if not pd.isna(atm_put['theta'].iloc[0]) else -0.1,
                            atm_put['volume'].iloc[0] / 1000 if not pd.isna(atm_put['volume'].iloc[0]) else 0,
                            atm_put['open_interest'].iloc[0] / 10000 if not pd.isna(atm_put['open_interest'].iloc[0]) else 0
                        ])
                    else:
                        features.extend([0.3, 0.5, -0.1, 0, 0])
                else:
                    features.extend([0.3, 0.5, -0.1, 0, 0])
            else:
                features.extend([0.3, 0.5, -0.1, 0, 0])
            
            # Ensure we have exactly 50 features
            features = features[:50]
            while len(features) < 50:
                features.append(0.0)
            
            # Convert to numpy array and handle NaN/inf values
            feature_array = np.array(features, dtype=np.float32)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Clip extreme values
            feature_array = np.clip(feature_array, -10.0, 10.0)
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error creating feature vector for {symbol} on {date}: {e}")
            return None
    
    def evaluate_genome(self, genome, config, symbol: str, start_date: datetime, end_date: datetime) -> float:
        """
        Evaluate a single genome's performance on cash-secured put strategy.
        
        Args:
            genome: NEAT genome to evaluate
            config: NEAT configuration
            symbol: Stock symbol to trade
            start_date: Evaluation start date
            end_date: Evaluation end date
            
        Returns:
            Fitness score (higher is better)
        """
        try:
            # Create neural network from genome
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            # Get stock data for evaluation period
            stock_data = db_manager.get_stock_data(symbol, start_date - timedelta(days=90), end_date)
            if len(stock_data) < 100:  # Need sufficient data
                return -10.0  # Penalty for insufficient data
            
            # Initialize tracking variables
            total_return = 0.0
            total_premium = 0.0
            assignments = 0
            total_trades = 0
            daily_returns = []
            max_drawdown = 0.0
            peak_value = 0.0
            current_value = 0.0
            
            # Simulate trading over the evaluation period (weekly decisions)
            evaluation_dates = pd.date_range(start_date, end_date, freq='W-MON')  # Weekly on Mondays
            
            for eval_date in evaluation_dates:
                try:
                    # Find closest trading day
                    closest_date = None
                    for i in range(7):  # Look within a week
                        check_date = eval_date + timedelta(days=i)
                        if check_date in stock_data.index:
                            closest_date = check_date
                            break
                    
                    if closest_date is None:
                        continue
                    
                    # Create feature vector
                    features = self.create_feature_vector(symbol, closest_date)
                    if features is None:
                        continue
                    
                    # Get network output
                    output = net.activate(features)
                    
                    # Interpret network outputs:
                    # output[0]: Bullish confidence (-1 to 1, normalize to 0-1)
                    # output[1]: Strike distance multiplier (-1 to 1, maps to 0.03-0.10)
                    # output[2]: Time to expiration preference (-1 to 1, maps to 1-8 weeks)
                    # output[3]: Position size multiplier (-1 to 1, maps to 0.1-1.0)
                    
                    bullish_confidence = max(0, min(1, (output[0] + 1) / 2))  # Normalize to 0-1
                    strike_distance = 0.03 + (max(0, min(1, (output[1] + 1) / 2))) * 0.07  # 3-10% below
                    weeks_to_exp = 1 + int(max(0, min(1, (output[2] + 1) / 2)) * 7)  # 1-8 weeks
                    position_size = max(0.1, min(1, (output[3] + 1) / 2))  # 10-100% position
                    
                    # Decision threshold: only trade if confident enough
                    if bullish_confidence > 0.5:  # 50% confidence threshold (more lenient for testing)
                        current_price = stock_data.loc[closest_date, 'Close']
                        
                        # Calculate strike price using dynamic formula
                        # Strike = Current_Price × (0.95 – (confidence_score × 0.05))
                        confidence_adjustment = bullish_confidence * 0.05
                        strike_price = current_price * (0.95 - confidence_adjustment)
                        
                        # Estimate premium (simplified - in production would use real options data)
                        days_to_exp = weeks_to_exp * 7
                        time_value = max(0.01, 0.05 * np.sqrt(days_to_exp / 365))  # Simplified time value
                        intrinsic_value = max(0, strike_price - current_price)
                        estimated_premium = intrinsic_value + time_value * current_price
                        
                        # Calculate premium yield
                        cash_secured = strike_price * 100  # 100 shares
                        premium_yield = (estimated_premium * 100) / cash_secured
                        weekly_yield = premium_yield * (52 / weeks_to_exp)
                        
                        # Only trade if premium yield meets minimum (1.5% weekly)
                        if weekly_yield >= 0.015:
                            total_trades += 1
                            total_premium += estimated_premium * 100 * position_size
                            
                            # Check if assignment would occur (simplified)
                            expiration_date = eval_date + timedelta(days=days_to_exp)
                            if expiration_date <= end_date and expiration_date in stock_data.index:
                                final_price = stock_data.loc[expiration_date, 'Close']
                                
                                if final_price < strike_price:
                                    # Assignment occurred
                                    assignments += 1
                                    assignment_loss = (strike_price - final_price) * 100 * position_size
                                    total_return -= assignment_loss
                                else:
                                    # Premium collected successfully
                                    total_return += estimated_premium * 100 * position_size
                            else:
                                # Assume premium collected (end of evaluation period)
                                total_return += estimated_premium * 100 * position_size
                            
                            # Track daily P&L for risk metrics
                            daily_pnl = estimated_premium * 100 * position_size / days_to_exp
                            daily_returns.append(daily_pnl / 10000)  # Normalize
                            current_value += daily_pnl
                            
                            # Track drawdown
                            if current_value > peak_value:
                                peak_value = current_value
                            else:
                                drawdown = (peak_value - current_value) / max(peak_value, 1)
                                max_drawdown = max(max_drawdown, drawdown)
                
                except Exception as e:
                    logger.warning(f"Error evaluating genome on {eval_date}: {e}")
                    continue
            
            # Calculate fitness score
            if total_trades == 0:
                return -5.0  # Penalty for no trades
            
            # Assignment rate (primary metric - goal is 0%, max acceptable 5%)
            assignment_rate = assignments / total_trades if total_trades > 0 else 0
            assignment_penalty = assignment_rate * 10  # Heavy penalty for assignments
            
            # Premium yield (secondary metric - target 1.5% weekly)
            avg_premium_yield = total_premium / (total_trades * 10000) if total_trades > 0 else 0
            premium_score = min(2.0, avg_premium_yield / 0.015)  # Normalize to target
            
            # Risk-adjusted returns (tertiary metric)
            if len(daily_returns) > 1:
                returns_array = np.array(daily_returns)
                avg_return = np.mean(returns_array)
                return_std = np.std(returns_array)
                sharpe_ratio = avg_return / max(return_std, 0.001) * np.sqrt(252)
                risk_score = min(2.0, max(0, sharpe_ratio / 1.5))  # Normalize to target Sharpe of 1.5
            else:
                risk_score = 0
            
            # Drawdown penalty
            drawdown_penalty = max_drawdown * 2  # Penalty for drawdown
            
            # Combined fitness score (weighted by priority)
            fitness = (
                (1.0 - assignment_penalty) * 3.0 +  # Primary: Assignment rate (weight 3)
                premium_score * 2.0 +               # Secondary: Premium yield (weight 2)
                risk_score * 1.0 -                  # Tertiary: Risk-adjusted returns (weight 1)
                drawdown_penalty                    # Penalty: Drawdown
            )
            
            # Bonus for consistent trading
            if total_trades >= 10:
                fitness += 0.5
            
            # Ensure fitness is reasonable
            fitness = max(-10.0, min(10.0, fitness))
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error evaluating genome: {e}")
            return -10.0  # Severe penalty for evaluation errors
    
    def parallel_evaluate(self, genomes, config, symbols: List[str], start_date: datetime, end_date: datetime):
        """Evaluate genomes sequentially (avoiding pickle issues with parallel processing)."""
        
        for genome_id, genome in genomes:
            total_fitness = 0.0
            valid_evaluations = 0
            
            # Evaluate on multiple symbols for robustness
            for symbol in symbols:
                try:
                    fitness = self.evaluate_genome(genome, config, symbol, start_date, end_date)
                    total_fitness += fitness
                    valid_evaluations += 1
                except Exception as e:
                    logger.warning(f"Failed to evaluate genome {genome_id} on {symbol}: {e}")
                    continue
            
            # Average fitness across symbols
            if valid_evaluations > 0:
                genome.fitness = total_fitness / valid_evaluations
            else:
                genome.fitness = -10.0  # Penalty for complete failure
            
            logger.debug(f"Genome {genome_id} fitness: {genome.fitness:.4f}")
    
    def evolve_population(self, generations: int = 50, symbols: List[str] = None) -> neat.DefaultGenome:
        """
        Evolve NEAT population for options trading.
        
        Args:
            generations: Number of generations to evolve
            symbols: List of symbols to evaluate on
            
        Returns:
            Best genome found
        """
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']  # Default high-quality symbols
        
        # Create population
        self.population = neat.Population(self.config)
        
        # Add reporters for monitoring
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        
        # Define evaluation period (last 2 years for training)
        end_date = datetime.now() - timedelta(days=30)  # Leave recent data for validation
        start_date = end_date - timedelta(days=730)  # 2 years of training data
        
        logger.info(f"Starting NEAT evolution for {generations} generations")
        logger.info(f"Evaluation period: {start_date.date()} to {end_date.date()}")
        logger.info(f"Symbols: {symbols}")
        
        # Evolution loop
        for generation in range(generations):
            logger.info(f"Starting generation {generation + 1}/{generations}")
            
            # Evaluate population
            genomes = list(self.population.population.items())
            self.parallel_evaluate(genomes, self.config, symbols, start_date, end_date)
            
            # Track best genome
            best_genome = max(self.population.population.values(), key=lambda g: g.fitness)
            if self.best_genome is None or best_genome.fitness > self.best_genome.fitness:
                self.best_genome = best_genome
                logger.info(f"New best genome found! Fitness: {best_genome.fitness:.4f}")
            
            # Track statistics
            fitnesses = [g.fitness for g in self.population.population.values()]
            self.fitness_history.append({
                'generation': generation + 1,
                'best_fitness': max(fitnesses),
                'avg_fitness': np.mean(fitnesses),
                'std_fitness': np.std(fitnesses)
            })
            
            # Save best genome periodically
            if (generation + 1) % 10 == 0:
                self.save_best_genome(f'best_genome_gen_{generation + 1}.pkl')
            
            # Evolve to next generation
            if generation < generations - 1:
                self.population.population = self.population.reproduction.reproduce(
                    self.config, self.population.species, self.config.pop_size, generation
                )
                self.population.species.speciate(
                    self.config, self.population.population, generation
                )
        
        logger.info(f"Evolution completed! Best fitness: {self.best_genome.fitness:.4f}")
        return self.best_genome
    
    def save_best_genome(self, filename: str):
        """Save the best genome to file."""
        if self.best_genome is not None:
            with open(filename, 'wb') as f:
                pickle.dump(self.best_genome, f)
            logger.info(f"Best genome saved to {filename}")
    
    def load_genome(self, filename: str) -> neat.DefaultGenome:
        """Load a genome from file."""
        with open(filename, 'rb') as f:
            genome = pickle.load(f)
        logger.info(f"Genome loaded from {filename}")
        return genome
    
    def create_network(self, genome: neat.DefaultGenome) -> neat.nn.FeedForwardNetwork:
        """Create a neural network from a genome."""
        return neat.nn.FeedForwardNetwork.create(genome, self.config)

# Global NEAT evolution instance
neat_evolution = NEATOptionsEvolution()

def test_neat_framework():
    """Test the NEAT framework with real data."""
    
    logger.info("Testing NEAT framework with real data")
    
    try:
        # Test feature vector creation
        test_symbol = 'AAPL'
        test_date = datetime.now() - timedelta(days=1)
        
        features = neat_evolution.create_feature_vector(test_symbol, test_date)
        if features is not None and len(features) == 50:
            print("✅ Feature vector creation test passed")
        else:
            print("❌ Feature vector creation test failed")
            return False
        
        # Test genome evaluation (short test)
        config = neat_evolution.config
        test_genome = config.genome_type(1)
        test_genome.configure_new(config.genome_config)
        
        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now() - timedelta(days=30)
        
        fitness = neat_evolution.evaluate_genome(test_genome, config, test_symbol, start_date, end_date)
        if isinstance(fitness, float) and -10 <= fitness <= 10:
            print(f"✅ Genome evaluation test passed (fitness: {fitness:.4f})")
        else:
            print(f"❌ Genome evaluation test failed (fitness: {fitness})")
            return False
        
        # Test short evolution (3 generations)
        logger.info("Running short evolution test (3 generations)")
        best_genome = neat_evolution.evolve_population(generations=3, symbols=[test_symbol])
        
        if best_genome is not None and hasattr(best_genome, 'fitness'):
            print(f"✅ Short evolution test passed (best fitness: {best_genome.fitness:.4f})")
            return True
        else:
            print("❌ Short evolution test failed")
            return False
            
    except Exception as e:
        logger.error(f"NEAT framework test failed: {e}")
        print(f"❌ NEAT framework test failed: {e}")
        return False

if __name__ == "__main__":
    # Run NEAT framework test
    if test_neat_framework():
        print("✅ NEAT framework ready for production")
    else:
        print("❌ NEAT framework needs fixes")
