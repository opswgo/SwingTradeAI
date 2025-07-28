"""
Agent Training Pipeline - NEAT Options System
Train individual NEAT agents for each selected stock with profit-focused fitness.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import neat
import json
import time
import tempfile
import os
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime, timedelta

from enhanced_data_pipeline import EnhancedDataPipeline
from src.database.agent_database import AgentDatabase
from comprehensive_agent_evaluator import ComprehensiveAgentEvaluator, TradingMetrics, HorizonResults
from enhanced_output_generator import EnhancedOutputGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingResult:
    """Training result for a single stock agent."""
    symbol: str
    success: bool
    best_fitness: float
    generations_completed: int
    training_time: float
    agent_id: str
    performance_metrics: Dict
    error_message: Optional[str] = None

class AgentTrainingPipeline:
    """Pipeline for training NEAT agents for multiple stocks."""
    
    def __init__(self, db_path: str = "data/neat_agents.db"):
        """Initialize training pipeline."""
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.agent_db = AgentDatabase(db_path)
        self.data_pipeline = EnhancedDataPipeline()
        self.evaluator = ComprehensiveAgentEvaluator(self.logger)
        self.output_generator = EnhancedOutputGenerator()
        
        # Training configuration
        self.training_config = {
            'population_size': 150,
            'max_generations': 100,
            'fitness_threshold': 100.0,
            'data_years': 5,
            'validation_split': 0.2,
            'test_split': 0.1,
            'early_stopping_patience': 20,
            'min_accuracy_threshold': 0.80,
            'profit_weight': 0.7,
            'risk_penalty': 0.1
        }
        
        self.logger = logging.getLogger(__name__)
    
    def load_stock_selection(self, selection_file: str = None) -> Dict:
        """Load stock selection results."""
        if selection_file is None:
            import glob
            selection_files = glob.glob("data/stock_selection_*.json")
            if not selection_files:
                raise FileNotFoundError("No stock selection file found")
            selection_file = max(selection_files)
        
        with open(selection_file, 'r') as f:
            selection_data = json.load(f)
        
        self.logger.info(f"📄 Loaded stock selection from: {selection_file}")
        return selection_data
    
    def prepare_training_data(self, symbol: str, years: int = 5) -> Dict:
        """Prepare training data for a single stock."""
        try:
            self.logger.info(f"📊 Preparing training data for {symbol}...")
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            data = self.data_pipeline.collect_extended_data(symbol, years=7)
            
            if data.empty or len(data) < 500:
                raise ValueError(f"Insufficient data for {symbol}: {len(data)} rows")
            
            # Feature engineering
            features = self.data_pipeline.engineer_features(data)
            
            if features.empty:
                raise ValueError(f"Feature engineering failed for {symbol}")
            
            # Create labels for options trading
            labels = self._create_options_labels(data, symbol)
            
            # Split data
            train_size = int(len(features) * (1 - self.training_config['validation_split'] - self.training_config['test_split']))
            val_size = int(len(features) * self.training_config['validation_split'])
            
            train_features = features.iloc[:train_size]
            train_labels = labels.iloc[:train_size]
            val_features = features.iloc[train_size:train_size + val_size]
            val_labels = labels.iloc[train_size:train_size + val_size]
            test_features = features.iloc[train_size + val_size:]
            test_labels = labels.iloc[train_size + val_size:]
            
            training_data = {
                'symbol': symbol,
                'train': {'features': train_features, 'labels': train_labels},
                'validation': {'features': val_features, 'labels': val_labels},
                'test': {'features': test_features, 'labels': test_labels},
                'raw_data': data,
                'feature_names': list(features.columns),
                'data_period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            }
            
            self.logger.info(f"✅ {symbol}: Prepared {len(features)} samples, {len(features.columns)} features")
            return training_data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare data for {symbol}: {str(e)}")
            raise
    
    def _create_options_labels(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Create multi-horizon labels for options trading strategy (1-8 weeks)."""
        horizons = [5, 10, 15, 20, 25, 30, 35, 40]  # 1-8 weeks
        horizon_weights = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]  # Longer horizons weighted more
        
        labels = []
        
        for i in range(len(data)):
            if i >= len(data) - max(horizons):
                labels.append(0)  # Neutral for insufficient data
                continue
            
            current_price = data['Close'].iloc[i]
            horizon_signals = []
            
            # Calculate signals for each horizon
            for j, horizon in enumerate(horizons):
                if i + horizon < len(data):
                    future_price = data['Close'].iloc[i + horizon]
                    return_pct = (future_price / current_price) - 1
                    
                    # Direction classification based on return thresholds
                    if return_pct > 0.03:  # 3%+ gain - strong buy signal (sell puts)
                        horizon_signals.append(2)
                    elif return_pct > 0.01:  # 1-3% gain - weak buy signal
                        horizon_signals.append(1)
                    elif return_pct < -0.03:  # 3%+ loss - strong sell signal
                        horizon_signals.append(-2)
                    elif return_pct < -0.01:  # 1-3% loss - weak sell signal
                        horizon_signals.append(-1)
                    else:  # -1% to +1% - neutral
                        horizon_signals.append(0)
            
            if horizon_signals:
                # Weighted average of signals (longer horizons matter more)
                weights = np.array(horizon_weights[:len(horizon_signals)])
                weighted_signal = np.average(horizon_signals, weights=weights)
                
                # Final direction classification
                if weighted_signal > 1.0:
                    labels.append(2)  # Strong buy
                elif weighted_signal > 0.3:
                    labels.append(1)  # Weak buy
                elif weighted_signal < -1.0:
                    labels.append(-2)  # Strong sell
                elif weighted_signal < -0.3:
                    labels.append(-1)  # Weak sell
                else:
                    labels.append(0)  # Neutral
            else:
                labels.append(0)
        
        return pd.Series(labels, index=data.index)
    
    def create_neat_config(self, symbol: str, num_features: int) -> neat.Config:
        """Create NEAT configuration for stock-specific training."""
        config_text = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = {self.training_config['fitness_threshold']}
pop_size              = {self.training_config['population_size']}
reset_on_extinction   = False

[DefaultGenome]
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh sigmoid relu
aggregation_default     = sum
aggregation_mutate_rate = 0.1
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.3
conn_delete_prob        = 0.3
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full
node_add_prob           = 0.2
node_delete_prob        = 0.2
num_hidden              = 0
num_inputs              = {num_features}
num_outputs             = 5
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
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
"""
        
        config_path = f"/tmp/neat_config_{symbol}_{os.getpid()}.txt"
        with open(config_path, 'w') as f:
            f.write(config_text)
        
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_path)
    
    def fitness_function(self, genomes, config, training_data: Dict):
        """Fitness function optimized for options trading profitability."""
        symbol = training_data['symbol']
        train_features = training_data['train']['features'].values
        train_labels = training_data['train']['labels'].values
        val_features = training_data['validation']['features'].values
        val_labels = training_data['validation']['labels'].values
        
        for genome_id, genome in genomes:
            try:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                
                # Training predictions
                train_predictions = []
                for features in train_features:
                    output = net.activate(features)
                    predicted_class = np.argmax(output) - 2
                    train_predictions.append(predicted_class)
                
                # Validation predictions
                val_predictions = []
                for features in val_features:
                    output = net.activate(features)
                    predicted_class = np.argmax(output) - 2
                    val_predictions.append(predicted_class)
                
                # Calculate accuracy
                train_accuracy = np.mean(np.array(train_predictions) == train_labels)
                val_accuracy = np.mean(np.array(val_predictions) == val_labels)
                
                # Simulate options trading performance
                profit_score = self._simulate_options_trading(val_predictions, val_labels)
                risk_score = self._calculate_risk_metrics(val_predictions, val_labels)
                
                # Composite fitness score
                accuracy_score = (train_accuracy + val_accuracy) / 2 * 100
                overfitting_penalty = max(0, (train_accuracy - val_accuracy) * 50)
                
                fitness = (
                    accuracy_score * (1 - self.training_config['profit_weight']) +
                    profit_score * self.training_config['profit_weight'] -
                    risk_score * self.training_config['risk_penalty'] -
                    overfitting_penalty
                )
                
                genome.fitness = max(0, fitness)
                
            except Exception as e:
                self.logger.warning(f"Fitness evaluation failed for genome {genome_id}: {str(e)}")
                genome.fitness = 0.0
    
    def _simulate_options_trading(self, predictions: List[int], actual: List[int]) -> float:
        """Simulate options trading based on predictions."""
        if len(predictions) != len(actual):
            return 0.0
        
        total_return = 0.0
        trades = 0
        winning_trades = 0
        
        for pred, actual_label in zip(predictions, actual):
            if pred > 0:  # Bullish prediction - sell puts
                trades += 1
                
                if actual_label >= 0:  # Stock went up or sideways
                    premium = 0.015 * min(pred, 2)  # 1.5% premium
                    total_return += premium
                    winning_trades += 1
                else:  # Stock went down
                    if actual_label == -2:  # Strong down move
                        loss = 0.05 * abs(actual_label)
                        total_return -= loss
                    else:  # Mild down move
                        total_return += 0.005
                        winning_trades += 1
        
        if trades == 0:
            return 0.0
        
        avg_return = total_return / trades
        win_rate = winning_trades / trades
        profit_score = avg_return * 1000 + win_rate * 50
        
        return max(0, profit_score)
    
    def _calculate_risk_metrics(self, predictions: List[int], actual: List[int]) -> float:
        """Calculate risk penalty score."""
        if len(predictions) != len(actual):
            return 100.0
        
        pred_volatility = np.std(predictions) if len(predictions) > 1 else 0
        
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for pred, act in zip(predictions, actual):
            if pred > 0 and act < 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        risk_score = pred_volatility * 10 + max_consecutive_losses * 5
        return risk_score
    
    def train_single_agent(self, symbol: str, training_data: Dict, 
                          generations: int = None) -> TrainingResult:
        """Train a single NEAT agent for a stock."""
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Starting training for {symbol}...")
            
            if generations is None:
                generations = self.training_config['max_generations']
            
            num_features = len(training_data['feature_names'])
            config = self.create_neat_config(symbol, num_features)
            
            population = neat.Population(config)
            population.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            population.add_reporter(stats)
            
            best_genome = None
            best_fitness = 0
            generations_without_improvement = 0
            
            for generation in range(generations):
                fitness_func = lambda genomes, config: self.fitness_function(
                    genomes, config, training_data
                )
                
                best_genome_gen = population.run(fitness_func, 1)
                
                if best_genome_gen.fitness > best_fitness:
                    best_fitness = best_genome_gen.fitness
                    best_genome = best_genome_gen
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                if generations_without_improvement >= self.training_config['early_stopping_patience']:
                    self.logger.info(f"Early stopping for {symbol} at generation {generation}")
                    break
                
                if best_fitness >= self.training_config['fitness_threshold']:
                    self.logger.info(f"Fitness threshold reached for {symbol} at generation {generation}")
                    break
            
            # Evaluate final performance
            performance_metrics = self._evaluate_agent_performance(
                best_genome, config, training_data
            )
            
            # Save agent to database
            agent_id = self._save_trained_agent(
                symbol, best_genome, config, performance_metrics, training_data
            )
            
            training_time = time.time() - start_time
            
            result = TrainingResult(
                symbol=symbol,
                success=True,
                best_fitness=best_fitness,
                generations_completed=generation + 1,
                training_time=training_time,
                agent_id=agent_id,
                performance_metrics=performance_metrics
            )
            
            self.logger.info(f"✅ {symbol}: Training completed in {training_time:.1f}s, "
                           f"fitness: {best_fitness:.2f}, accuracy: {performance_metrics['accuracy']:.1%}")
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"❌ {symbol}: Training failed after {training_time:.1f}s: {str(e)}")
            
            return TrainingResult(
                symbol=symbol,
                success=False,
                best_fitness=0,
                generations_completed=0,
                training_time=training_time,
                agent_id="",
                performance_metrics={},
                error_message=str(e)
            )
    
    def _evaluate_agent_performance(self, genome, config, training_data: Dict) -> Dict:
        """Comprehensive agent performance evaluation with backtesting and forward testing."""
        try:
            # Use comprehensive evaluator for full analysis
            trading_metrics, horizon_results = self.evaluator.evaluate_agent_comprehensive(
                genome, config, training_data
            )
            
            # Generate performance report
            symbol = training_data['symbol']
            report = self.evaluator.generate_performance_report(
                symbol, trading_metrics, horizon_results
            )
            
            # Log comprehensive results
            self.logger.info(f"\n{report}")
            
            # Check if agent meets performance threshold
            meets_threshold = self.evaluator.meets_performance_threshold(trading_metrics)
            
            if not meets_threshold:
                self.logger.warning(
                    f"⚠️ {symbol}: Agent does not meet 50%+ outperformance threshold "
                    f"(Excess return: {trading_metrics.excess_return_pct:.1%})"
                )
            
            # Return comprehensive metrics as dictionary
            return {
                'total_return': trading_metrics.total_return,
                'buy_hold_return': trading_metrics.buy_hold_return,
                'excess_return': trading_metrics.excess_return,
                'excess_return_pct': trading_metrics.excess_return_pct,
                'max_drawdown': trading_metrics.max_drawdown,
                'current_drawdown': trading_metrics.current_drawdown,
                'volatility': trading_metrics.volatility,
                'sharpe_ratio': trading_metrics.sharpe_ratio,
                'win_rate': trading_metrics.win_rate,
                'total_trades': trading_metrics.total_trades,
                'winning_trades': trading_metrics.winning_trades,
                'losing_trades': trading_metrics.losing_trades,
                'avg_win': trading_metrics.avg_win,
                'avg_loss': trading_metrics.avg_loss,
                'profit_factor': trading_metrics.profit_factor,
                'lookahead_days': trading_metrics.lookahead_days,
                'lookahead_weeks': trading_metrics.lookahead_weeks,
                'direction_accuracy': trading_metrics.direction_accuracy,
                'price_accuracy': trading_metrics.price_accuracy,
                'fitness_score': trading_metrics.fitness_score,
                'fitness_interpretation': trading_metrics.fitness_interpretation,
                'meets_threshold': meets_threshold,
                'horizon_results': {k: asdict(v) for k, v in horizon_results.items()},
                'accuracy': trading_metrics.direction_accuracy,  # Legacy compatibility
                'validation_accuracy': trading_metrics.direction_accuracy
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive evaluation failed: {str(e)}")
            return self._default_performance_metrics()
    
    def _default_performance_metrics(self) -> Dict:
        """Return default performance metrics when evaluation fails."""
        return {
            'total_return': 0.0,
            'buy_hold_return': 0.0,
            'excess_return': 0.0,
            'excess_return_pct': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'lookahead_days': 0,
            'lookahead_weeks': 0.0,
            'direction_accuracy': 0.0,
            'price_accuracy': 0.0,
            'fitness_score': 0.1,
            'fitness_interpretation': 'Failed',
            'meets_threshold': False,
            'horizon_results': {},
            'accuracy': 0.0,
            'validation_accuracy': 0.0
        }
    
    def _simulate_detailed_trading(self, predictions: List[int], actual: List[int]) -> Dict:
        """Detailed trading simulation for performance evaluation."""
        returns = []
        trades = 0
        winning_trades = 0
        equity_curve = [1.0]
        
        for pred, actual_label in zip(predictions, actual):
            if pred > 0:
                trades += 1
                
                if actual_label >= 0:
                    trade_return = 0.015 * min(pred, 2)
                    winning_trades += 1
                else:
                    trade_return = -0.03 * abs(actual_label)
                
                returns.append(trade_return)
                equity_curve.append(equity_curve[-1] * (1 + trade_return))
        
        if not returns:
            return {
                'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
                'win_rate': 0, 'total_trades': 0
            }
        
        total_return = (equity_curve[-1] - 1.0)
        
        returns_array = np.array(returns)
        sharpe_ratio = (returns_array.mean() / returns_array.std() * np.sqrt(252) 
                       if returns_array.std() > 0 else 0)
        
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array / running_max) - 1
        max_drawdown = drawdown.min()
        
        win_rate = winning_trades / trades if trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': trades
        }
    
    def _save_trained_agent(self, symbol: str, genome, config, 
                          performance_metrics: Dict, training_data: Dict) -> str:
        """Save trained agent to database."""
        # Convert NEAT genome data to JSON-serializable format
        genome_data = {
            'nodes': {str(k): {'bias': float(v.bias), 'response': float(v.response), 
                             'activation': str(v.activation), 'aggregation': str(v.aggregation)} 
                     for k, v in genome.nodes.items()},
            'connections': {f"{k[0]}-{k[1]}": {'weight': float(v.weight), 'enabled': bool(v.enabled)} 
                           for k, v in genome.connections.items()},
            'fitness': float(genome.fitness) if genome.fitness else 0.0
        }
        
        agent_data = {
            'symbol': str(symbol),
            'generation': 100,
            'fitness_score': float(genome.fitness) if genome.fitness else 0.0,
            'network_config': {
                'num_inputs': int(config.genome_config.num_inputs),
                'num_outputs': int(config.genome_config.num_outputs),
                'activation_default': str(config.genome_config.activation_default)
            },
            'genome_data': genome_data,
            'feature_names': list(training_data['feature_names']),
            'training_period': str(training_data['data_period']),
            'is_champion': True
        }
        
        training_info = {
            'generations': 100,
            'population_size': self.training_config['population_size'],
            'training_time': 3600,
            'data_start': training_data['data_period'].split(' to ')[0],
            'data_end': training_data['data_period'].split(' to ')[1]
        }
        
        agent_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.agent_db.save_agent(agent_id, symbol, agent_data, performance_metrics, training_info)
        
        return agent_id
    
    def train_batch(self, symbols: List[str], max_workers: int = 4) -> List[TrainingResult]:
        """Train multiple agents in parallel."""
        self.logger.info(f"🚀 Starting batch training for {len(symbols)} stocks...")
        
        # Prepare training data
        training_data_map = {}
        for symbol in symbols:
            try:
                training_data_map[symbol] = self.prepare_training_data(symbol)
            except Exception as e:
                self.logger.error(f"❌ Failed to prepare data for {symbol}: {str(e)}")
        
        # Train agents with parallel data prep and optimized NEAT
        results = []
        import concurrent.futures
        import multiprocessing as mp
        
        # Use ThreadPoolExecutor for I/O bound operations, limit to avoid overload
        max_workers = min(len(training_data_map), mp.cpu_count() // 2, 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all training jobs
            future_to_symbol = {
                executor.submit(self._train_single_optimized, symbol, training_data): symbol
                for symbol, training_data in training_data_map.items()
            }
            
            # Process completed jobs
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.success:
                        self.logger.info(f"✅ {symbol}: Training completed (Fitness: {result.best_fitness:.2f}, Time: {result.training_time:.1f}s)")
                    else:
                        self.logger.error(f"❌ {symbol}: Training failed - {result.error_message}")
                        
                except Exception as e:
                    self.logger.error(f"❌ {symbol}: Unexpected error - {str(e)}")
                    results.append(TrainingResult(
                        symbol=symbol, success=False, best_fitness=0,
                        generations_completed=0, training_time=0,
                        agent_id="", performance_metrics={}, error_message=str(e)
                    ))
        
        return results
    
    def _train_single_optimized(self, symbol: str, training_data: Dict) -> TrainingResult:
        """Optimized single agent training with better progress reporting."""
        start_time = time.time()
        
        try:
            self.logger.info(f"🚀 Starting optimized training for {symbol}...")
            
            num_features = len(training_data['feature_names'])
            config = self._create_optimized_neat_config(symbol, num_features)
            
            population = neat.Population(config)
            
            # Enhanced progress reporter
            class OptimizedProgressReporter(neat.reporting.BaseReporter):
                def __init__(self, symbol, logger):
                    self.symbol = symbol
                    self.logger = logger
                    self.generation = 0
                    self.start_time = time.time()
                
                def start_generation(self, generation):
                    self.generation = generation
                    elapsed = time.time() - self.start_time
                    self.logger.info(f"🧬 {self.symbol} - Generation {generation} (Elapsed: {elapsed:.1f}s)")
                
                def end_generation(self, config, population, species_set):
                    fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
                    if fitnesses:
                        best_fitness = max(fitnesses)
                        avg_fitness = sum(fitnesses) / len(fitnesses)
                        self.logger.info(f"📊 {self.symbol} - Gen {self.generation}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}, Pop={len(fitnesses)}")
            
            population.add_reporter(OptimizedProgressReporter(symbol, self.logger))
            
            # Optimized fitness function with vectorization
            def optimized_fitness_function(genomes, config):
                import numpy as np
                
                features = training_data['train']['features'].values
                labels = training_data['train']['labels'].values
                
                for genome_id, genome in genomes:
                    try:
                        net = neat.nn.FeedForwardNetwork.create(genome, config)
                        
                        # Simplified fitness calculation to avoid broadcasting issues
                        sample_size = min(100, len(features))  # Use fixed sample size
                        sample_indices = np.random.choice(len(features), sample_size, replace=False)
                        
                        correct_predictions = 0
                        total_return = 0.0
                        returns_list = []
                        
                        for idx in sample_indices:
                            try:
                                output = net.activate(features[idx])
                                pred = np.argmax(output) - 2
                                actual = labels[idx]
                                
                                # Accuracy tracking
                                if pred == actual:
                                    correct_predictions += 1
                                
                                # Return calculation
                                if pred > 0 and actual > 0:
                                    ret = 0.02  # Win: +2%
                                elif pred > 0 and actual <= 0:
                                    ret = -0.015  # Loss: -1.5%
                                elif pred < 0 and actual < 0:
                                    ret = 0.015  # Win: +1.5%
                                elif pred < 0 and actual >= 0:
                                    ret = -0.01  # Loss: -1%
                                else:
                                    ret = 0  # No position
                                
                                returns_list.append(ret)
                                total_return += ret
                                
                            except Exception:
                                returns_list.append(0)
                        
                        accuracy = correct_predictions / sample_size
                        returns_array = np.array(returns_list)
                        
                        sharpe = (np.mean(returns_array) / (np.std(returns_array) + 1e-8)) * np.sqrt(252)
                        
                        # Profit-focused fitness
                        fitness = (
                            total_return * 100 +     # Primary: total profit
                            sharpe * 10 +            # Secondary: risk-adjusted returns
                            accuracy * 20 +          # Tertiary: prediction accuracy
                            -abs(total_return) * 2   # Penalty for excessive risk
                        )
                        
                        genome.fitness = max(fitness, 0.1)  # Ensure positive fitness
                        
                    except Exception as e:
                        genome.fitness = 0.1  # Minimal fitness for failed genomes
            
            # Run evolution with early stopping
            best_genome = None
            best_fitness = 0
            generations_without_improvement = 0
            max_generations = min(self.training_config['max_generations'], 30)  # Limit for speed
            
            for generation in range(max_generations):
                population.run(optimized_fitness_function, 1)
                
                # Check for improvement
                current_best = max(population.population.values(), key=lambda g: g.fitness or 0)
                if current_best.fitness > best_fitness:
                    best_fitness = current_best.fitness
                    best_genome = current_best
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                # Early stopping
                if generations_without_improvement >= 8:  # Reduced patience for speed
                    self.logger.info(f"🛑 {symbol}: Early stopping at generation {generation}")
                    break
                
                if best_fitness >= self.training_config['fitness_threshold']:
                    self.logger.info(f"🎯 {symbol}: Fitness threshold reached at generation {generation}")
                    break
            
            # Evaluate final performance
            performance_metrics = self._evaluate_agent_performance(
                best_genome, config, training_data
            )
            
            # Save trained agent
            agent_id = self._save_trained_agent(symbol, best_genome, config, 
                                              performance_metrics, training_data)
            
            training_time = time.time() - start_time
            
            return TrainingResult(
                symbol=symbol,
                success=True,
                best_fitness=best_fitness,
                generations_completed=generation + 1,
                training_time=training_time,
                agent_id=agent_id,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"❌ {symbol}: Training failed after {training_time:.1f}s - {str(e)}")
            
            return TrainingResult(
                symbol=symbol,
                success=False,
                best_fitness=0,
                generations_completed=0,
                training_time=training_time,
                agent_id="",
                performance_metrics={},
                error_message=str(e)
            )
    
    def _create_optimized_neat_config(self, symbol: str, num_features: int) -> neat.Config:
        """Create optimized NEAT configuration for faster training."""
        config_text = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = {self.training_config['fitness_threshold']}
pop_size              = 100
reset_on_extinction   = False

[DefaultGenome]
activation_default      = relu
activation_mutate_rate  = 0.05
activation_options      = relu sigmoid tanh
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.4
conn_delete_prob        = 0.2
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = 0.3
node_delete_prob        = 0.1
num_hidden              = 1
num_inputs              = {num_features}
num_outputs             = 5
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
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
max_stagnation       = 10
species_elitism      = 2

[DefaultReproduction]
elitism            = 3
survival_threshold = 0.3
"""
        
        config_path = f"/tmp/neat_config_opt_{symbol}_{os.getpid()}.txt"
        with open(config_path, 'w') as f:
            f.write(config_text)
        
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_path)
    
    def run_full_training_pipeline(self, selection_file: str = None) -> Dict:
        """Run complete training pipeline for all selected stocks."""
        self.logger.info("🚀 Starting full agent training pipeline...")
        
        selection_data = self.load_stock_selection(selection_file)
        training_schedule = selection_data['training_schedule']
        
        all_results = []
        total_batches = len(training_schedule['training_batches'])
        
        for i, batch in enumerate(training_schedule['training_batches'], 1):
            self.logger.info(f"🔄 Processing batch {i}/{total_batches}: {batch['symbols']}")
            
            batch_results = self.train_batch(batch['symbols'])
            all_results.extend(batch_results)
            
            successful = sum(1 for r in batch_results if r.success)
            self.logger.info(f"✅ Batch {i} completed: {successful}/{len(batch_results)} successful")
        
        # Generate training report
        report = self._generate_training_report(all_results, selection_data)
        
        report_file = f"data/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📄 Training report saved: {report_file}")
        return report
    
    def _generate_training_report(self, results: List[TrainingResult], 
                                selection_data: Dict) -> Dict:
        """Generate comprehensive training report."""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        report = {
            'training_date': datetime.now().isoformat(),
            'total_stocks': len(results),
            'successful_training': len(successful_results),
            'failed_training': len(failed_results),
            'success_rate': len(successful_results) / len(results) if results else 0,
            'training_config': self.training_config,
            'successful_agents': [
                {
                    'symbol': r.symbol,
                    'agent_id': r.agent_id,
                    'fitness_score': r.best_fitness,
                    'accuracy': r.performance_metrics.get('accuracy', 0),
                    'total_return': r.performance_metrics.get('total_return', 0),
                    'sharpe_ratio': r.performance_metrics.get('sharpe_ratio', 0),
                    'win_rate': r.performance_metrics.get('win_rate', 0)
                }
                for r in successful_results
            ],
            'failed_agents': [
                {
                    'symbol': r.symbol,
                    'error_message': r.error_message
                }
                for r in failed_results
            ]
        }
        
        if successful_results:
            accuracies = [r.performance_metrics.get('accuracy', 0) for r in successful_results]
            fitness_scores = [r.best_fitness for r in successful_results]
            
            report['results_summary'] = {
                'avg_accuracy': np.mean(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'avg_fitness': np.mean(fitness_scores),
                'agents_above_80_accuracy': sum(1 for acc in accuracies if acc >= 0.80)
            }
        
        return report


def run_agent_training():
    """Run complete agent training pipeline with comprehensive evaluation."""
    print("🚀 NEAT Options System - Comprehensive Agent Training Pipeline")
    print("=" * 80)
    
    pipeline = AgentTrainingPipeline()
    
    try:
        # Run full training pipeline
        report = pipeline.run_full_training_pipeline()
        
        # Extract training results for comprehensive analysis
        training_results = report.get('training_results', [])
        
        # Generate comprehensive output with dual tables
        pipeline.output_generator.print_comprehensive_results(training_results)
        
        # Save detailed results to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_filename = f"data/comprehensive_training_results_{timestamp}.json"
        pipeline.output_generator.save_results_to_json(training_results, results_filename)
        
        # Check performance threshold compliance
        passed_agents = [r for r in training_results if r.success and r.performance_metrics.get('meets_threshold', False)]
        total_agents = len([r for r in training_results if r.success])
        
        print(f"\n🎯 PERFORMANCE THRESHOLD ANALYSIS")
        print(f"  Agents meeting 50%+ outperformance: {len(passed_agents)}/{total_agents}")
        print(f"  Threshold compliance rate: {len(passed_agents)/total_agents:.1%}" if total_agents > 0 else "  No successful agents")
        
        if len(passed_agents) == 0:
            print(f"\n⚠️  WARNING: NO AGENTS MEET 50%+ OUTPERFORMANCE THRESHOLD")
            print(f"  System requires further optimization for maximum profitability.")
        else:
            print(f"\n✅ {len(passed_agents)} agents successfully exceed buy-and-hold by 50%+")
        
        return report
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_agent_training()
