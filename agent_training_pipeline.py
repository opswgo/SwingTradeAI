"""
Agent Training Pipeline - NEAT Options System
Train individual NEAT agents for each selected stock with profit-focused fitness.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import time

# Import NEAT and our modules
import neat
from enhanced_data_pipeline import EnhancedDataPipeline
from src.database.agent_database import AgentDatabase

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
        self.db_path = db_path
        self.agent_db = AgentDatabase(db_path)
        self.data_pipeline = EnhancedDataPipeline()
        
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
        """Create labels for options trading strategy."""
        horizons = [5, 10, 15, 20]  # 1-4 weeks
        labels = []
        
        for i in range(len(data)):
            if i >= len(data) - max(horizons):
                labels.append(0)
                continue
            
            current_price = data['Close'].iloc[i]
            signals = []
            
            for horizon in horizons:
                if i + horizon < len(data):
                    future_price = data['Close'].iloc[i + horizon]
                    return_pct = (future_price / current_price) - 1
                    
                    if return_pct > 0.02:
                        signals.append(2)  # Strong buy signal (sell puts)
                    elif return_pct > 0.005:
                        signals.append(1)  # Weak buy signal
                    elif return_pct < -0.02:
                        signals.append(-2)  # Strong sell signal
                    elif return_pct < -0.005:
                        signals.append(-1)  # Weak sell signal
                    else:
                        signals.append(0)  # Neutral
            
            if signals:
                weights = np.array([1, 1.5, 2, 2.5][:len(signals)])
                weighted_signal = np.average(signals, weights=weights)
                
                if weighted_signal > 1:
                    labels.append(2)
                elif weighted_signal > 0.25:
                    labels.append(1)
                elif weighted_signal < -1:
                    labels.append(-2)
                elif weighted_signal < -0.25:
                    labels.append(-1)
                else:
                    labels.append(0)
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
        """Evaluate trained agent performance on test data."""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        test_features = training_data['test']['features'].values
        test_labels = training_data['test']['labels'].values
        
        predictions = []
        for features in test_features:
            output = net.activate(features)
            predicted_class = np.argmax(output) - 2
            predictions.append(predicted_class)
        
        accuracy = np.mean(np.array(predictions) == test_labels)
        
        # Simulate detailed trading
        profit_simulation = self._simulate_detailed_trading(predictions, test_labels)
        
        performance_metrics = {
            'accuracy': accuracy,
            'fitness_score': genome.fitness,
            'total_return': profit_simulation['total_return'],
            'sharpe_ratio': profit_simulation['sharpe_ratio'],
            'max_drawdown': profit_simulation['max_drawdown'],
            'win_rate': profit_simulation['win_rate'],
            'total_trades': profit_simulation['total_trades'],
            'validation_accuracy': accuracy
        }
        
        return performance_metrics
    
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
        genome_data = {
            'nodes': {k: {'bias': v.bias, 'response': v.response, 'activation': v.activation, 'aggregation': v.aggregation} 
                     for k, v in genome.nodes.items()},
            'connections': {k: {'weight': v.weight, 'enabled': v.enabled} 
                           for k, v in genome.connections.items()},
            'fitness': genome.fitness
        }
        
        agent_data = {
            'symbol': symbol,
            'generation': 100,
            'fitness_score': genome.fitness,
            'network_config': {
                'num_inputs': config.genome_config.num_inputs,
                'num_outputs': config.genome_config.num_outputs,
                'activation_default': config.genome_config.activation_default
            },
            'genome_data': genome_data,
            'feature_names': training_data['feature_names'],
            'training_period': training_data['data_period'],
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
        
        # Train agents sequentially (multiprocessing complex with NEAT)
        results = []
        for symbol in training_data_map.keys():
            try:
                result = self.train_single_agent(symbol, training_data_map[symbol])
                results.append(result)
                
                if result.success:
                    self.logger.info(f"✅ {symbol}: Training completed successfully")
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
    """Run complete agent training pipeline."""
    print("🚀 NEAT Options System - Agent Training Pipeline")
    print("=" * 60)
    
    pipeline = AgentTrainingPipeline()
    
    try:
        # Run full training pipeline
        report = pipeline.run_full_training_pipeline()
        
        # Display results
        print(f"\n✅ Training Pipeline Complete!")
        print(f"📊 Results Summary:")
        print(f"  Total stocks: {report['total_stocks']}")
        print(f"  Successful: {report['successful_training']}")
        print(f"  Failed: {report['failed_training']}")
        print(f"  Success rate: {report['success_rate']:.1%}")
        
        if 'results_summary' in report:
            summary = report['results_summary']
            print(f"  Average accuracy: {summary['avg_accuracy']:.1%}")
            print(f"  Agents ≥80% accuracy: {summary['agents_above_80_accuracy']}")
        
        print(f"\n🏆 Top Performing Agents:")
        successful_agents = sorted(report['successful_agents'], 
                                 key=lambda x: x['fitness_score'], reverse=True)
        
        for i, agent in enumerate(successful_agents[:10], 1):
            print(f"  {i:2d}. {agent['symbol']:6s} | "
                  f"Fitness: {agent['fitness_score']:6.1f} | "
                  f"Accuracy: {agent['accuracy']:5.1%} | "
                  f"Return: {agent['total_return']:6.1%}")
        
        return report
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_agent_training()
