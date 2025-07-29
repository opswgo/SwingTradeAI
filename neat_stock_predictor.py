#!/usr/bin/env python3
"""
NEAT stock price predictor.
Implements NEAT algorithm for stock price prediction following the official NEAT pattern.
"""
import os
import time
import neat
import numpy as np
import pandas as pd
import logging
import json
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys
from data_handler import DataHandler

class NEATStockPredictor:
    """
    Stock price prediction using NEAT algorithm.
    Strictly follows the official NEAT pattern and uses real market data only.
    """
    
    def __init__(self, ticker="AAPL", config_path=None):
        """
        Initialize the NEAT stock predictor.
        
        Args:
            ticker (str): Stock ticker symbol
            config_path (str): Path to NEAT config file
        """
        # Set up logging
        self.logger = self._setup_logging()
        
        # Store the ticker
        self.ticker = ticker
        self.logger.info(f"Initializing NEAT Stock Predictor for {ticker}")
        
        # Set up paths
        self.ticker_dir = f"{ticker}"
        self.winners_dir = f"{ticker}/winners"
        self.logs_dir = f"{ticker}/logs"
        self.charts_dir = f"{ticker}/charts"
        self.predictions_dir = f"{ticker}/predictions"
        
        # Create directories if they don't exist
        os.makedirs(self.ticker_dir, exist_ok=True)
        os.makedirs(self.winners_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        
        # Set up config path
        if config_path is None:
            config_path = f"{ticker}/neat_config.txt"
            
        # Check if config file exists, if not, create a default one
        if not os.path.exists(config_path):
            self.logger.warning(f"Config file not found at {config_path}, using default configuration")
            config_path = self._create_default_config(ticker)
        
        self.config_path = config_path
        
        # Initialize data handler
        self.data_handler = DataHandler()
        
        # State variables for tracking progress and interrupts
        self.best_genome = None
        self.generation = 0
        self.interrupted = False
        self.state_path = f"{ticker}/state.json"
        
        # Register interrupt handler
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _setup_logging(self):
        """Set up logging configuration."""
        logger = logging.getLogger('neat_stock_predictor')
        logger.setLevel(logging.INFO)
        
        # Check if handler already exists
        if not logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(console_handler)
        
        return logger
    
    def _create_default_config(self, ticker):
        """Create default NEAT config file."""
        config_path = f"{ticker}/neat_config.txt"
        
        with open(config_path, 'w') as f:
            f.write("""# NEAT configuration for stock price prediction

[NEAT]
fitness_criterion     = max
fitness_threshold     = 95.0
pop_size              = 150
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = sigmoid
activation_mutate_rate  = 0.1
activation_options      = sigmoid relu tanh

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
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full_direct

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 10
num_inputs              = 7
num_outputs             = 56  # 8 weeks of daily price predictions + 8 weeks of daily up/down

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
max_stagnation       = 15
species_elitism      = 2

[DefaultReproduction]
elitism            = 10
survival_threshold = 0.2
""")
        
        return config_path
    
    def _create_custom_file_reporter(self, log_file):
        """Create a custom reporter class for writing to a file."""
        class CustomFileReporter(neat.reporting.BaseReporter):
            def __init__(self, file_object):
                self.file = file_object
                
            def start_generation(self, generation):
                self.file.write(f"\n****** Running generation {generation} ******\n")
                self.file.flush()
                
            def end_generation(self, config, population, species_set):
                self.file.write(f"Population size: {len(population)}\n")
                self.file.write(f"Species count: {len(species_set.species)}\n")
                self.file.flush()
            
            def post_evaluate(self, config, population, species, best_genome):
                self.file.write(f"Best fitness: {best_genome.fitness}\n")
                self.file.write(f"Best genome size: {len(best_genome.connections)}\n")
                self.file.flush()
                
            def complete_extinction(self):
                self.file.write("\n****** EXTINCTION ******\n\n")
                self.file.flush()
                
            def found_solution(self, config, generation, best):
                self.file.write(f"\n****** SOLUTION FOUND at generation {generation} ******\n")
                self.file.write(f"Fitness: {best.fitness}\n")
                self.file.flush()
                
            def species_stagnant(self, sid, species):
                self.file.write(f"Species {sid} is stagnant\n")
                self.file.flush()
                
            def info(self, msg):
                self.file.write(f"INFO: {msg}\n")
                self.file.flush()
        
        return CustomFileReporter(log_file)

    def _handle_interrupt(self, sig, frame):
        """Handle interrupt signal (Ctrl+C) by saving state."""
        self.logger.info("Interrupt detected! Saving state and exiting gracefully...")
        self.interrupted = True
        self._save_state()
        self.logger.info(f"State saved to {self.state_path}. Resume with --resume flag.")
        sys.exit(0)
    
    def _save_state(self):
        """Save current training state."""
        if self.best_genome:
            state = {
                'generation': self.generation,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.state_path, 'w') as f:
                json.dump(state, f)
                
            # Save best genome
            with open(f"{self.ticker}/best_genome.pkl", 'wb') as f:
                pickle.dump(self.best_genome, f)
    
    def _load_state(self):
        """Load saved training state."""
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r') as f:
                state = json.load(f)
            
            self.generation = state['generation']
            self.logger.info(f"Resuming from generation {self.generation}")
            
            # Load best genome
            if os.path.exists(f"{self.ticker}/best_genome.pkl"):
                with open(f"{self.ticker}/best_genome.pkl", 'rb') as f:
                    self.best_genome = pickle.load(f)
                    
            return True
        
        return False
        
    def fitness_function(self, genomes, config):
        """
        Evaluate all genomes using the training data.
        This is the fitness function called by NEAT's population.run().
        
        Args:
            genomes: List of (genome_id, genome) tuples
            config: NEAT configuration
        """
        self.generation += 1
        self.logger.info(f"Generation {self.generation}")
        start_time = time.time()
        
        fitness_results = []  # Track all fitness scores
        
        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor() as executor:
            future_to_genome = {}
            
            # Submit all genomes for evaluation
            for genome_id, genome in genomes:
                future = executor.submit(self._evaluate_genome, genome, config)
                future_to_genome[future] = (genome_id, genome)
            
            # Collect results as they complete
            for future in as_completed(future_to_genome):
                genome_id, genome = future_to_genome[future]
                try:
                    fitness = future.result()
                    genome.fitness = fitness
                    fitness_results.append(fitness)
                    
                    # Track best genome
                    if self.best_genome is None or fitness > self.best_genome.fitness:
                        self.best_genome = genome
                        
                except Exception as e:
                    self.logger.error(f"Error evaluating genome {genome_id}: {e}")
                    genome.fitness = -1.0
        
        # Save the best genome of this generation
        if self.best_genome:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{self.winners_dir}/winner_{timestamp}_gen{self.generation}.json"
            
            # Save genome data to JSON
            genome_data = {
                "fitness": self.best_genome.fitness,
                "generation": self.generation,
                "nodes": list(self.best_genome.nodes.keys()),
                "connections": [(k[0], k[1], v.weight) for k, v in self.best_genome.connections.items() if v.enabled]
            }
            
            # Also save the actual genome object
            with open(f"{self.winners_dir}/best_genome_gen{self.generation}.pkl", 'wb') as f:
                pickle.dump(self.best_genome, f)
        
        # Log progress
        elapsed = time.time() - start_time
        avg_fitness = sum(fitness_results) / len(fitness_results) if fitness_results else 0
        best_fitness = max(fitness_results) if fitness_results else 0
        
        self.logger.info(f"Generation {self.generation}: "
                          f"Best={best_fitness:.2f}, "
                          f"Avg={avg_fitness:.2f}, "
                          f"Time={elapsed:.2f}s")
        
        # Save state every 5 generations
        if self.generation % 5 == 0:
            self._save_state()
            
            with open(filename, 'w') as f:
                json.dump(genome_data, f, indent=2)
            
            # Also save the actual genome object
            with open(f"{self.winners_dir}/best_genome_gen{self.generation}.pkl", 'wb') as f:
                pickle.dump(self.best_genome, f)
        
        # Log progress
        elapsed = time.time() - start_time
        avg_fitness = sum(fitness_results) / len(fitness_results) if fitness_results else 0
        best_fitness = max(fitness_results) if fitness_results else 0
        
        self.logger.info(f"Generation {self.generation}: "
                          f"Best={best_fitness:.2f}, "
                          f"Avg={avg_fitness:.2f}, "
                          f"Time={elapsed:.2f}s")
        
        # Save state every 5 generations
        if self.generation % 5 == 0:
            self._save_state()
    
    def _evaluate_genome(self, genome, config):
        """
        Evaluate a single genome for stock price prediction.
        
        Args:
            genome: NEAT genome
            config: NEAT configuration
            
        Returns:
            float: Fitness score
        """
        try:
            # Load training data
            training_data = self.training_data
            features = training_data['X_train']  # Changed from 'features' to 'X_train'
            price_targets = training_data['price_targets_train']  # Changed from 'price_targets' to 'price_targets_train'
            direction_targets = training_data['direction_targets_train']  # Changed from 'direction_targets' to 'direction_targets_train'
            
            # Create neural network from genome
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            # Initialize metrics
            price_mse = 0.0
            direction_accuracy = 0.0
            num_samples = len(features)
            
            # Make predictions for each sample
            for i, feature_vector in enumerate(features):
                # Get network output
                outputs = net.activate(feature_vector)
                
                # Split outputs into price predictions and direction predictions
                num_days = len(outputs) // 2
                price_preds = outputs[:num_days]
                direction_preds = outputs[num_days:]
                
                # Calculate price MSE (Mean Squared Error)
                price_target = price_targets[i]
                price_errors = [(pred - target) ** 2 for pred, target in zip(price_preds, price_target)]
                sample_price_mse = sum(price_errors) / len(price_errors)
                price_mse += sample_price_mse
                
                # Calculate direction accuracy
                direction_target = direction_targets[i]
                direction_correct = sum(1 for pred, target in zip(direction_preds, direction_target) 
                                       if (pred > 0.5) == (target > 0.5))
                sample_direction_accuracy = direction_correct / len(direction_target)
                direction_accuracy += sample_direction_accuracy
            
            # Calculate average metrics
            avg_price_mse = price_mse / num_samples
            avg_direction_accuracy = direction_accuracy / num_samples
            
            # Calculate fitness (weighting accuracy higher than MSE)
            # Price MSE is negative (lower is better), scale to [0, 50]
            price_score = 50 * max(0, 1 - avg_price_mse)
            # Direction accuracy is positive (higher is better), scale to [0, 50]
            direction_score = 50 * avg_direction_accuracy
            
            # Combine for total fitness score [0, 100]
            fitness = price_score + direction_score
            
            return fitness
            
        except Exception as e:
            self.logger.error(f"Error in genome evaluation: {str(e)}")
            return -1.0
    
    def train(self, generations=200, resume=False):
        """
        Train the NEAT model for stock price prediction.
        Follows the official NEAT pattern with a single call to population.run().
        
        Args:
            generations (int): Maximum number of generations
            resume (bool): Whether to resume from saved state
            
        Returns:
            dict: Training results and best model
        """
        self.logger.info(f"Starting NEAT training for {self.ticker}")
        start_time = time.time()
        
        try:
            # Load data
            self.training_data = self.data_handler.prepare_data(self.ticker)
            
            # Load NEAT configuration
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                self.config_path
            )
            
            # Resume from saved state if requested
            if resume and self._load_state():
                self.logger.info(f"Resuming from generation {self.generation}")
                population = neat.Population(config, (self.best_genome,))
            else:
                self.logger.info("Starting fresh training")
                self.generation = 0
                population = neat.Population(config)
            
            # Add reporters
            population.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            population.add_reporter(stats)
            
            # Create a log file for writing results
            log_path = f"{self.logs_dir}/{self.ticker}_log.txt"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log_file = open(log_path, 'w')
            
            # Create a custom file reporter
            file_reporter = self._create_custom_file_reporter(self.log_file)
            population.add_reporter(file_reporter)
            
            self.logger.info(f"Starting evolution with population size {len(population.population)}")
            
            # Run NEAT evolution with a single call to population.run
            # This is the official NEAT pattern as mandated in AI_Index.md
            best_genome = population.run(self.fitness_function, n=generations)
            
            # Save the best genome
            if best_genome:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                filename = f"{self.winners_dir}/winner_final_{timestamp}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(best_genome, f)
                
                # Also save as JSON for human readability
                genome_data = {
                    "fitness": best_genome.fitness,
                    "generation": self.generation,
                    "nodes": list(best_genome.nodes.keys()),
                    "connections": [(k[0], k[1], v.weight) for k, v in best_genome.connections.items() if v.enabled]
                }
                
                with open(f"{self.winners_dir}/winner_final_{timestamp}.json", 'w') as f:
                    json.dump(genome_data, f, indent=2)
            
            # Calculate total training time
            training_time = time.time() - start_time
            
            self.logger.info(f"Training complete in {training_time:.2f} seconds")
            self.logger.info(f"Best fitness: {best_genome.fitness if best_genome else 'N/A'}")
            
            # Close log file if it exists
            if hasattr(self, 'log_file') and self.log_file:
                self.log_file.close()
                self.log_file = None
            
            return {
                "success": True,
                "best_genome": best_genome,
                "config": config,
                "generations": self.generation,
                "training_time": training_time
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            
            # Close log file if it exists
            if hasattr(self, 'log_file') and self.log_file:
                self.log_file.close()
                self.log_file = None
                
            return {"success": False, "error": str(e)}
    
    def predict(self, genome=None, config=None):
        """
        Generate predictions using the trained model.
        
        Args:
            genome: NEAT genome to use (or best genome if None)
            config: NEAT configuration
            
        Returns:
            dict: Prediction results
        """
        self.logger.info(f"Generating predictions for {self.ticker}")
        
        try:
            # Load best genome if none provided
            if genome is None:
                if self.best_genome:
                    genome = self.best_genome
                elif os.path.exists(f"{self.ticker}/best_genome.pkl"):
                    with open(f"{self.ticker}/best_genome.pkl", 'rb') as f:
                        genome = pickle.load(f)
                else:
                    latest_winner = None
                    latest_time = 0
                    for filename in os.listdir(self.winners_dir):
                        if filename.endswith(".pkl"):
                            file_path = os.path.join(self.winners_dir, filename)
                            file_time = os.path.getmtime(file_path)
                            if file_time > latest_time:
                                latest_time = file_time
                                latest_winner = file_path
                    
                    if latest_winner:
                        with open(latest_winner, 'rb') as f:
                            genome = pickle.load(f)
                    else:
                        raise ValueError("No trained model found")
            
            # Load config if none provided
            if config is None:
                config = neat.Config(
                    neat.DefaultGenome,
                    neat.DefaultReproduction,
                    neat.DefaultSpeciesSet,
                    neat.DefaultStagnation,
                    self.config_path
                )
            
            # Fetch latest data for prediction
            data = self.data_handler.fetch_data(self.ticker, period="2y")
            features_df, normalized_features = self.data_handler.engineer_features(data)
            
            # Get the latest feature vector
            latest_features = normalized_features[-1]
            
            # Create neural network from genome
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            # Get predictions from network
            outputs = net.activate(latest_features)
            
            # Split outputs into price predictions and direction predictions
            num_days = len(outputs) // 2
            price_ratios = outputs[:num_days]  # Price ratios (current price = 1.0)
            direction_probs = outputs[num_days:]  # Direction probabilities (>0.5 means up)
            
            # Get current price
            current_price = data['Close'].iloc[-1]
            
            # Calculate predicted prices
            predicted_prices = [current_price * ratio for ratio in price_ratios]
            
            # Generate dates for predictions
            last_date = data.index[-1]
            prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_days)
            
            # Create prediction DataFrame
            predictions_df = pd.DataFrame({
                'Date': prediction_dates,
                'Price': predicted_prices,
                'Direction': ['Up' if p > 0.5 else 'Down' for p in direction_probs],
                'Confidence': [abs(p - 0.5) * 2 for p in direction_probs]  # Scale to [0, 1]
            })
            
            # Save predictions
            predictions_file = f"{self.predictions_dir}/predictions_{self.ticker}.csv"
            predictions_df.to_csv(predictions_file)
            
            self.logger.info(f"Predictions saved to {predictions_file}")
            
            # Highlight days 5, 10, 20, 40
            highlight_days = [4, 9, 19, 39]  # 0-indexed
            highlights = {}
            for day_idx in highlight_days:
                if day_idx < len(predictions_df):
                    day_num = day_idx + 1
                    highlights[f"Day_{day_num}"] = {
                        "Date": predictions_df['Date'].iloc[day_idx].strftime("%Y-%m-%d"),
                        "Price": predictions_df['Price'].iloc[day_idx],
                        "Direction": predictions_df['Direction'].iloc[day_idx],
                        "Confidence": predictions_df['Confidence'].iloc[day_idx]
                    }
            
            return {
                "success": True,
                "current_price": current_price,
                "current_date": last_date.strftime("%Y-%m-%d"),
                "predictions": predictions_df.to_dict(orient="records"),
                "highlights": highlights,
                "predictions_file": predictions_file
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_charts(self, prediction_results=None):
        """
        Generate charts showing actual vs. predicted prices.
        
        Args:
            prediction_results: Results from predict() or None to run predict() first
            
        Returns:
            list: Paths to generated chart files
        """
        self.logger.info(f"Generating charts for {self.ticker}")
        
        try:
            # Get predictions if not provided
            if prediction_results is None or not prediction_results.get("success"):
                prediction_results = self.predict()
                
                if not prediction_results.get("success"):
                    return {"success": False, "error": prediction_results.get("error")}
            
            # Get historical data for comparison
            data = self.data_handler.fetch_data(self.ticker, period="2y")
            
            # Extract prediction data as native Python types
            current_price = float(prediction_results["current_price"])
            current_date = str(prediction_results["current_date"])
            
            # Convert predictions to native Python types
            raw_predictions = prediction_results["predictions"]
            predictions_native = []
            
            for p in raw_predictions:
                predictions_native.append({
                    'Date': p['Date'] if isinstance(p['Date'], str) else p['Date'].strftime('%Y-%m-%d'),
                    'Price': float(p['Price']),
                    'Direction': str(p['Direction']),
                    'Confidence': float(p['Confidence'])
                })
            
            # Convert to lists for plotting
            pred_dates = [np.datetime64(p['Date']) for p in predictions_native]
            pred_prices = [p['Price'] for p in predictions_native]
            pred_directions = [p['Direction'] for p in predictions_native]
            pred_confidences = [p['Confidence'] for p in predictions_native]
            
            # Convert historical data to native Python types
            hist_dates = data.index[-60:].to_numpy()
            hist_prices = data['Close'].iloc[-60:].to_numpy()
            
            # Generate main prediction chart
            plt.figure(figsize=(14, 8))
            
            # Plot historical prices
            plt.plot(hist_dates, hist_prices, 'b-', label='Historical Prices', linewidth=2)
            
            # Plot predicted prices
            plt.plot(pred_dates, pred_prices, 'r--', label='Predicted Prices', linewidth=2)
            
            # Add confidence bands (20% above and below predictions)
            plt.fill_between(pred_dates, 
                            [p * 0.8 for p in pred_prices], 
                            [p * 1.2 for p in pred_prices], 
                            color='red', alpha=0.2, 
                            label='Prediction Confidence Band')
            
            # Add horizontal line for current price
            plt.axhline(y=current_price, color='green', linestyle='-', alpha=0.5, 
                        label=f'Current Price (${current_price:.2f})')
            
            # Add vertical line for current date
            plt.axvline(x=np.datetime64(current_date), color='green', linestyle='-', alpha=0.5, 
                        label=f'Current Date ({current_date})')
            
            # Highlight key prediction days (5, 10, 20, 40)
            highlight_days = [4, 9, 19, 39]  # 0-indexed
            for day_idx in highlight_days:
                if day_idx < len(pred_dates):
                    day_num = day_idx + 1
                    plt.scatter(pred_dates[day_idx], 
                            pred_prices[day_idx], 
                            color='orange', s=100, zorder=5)
                    plt.annotate(f"Day {day_num}", 
                                xy=(pred_dates[day_idx], pred_prices[day_idx]),
                                xytext=(10, 10), textcoords='offset points', 
                                fontsize=12, color='black', 
                                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
            
            # Set chart properties
            plt.title(f"{self.ticker} Stock Price Prediction (8 Weeks)", fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Price ($)', fontsize=12)
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save price prediction chart
            price_chart_path = f"{self.charts_dir}/{self.ticker}_price_prediction.png"
            plt.savefig(price_chart_path, dpi=300)
            plt.close()
            
            # Generate direction prediction chart
            plt.figure(figsize=(14, 6))
            
            # Direction predictions as bar chart
            direction_values = [(pred_confidences[i] if pred_directions[i] == 'Up' else -pred_confidences[i]) 
                                for i in range(len(pred_directions))]
            
            # Plot direction bars
            bars = plt.bar(range(len(direction_values)), direction_values, 
                        color=['green' if d > 0 else 'red' for d in direction_values])
            
            # Add confidence labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height < 0:
                    height = -height
                    y_pos = -height - 0.05
                else:
                    y_pos = height + 0.02
                
                plt.text(bar.get_x() + bar.get_width()/2, y_pos,
                        f'{pred_confidences[i]:.2f}',
                        ha='center', va='bottom', rotation=0,
                        fontsize=8)
            
            # Set chart properties
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title(f"{self.ticker} Price Direction Prediction (Up/Down) for 8 Weeks", fontsize=16, fontweight='bold')
            plt.xlabel('Days from Now', fontsize=12)
            plt.ylabel('Direction Confidence [-1 to 1]', fontsize=12)
            plt.ylim(-1.2, 1.2)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save direction prediction chart
            direction_chart_path = f"{self.charts_dir}/{self.ticker}_direction_prediction.png"
            plt.savefig(direction_chart_path, dpi=300)
            plt.close()
            
            self.logger.info(f"Charts saved to {self.charts_dir}/")
            
            return {
                "success": True,
                "charts": {
                    "price_chart": price_chart_path,
                    "direction_chart": direction_chart_path
                }
            }
            
        except Exception as e:
            self.logger.error(f"Chart generation failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def evaluate(self):
        """
        Evaluate the trained model on validation data.
        
        Returns:
            dict: Evaluation results
        """
        self.logger.info(f"Evaluating model for {self.ticker}")
        
        try:
            # Load best genome
            if self.best_genome is None:
                if os.path.exists(f"{self.ticker}/best_genome.pkl"):
                    with open(f"{self.ticker}/best_genome.pkl", 'rb') as f:
                        self.best_genome = pickle.load(f)
                else:
                    latest_winner = None
                    latest_time = 0
                    for filename in os.listdir(self.winners_dir):
                        if filename.endswith(".pkl"):
                            file_path = os.path.join(self.winners_dir, filename)
                            file_time = os.path.getmtime(file_path)
                            if file_time > latest_time:
                                latest_time = file_time
                                latest_winner = file_path
                    
                    if latest_winner:
                        with open(latest_winner, 'rb') as f:
                            self.best_genome = pickle.load(f)
                    else:
                        raise ValueError("No trained model found")
            
            # Load configuration
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                self.config_path
            )
            
            # Create neural network from genome
            net = neat.nn.FeedForwardNetwork.create(self.best_genome, config)
            
            # Load validation data (use 20% of the data that wasn't used for training)
            data = self.data_handler.fetch_data(self.ticker, period="2y")
            features_df, normalized_features = self.data_handler.engineer_features(data)
            price_targets, direction_targets = self.data_handler.create_targets(data)
            
            # Use the last 20% for validation
            val_size = int(len(normalized_features) * 0.2)
            val_features = normalized_features[-val_size:]
            val_price_targets = price_targets[-val_size:]
            val_direction_targets = direction_targets[-val_size:]
            
            # Initialize metrics for each prediction day
            num_days = val_price_targets.shape[1]  # Number of days we're predicting ahead
            price_mse = np.zeros(num_days)
            direction_accuracy = np.zeros(num_days)
            
            # Make predictions for each validation sample
            for i, feature_vector in enumerate(val_features):
                # Get network output
                outputs = net.activate(feature_vector)
                
                # Split outputs into price predictions and direction predictions
                price_preds = outputs[:num_days]
                direction_preds = outputs[num_days:]
                
                # Calculate price MSE for each day
                price_target = val_price_targets[i]
                for day in range(num_days):
                    price_mse[day] += (price_preds[day] - price_target[day]) ** 2
                
                # Calculate direction accuracy for each day
                direction_target = val_direction_targets[i]
                for day in range(num_days):
                    if (direction_preds[day] > 0.5) == (direction_target[day] > 0.5):
                        direction_accuracy[day] += 1
            
            # Calculate average metrics
            price_mse /= val_size
            direction_accuracy /= val_size
            
            # Create DataFrame for evaluation results
            eval_df = pd.DataFrame({
                'Day': range(1, num_days + 1),
                'Price_MSE': price_mse,
                'Direction_Accuracy': direction_accuracy
            })
            
            # Save evaluation results
            eval_file = f"{self.logs_dir}/{self.ticker}_evaluation.csv"
            eval_df.to_csv(eval_file)
            
            # Log evaluation results
            with open(f"{self.logs_dir}/{self.ticker}_log.txt", 'a') as f:
                f.write(f"\n\nEVALUATION RESULTS:\n")
                f.write(f"Average Price MSE: {np.mean(price_mse):.4f}\n")
                f.write(f"Average Direction Accuracy: {np.mean(direction_accuracy):.4f}\n\n")
                
                f.write(f"Results by day:\n")
                for day in [0, 4, 9, 19, 39]:  # Days 1, 5, 10, 20, 40
                    if day < len(price_mse):
                        f.write(f"Day {day+1}: Price MSE={price_mse[day]:.4f}, "
                                f"Direction Accuracy={direction_accuracy[day]:.4f}\n")
            
            self.logger.info(f"Evaluation complete, results saved to {eval_file}")
            
            return {
                "success": True,
                "price_mse": price_mse.tolist(),
                "direction_accuracy": direction_accuracy.tolist(),
                "avg_price_mse": float(np.mean(price_mse)),
                "avg_direction_accuracy": float(np.mean(direction_accuracy)),
                "evaluation_file": eval_file
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return {"success": False, "error": str(e)}
