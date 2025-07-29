#!/usr/bin/env python3
"""
NEAT stock price predictor.
Implements NEAT algorithm for stock price prediction following the official NEAT pattern.
"""
import os
import time
import neat
import json
import pickle
import logging
import shutil
import psutil
import numpy as np
import pandas as pd
import multiprocessing
import signal
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from functools import partial
from data_handler import DataHandler

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class NEATStockPredictor:
    def __init__(self, ticker="AAPL", config_path=None, num_workers=None):
        """
        Initialize the NEAT stock predictor.
        Args:
            ticker (str): Stock ticker symbol
            config_path (str): Path to NEAT config file
            num_workers (int or None): Number of parallel workers (processes) for evaluation
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
        os.makedirs(self.ticker_dir, exist_ok=True)
        os.makedirs(self.winners_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        # Set up config path
        if config_path is None:
            config_path = f"{ticker}/neat_config.txt"
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
        # Store number of workers for parallelism
        self.num_workers = num_workers

    """
    Stock price prediction using NEAT algorithm.
    Strictly follows the official NEAT pattern and uses real market data only.
    """
    
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
    
    def _validate_generation_results(self, fitness_results, elapsed_time):
        """
        Validate generation results according to AI_Index.md requirements.
        Ensures no false success claims and verifies the quality of results.
        
        Args:
            fitness_results: List of fitness scores from current generation
            elapsed_time: Time taken for the generation to complete
        """
        # Sanity check fitness values
        if not fitness_results:
            self.logger.warning("VALIDATION FAILURE: No fitness results available for validation")
            return
            
        # Check for unrealistic fitness values
        if max(fitness_results) > 100.0:  # Our fitness scale is 0-100
            self.logger.warning(f"VALIDATION FAILURE: Unrealistic max fitness value: {max(fitness_results)}")
            
        # Performance validation
        genomes_per_second = len(fitness_results) / elapsed_time if elapsed_time > 0 else 0
        self.logger.info(f"Performance validation: {genomes_per_second:.1f} genomes/second")
            
        # Check fitness trend (should improve over generations)
        if self.generation > 1 and hasattr(self, 'previous_best_fitness'):
            improvement = max(fitness_results) - self.previous_best_fitness
            self.logger.info(f"Fitness improvement: {improvement:.2f} from previous generation")
            
            # Alert if fitness is consistently decreasing
            if improvement < -1.0 and hasattr(self, 'fitness_decrease_count'):
                self.fitness_decrease_count += 1
                if self.fitness_decrease_count >= 3:
                    self.logger.warning("VALIDATION ALERT: Fitness decreasing for 3+ consecutive generations")
            else:
                self.fitness_decrease_count = 0
        
        # Store current best for future comparison
        self.previous_best_fitness = max(fitness_results) if fitness_results else 0
        
    def _save_state(self):
        """Save the current state of the trainer to disk."""
        state_path = f"{self.logs_dir}/checkpoint_gen{self.generation}.pkl"
        
        checkpoint = {
            'generation': self.generation,
            'best_genome': self.best_genome,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(state_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.logger.info(f"Saved checkpoint to {state_path}")
    
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
        # Force garbage collection before evaluation to free memory
        import gc
        gc.collect()
        
        self.generation += 1
        self.logger.info(f"Generation {self.generation}")
        start_time = time.time()
        
        fitness_results = []  # Track all fitness scores
        
        # Dynamically determine optimal number of workers based on available hardware
        cpu_count = multiprocessing.cpu_count()
        available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        # Use user-specified workers if provided, else auto-detect
        if self.num_workers is not None and self.num_workers > 0:
            optimal_workers = self.num_workers
        else:
            worker_memory_estimate_gb = 0.05  # 50MB per worker in GB
            memory_based_workers = int(available_memory_gb / worker_memory_estimate_gb * 0.95)
            optimal_workers = min(cpu_count, memory_based_workers)
            optimal_workers = max(2, optimal_workers)  # At least 2 workers for ProcessPoolExecutor
        self.logger.info(f"Hardware detection: {cpu_count} CPU cores, {available_memory_gb:.2f}GB available memory")
        self.logger.info(f"Using {optimal_workers} PROCESS workers for parallel evaluation (full CPU utilization)")

        
        # Use numpy.memmap for large training arrays to minimize serialization
        import tempfile
        X_train = np.array(self.training_data['X_train'])
        price_targets_train = np.array(self.training_data['price_targets_train'])
        tmp_dir = tempfile.TemporaryDirectory()
        X_path = f"{tmp_dir.name}/X_train.dat"
        y_path = f"{tmp_dir.name}/y_train.dat"
        X_train_mem = np.memmap(X_path, dtype=X_train.dtype, mode='w+', shape=X_train.shape)
        X_train_mem[:] = X_train[:]
        y_train_mem = np.memmap(y_path, dtype=price_targets_train.dtype, mode='w+', shape=price_targets_train.shape)
        y_train_mem[:] = price_targets_train[:]
        X_train_mem.flush()
        y_train_mem.flush()
        del X_train
        del price_targets_train
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            future_to_batch = {}
            genome_batches = []
            batch_size = max(2, len(genomes) // (optimal_workers * 4))
            current_batch = []
            for genome_id, genome in genomes:
                current_batch.append((genome_id, genome))
                if len(current_batch) >= batch_size:
                    genome_batches.append(list(current_batch))
                    current_batch = []
            if current_batch:
                genome_batches.append(current_batch)
            for batch in genome_batches:
                genome_ids = [g[0] for g in batch]
                genome_objs = [g[1] for g in batch]
                future = executor.submit(
                    NEATStockPredictor._evaluate_genome_batch_static,
                    genome_ids, genome_objs, X_path, y_path, X_train_mem.shape, y_train_mem.shape, X_train_mem.dtype.str, y_train_mem.dtype.str, config
                )
                future_to_batch[future] = batch
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    for (genome_id, fitness) in batch_results:
                        for orig_id, orig_genome in genomes:
                            if orig_id == genome_id:
                                orig_genome.fitness = fitness
                                fitness_results.append(fitness)
                                if self.best_genome is None or fitness > self.best_genome.fitness:
                                    self.best_genome = orig_genome
                                break
                except Exception as e:
                    self.logger.error(f"Error evaluating genome batch: {e}")
                    for genome_id, _ in batch:
                        for orig_id, orig_genome in genomes:
                            if orig_id == genome_id:
                                orig_genome.fitness = -1.0
        tmp_dir.cleanup()
        
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

        
        # Calculate metrics
        avg_fitness = sum(fitness_results) / len(fitness_results) if fitness_results else 0
        best_fitness = max(fitness_results) if fitness_results else 0
        
        # Log comprehensive metrics
        self.logger.info(f"Generation {self.generation}: "
                         f"Best={best_fitness:.2f}, "
                         f"Avg={avg_fitness:.2f}, "
                         f"Time={elapsed:.2f}s")
        
        # Validate results according to AI_Index.md requirements
        self._validate_generation_results(fitness_results, elapsed)
        
        # Save state every 5 generations
        if self.generation % 5 == 0:
            self._save_state()
            
            # Save the best genome to JSON and pickle
            if self.best_genome and 'filename' in locals():
                with open(filename, 'w') as f:
                    json.dump(genome_data, f, indent=2)
                
                with open(f"{self.winners_dir}/best_genome_gen{self.generation}.pkl", 'wb') as f:
                    pickle.dump(self.best_genome, f)
            self._save_state()
    
    @staticmethod
    def _evaluate_genome_batch_static(genome_ids, genome_objs, X_path, y_path, X_shape, y_shape, X_dtype, y_dtype, config):
        """
        Static method for process-safe batch genome evaluation.
        Uses numpy.memmap for large arrays, sets numpy/BLAS thread env vars to 1, logs per-worker CPU/time.
        Args:
            genome_ids: List of genome IDs
            genome_objs: List of genome objects
            X_path, y_path: Paths to memmap files
            X_shape, y_shape: Shapes of memmap arrays
            X_dtype, y_dtype: Dtypes of memmap arrays
            config: NEAT configuration
        Returns:
            List of (genome_id, fitness) tuples
        """
        import os, time, numpy as np
        start_time = time.time()
        X_train = np.memmap(X_path, dtype=np.dtype(X_dtype), mode='r', shape=tuple(X_shape))
        y_train = np.memmap(y_path, dtype=np.dtype(y_dtype), mode='r', shape=tuple(y_shape))
        results = []
        for genome_id, genome in zip(genome_ids, genome_objs):
            try:
                fitness = NEATStockPredictor._evaluate_genome_static(genome, X_train, y_train, config)
                results.append((genome_id, fitness))
            except Exception:
                results.append((genome_id, -1.0))
        elapsed = time.time() - start_time
        print(f"[Worker PID {os.getpid()}] Batch size: {len(genome_ids)}, Time: {elapsed:.2f}s", flush=True)
        return results

    @staticmethod
    def _evaluate_genome_static(genome, X_train, y_train, config):
        """
        Aggressively optimized static method for process-safe genome evaluation.
        Uses vectorized NumPy operations for all possible steps.
        Args:
            genome: NEAT genome (must be pickleable)
            X_train: numpy memmap array
            y_train: numpy memmap array
            config: NEAT configuration (must be pickleable)
        Returns:
            float: Fitness score
        """
        try:
            num_samples = X_train.shape[0]
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            # Pre-allocate output array for all predictions
            predictions = np.empty((num_samples, 28), dtype=np.float32)
            # Evaluate all samples (still must call net.activate per sample, but minimize Python overhead)
            for i in range(num_samples):
                out = net.activate(X_train[i])
                if isinstance(out, (list, np.ndarray)):
                    if len(out) < 28:
                        # Pad with last value if too short
                        pad = [out[-1]] * (28 - len(out)) if out else [1.0] * 28
                        out = list(out) + pad
                    elif len(out) > 28:
                        out = out[:28]
                    predictions[i, :] = out
                else:
                    predictions[i, :] = [out] * 28
            # Vectorized MSE calculation
            mse = np.mean((predictions - y_train[:num_samples, :28]) ** 2)
            return -mse
        except Exception:
            return -1.0
        
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
            features = training_data['X_train']  
            price_targets = training_data['price_targets_train']  
            
            # Create neural network from genome
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            # Initialize metrics - only price MSE, no direction accuracy
            price_mse = 0.0
            num_samples = len(features)
            
            # Vectorized approach for batch processing
            # Process in batches to avoid memory issues with very large datasets
            batch_size = 128
            num_samples = len(features)
            
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch_size_actual = batch_end - batch_start
                
                # Process batch with vectorized operations for better CPU utilization
                # Use parallel processing with smaller chunks for better CPU utilization
                batch_features = [features[i] for i in range(batch_start, batch_end)]
                
                # Process in smaller chunks for better CPU utilization
                batch_outputs = []
                chunk_size = max(1, batch_size_actual // 4)  
                
                for chunk_start in range(0, batch_size_actual, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, batch_size_actual)
                    # Process this chunk in parallel using list comprehension
                    chunk_outputs = [net.activate(batch_features[i]) for i in range(chunk_start, chunk_end)]
                    batch_outputs.extend(chunk_outputs)
                
                # Convert batch outputs to numpy array for vectorized operations
                batch_predictions = np.array(batch_outputs)
                
                # Check shape and ensure all predictions have correct length
                if batch_predictions.shape[1] != 28:
                    self.logger.warning(f"Network produced {batch_predictions.shape[1]} outputs, expected 28. Correcting dimensions.")
                    if batch_predictions.shape[1] < 28:
                        # Pad predictions if too short (vectorized operation)
                        pad_width = ((0, 0), (0, 28 - batch_predictions.shape[1]))
                        last_vals = batch_predictions[:, -1:] if batch_predictions.shape[1] > 0 else np.ones((batch_size_actual, 1))
                        batch_predictions = np.pad(batch_predictions, pad_width, 'constant')
                        # Fill padded values with last value for each sample
                        for i in range(batch_size_actual):
                            batch_predictions[i, batch_predictions.shape[1] - (28 - batch_predictions.shape[1]):] = last_vals[i]
                    else:
                        # Truncate if too long (vectorized operation)
                        batch_predictions = batch_predictions[:, :28]
                
                # Get corresponding batch of price targets
                batch_indices = np.arange(batch_start, batch_end)
                batch_price_targets = price_targets[batch_indices]
                
                # Calculate MSE using vectorized operations
                batch_price_mse = np.mean((batch_predictions - batch_price_targets) ** 2, axis=1).sum()
                price_mse += batch_price_mse
                
            # Calculate average metrics - only price MSE now
            avg_price_mse = price_mse / num_samples
            
            # Calculate fitness based solely on price MSE
            # Price MSE is negative (lower is better), scale to [0, 100]
            fitness = 100 * max(0, 1 - avg_price_mse)
            
            # Add a small penalty for large errors to prevent extreme predictions
            if avg_price_mse > 1.0:
                fitness = max(0, fitness - 20 * (avg_price_mse - 1.0))
                
            # Cap fitness at 100
            fitness = min(fitness, 100.0)
            
            return fitness
            
        except Exception as e:
            self.logger.error(f"Error in genome evaluation: {str(e)}")
            return -1.0
    
    def train(self, generations=200, resume=False):
        import time
        train_start = time.time()
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
                
                # Save genome data to JSON
                genome_data = {
                    "fitness": best_genome.fitness,
                    "generation": self.generation,
                    "nodes": list(best_genome.nodes.keys()),
                    "connections": [(k[0], k[1], v.weight) for k, v in best_genome.connections.items() if v.enabled]
                }
                
                # Also save the actual genome object
                with open(filename, 'wb') as f:
                    pickle.dump(best_genome, f)
            
            # Calculate total training time
            train_end = time.time()
            total_time = train_end - train_start
            
            self.logger.info(f"Training complete in {total_time:.2f} seconds")
            self.logger.info(f"Total training runtime: {total_time:.2f} seconds")
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
            
            # With the updated NEAT config, outputs are price predictions only (28 days)
            price_ratios = outputs  # All outputs are price ratios (current price = 1.0)
            
            # Ensure we have exactly 28 outputs as expected
            if len(price_ratios) != 28:
                self.logger.warning(f"Network produced {len(price_ratios)} outputs, expected 28. Adjusting...")
                if len(price_ratios) < 28:
                    # Pad with last value if needed
                    last_val = price_ratios[-1] if len(price_ratios) > 0 else 1.0
                    price_ratios = np.pad(price_ratios, (0, 28 - len(price_ratios)), 'constant', constant_values=last_val)
                else:
                    # Truncate if too long
                    price_ratios = price_ratios[:28]
            
            # Convert to numpy array for consistent handling
            price_ratios = np.array(price_ratios)
            
            # Get current price
            current_price = float(data['Close'].iloc[-1])
            
            # Calculate predicted prices
            predicted_prices = [float(current_price * ratio) for ratio in price_ratios]
            
            # Generate dates for predictions
            last_date = data.index[-1]
            prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predicted_prices))
            
            # Create prediction DataFrame - no more direction predictions
            # Instead, determine direction from day-to-day price changes
            # Ensure all arrays are the same length before creating DataFrame
            dates_list = prediction_dates.tolist()
            
            # Debug information
            self.logger.info(f"Lengths check before creating DataFrame: dates={len(dates_list)}, prices={len(predicted_prices)}")
            
            # Ensure arrays have the same length
            min_length = min(len(dates_list), len(predicted_prices))
            dates_list = dates_list[:min_length]
            predicted_prices = predicted_prices[:min_length]
            
            # Convert to python primitives to avoid pandas type issues
            prices_list = [float(p) for p in predicted_prices]
            
            # Create the DataFrame with explicit index
            predictions_df = pd.DataFrame({
                'Date': dates_list,
                'Price': prices_list
            }, index=range(len(dates_list)))
            
            # Calculate direction based on price changes
            # Initialize with a list of correct length
            directions = ['Initial'] + ['Unknown'] * (len(predictions_df) - 1)
            
            # Update directions based on price changes
            for i in range(1, len(predictions_df)):
                price_now = float(predictions_df['Price'].iloc[i])
                price_prev = float(predictions_df['Price'].iloc[i-1])
                directions[i] = 'Up' if price_now > price_prev else 'Down'
                
            # Assign the full list to the DataFrame at once
            predictions_df['Direction'] = directions
            
            # Calculate confidence based on magnitude of price change (simple method)
            # Initialize with a list of correct length
            confidences = [0.5] * len(predictions_df)
            
            # Update confidences based on price changes
            for i in range(1, len(predictions_df)):
                price_now = float(predictions_df['Price'].iloc[i])
                price_prev = float(predictions_df['Price'].iloc[i-1])
                price_change_pct = abs(price_now / price_prev - 1)
                # Scale confidence: 0% change = 0.5, 5% change = 1.0
                confidences[i] = min(0.5 + price_change_pct * 10, 1.0)
                
            # Assign the full list to the DataFrame at once
            predictions_df['Confidence'] = confidences
            
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
                        "Price": float(predictions_df['Price'].iloc[day_idx]),
                        "Direction": str(predictions_df['Direction'].iloc[day_idx]),
                        "Confidence": float(predictions_df['Confidence'].iloc[day_idx])
                    }
            
            return {
                "success": True,
                "current_price": float(current_price),
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
            price_targets, _ = self.data_handler.create_targets(data)  # We're only using price targets
            
            # Use the last 20% for validation
            val_size = int(len(normalized_features) * 0.2)
            val_features = normalized_features[-val_size:]
            val_price_targets = price_targets[-val_size:]
            
            # Initialize metrics for each prediction day
            num_days = val_price_targets.shape[1]  # Number of days we're predicting ahead
            price_mse = np.zeros(num_days)
            
            # Make predictions for each validation sample
            for i, feature_vector in enumerate(val_features):
                # Get network output
                outputs = net.activate(feature_vector)
                
                # Ensure outputs has the right length
                if len(outputs) != 28:
                    self.logger.warning(f"Network produced {len(outputs)} outputs, expected 28.")
                    if len(outputs) < 28:
                        # Pad with last value
                        last_val = outputs[-1] if outputs else 1.0
                        outputs = np.pad(outputs, (0, 28 - len(outputs)), 'constant', constant_values=last_val)
                    else:
                        # Truncate
                        outputs = outputs[:28]
                
                # Calculate price MSE for each day
                price_target = val_price_targets[i]
                for day in range(num_days):
                    price_mse[day] += (outputs[day] - price_target[day]) ** 2
            
            # Calculate average metrics
            price_mse /= val_size
            
            # Create DataFrame for evaluation results
            eval_df = pd.DataFrame({
                'Day': range(1, num_days + 1),
                'Price_MSE': price_mse
            })
            
            # Save evaluation results
            eval_file = f"{self.logs_dir}/{self.ticker}_evaluation.csv"
            eval_df.to_csv(eval_file)
            
            # Log evaluation results
            with open(f"{self.logs_dir}/{self.ticker}_log.txt", 'a') as f:
                f.write(f"\n\nEVALUATION RESULTS:\n")
                f.write(f"Average Price MSE: {np.mean(price_mse):.4f}\n\n")
                
                f.write(f"Results by day:\n")
                for day in [0, 4, 9, 19]:  # Days 1, 5, 10, 20
                    if day < len(price_mse):
                        f.write(f"Day {day+1}: Price MSE={price_mse[day]:.4f}\n")
            
            self.logger.info(f"Evaluation complete, results saved to {eval_file}")
            
            return {
                "success": True,
                "price_mse": price_mse.tolist(),
                "avg_price_mse": float(np.mean(price_mse)),
                "evaluation_file": eval_file
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return {"success": False, "error": str(e)}
