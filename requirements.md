Stock Price Direction Prediction System Requirements
1. Overview
This document specifies the requirements for a production-level system to predict stock price direction (up or down) for 1 to 8 weeks using a NEAT algorithm with genetic optimization. The system is designed to be stock-specific, scalable, and robust, supporting command-line input for ticker selection, multithreading for efficient computation, interruptible execution with state resumption, and comprehensive error handling. Outputs are organized in stock-specific subdirectories (e.g., ./HOOD/, ./AVGO/). The system targets a high accuracy (90%+, with realistic expectations of 60–70% due to market noise) and is optimized for a macOS environment with an M1 Pro Max (48GB RAM).

Model Training:
Initialize a feedforward neural network (FNN) with 7 input nodes, 10 hidden nodes (ReLU activation), and 11 output nodes (sigmoid activation).
Use a genetic algorithm to optimize FNN weights, with a population of 100 networks, 200 generations, mutation rate 0.01, and crossover rate 0.7.
Evaluate fitness (accuracy) in parallel using multithreading (ThreadPoolExecutor).
Implement elitism to preserve the top 10% of networks each generation.
Save the best network per generation to ./ticker/winners/winner_<timestamp>_gen<gen>.json.
Support interruption (Ctrl+C) with state saving to ./ticker/state.json and resumption via --resume.


Backtesting and Evaluation:
Evaluate the best model on the validation set to compute average accuracy across 1–11 day predictions.
Log accuracy to ./ticker/logs/ticker_log.txt.


Prediction:
Use the best model to predict price direction for 1–11 days based on the latest data point.
Highlight predictions for Day 5 and Day 10 (e.g., "Up (Day 5)").
Save predictions to ./ticker/predictions_ticker.csv with dates, predictions, and confidence scores.


Visualization:
Plot the best fitness (accuracy) per generation and save to ./ticker/fitness_history_ticker.png.


Error Handling and Logging:
Implement comprehensive error handling for data fetching, computation, training, and file operations.
Log all events and errors to ./ticker/logs/ticker_log.txt.



2.2. Design Decisions

Modularity: Split functionality across files for maintainability:
main.py: Orchestrates the pipeline.
data_handler.py: Manages data fetching and preprocessing.
neural_network.py: Implements the FNN.
genetic_algorithm.py: Optimizes weights using a genetic algorithm.
utils.py: Provides logging and visualization utilities.


Stock-Specific Directories: Store all outputs (data, predictions, winners, logs, plots) in ./ticker/ to support multiple stocks without conflicts.
Multithreading: Use ThreadPoolExecutor for parallel fitness evaluation to leverage the M1 Pro Max’s multi-core architecture.
Interruptibility: Save training state on interruption and allow resumption to prevent data loss.
Production-Level Code:
Use Python 3.13 with type hints and PEP 8/PEP 257 compliance.
Include detailed docstrings with parameter pros/cons for genetic algorithm settings.
Avoid deprecated libraries (e.g., yfinance>=0.2.40, ta>=0.11.0).


Scalability: Design the system to support adding new indicators or switching to regression tasks with minimal code changes (guided by comments).

3. Functional Requirements
3.1. Stock-Specific Models

Create and train a separate FNN model for each stock ticker (e.g., HOOD, AVGO).
Store model outputs and data in stock-specific subdirectories (e.g., ./HOOD/, ./AVGO/).

3.2. Command-Line Interface

Accept a required --ticker argument to specify the stock (e.g., python main.py --ticker HOOD).
Support optional arguments:
--generations: Number of genetic algorithm generations (default: 200).
--resume: Boolean flag to resume training from a saved state.



3.3. Data Input

Source: Fetch historical stock data from yfinance for July 1, 2015, to July 22, 2025, or maximum available for newer stocks.
Features: Compute 7 technical indicators as inputs:
RSI: Relative Strength Index (14-day window).
MACD: Moving Average Convergence Divergence (12-26-9 configuration).
ATR: Average True Range (14-day window).
Volume: Daily trading volume.
ROC: Rate of Change (10-day window).
Bollinger Width: (Upper - Lower Bollinger Band) / Moving Average (20-day window).
10 DMA: 10-day Moving Average.


Data Storage: Save raw OHLCV data to ./ticker/ticker_data.csv.
Preprocessing:
Normalize features using z-score normalization (mean=0, std=1).
Create binary labels (1 for price increase, 0 for decrease) for 1–11 days ahead.
Split data into 80% training and 20% validation sets.
Handle missing data by dropping rows with NaN values after indicator computation.



3.4. Neural Network

Architecture:
Input layer: 7 nodes (one per technical indicator).
Hidden layer: 10 nodes with ReLU activation.
Output layer: 11 nodes (one per day, 1–11) with sigmoid activation for binary classification.


Functionality:
Forward propagation to compute output probabilities.
Methods to set and get weights for genetic algorithm optimization.
Evaluate accuracy on validation data (average across 1–11 day predictions).
Predict price direction (up/down) and probabilities for 1–11 days.
Highlight Day 5 and Day 10 predictions in output (e.g., "Up (Day 5)").



3.5. Genetic Algorithm

Parameters:
Population size: 100 networks.
Pros: Increases diversity, improves exploration, reduces risk of local optima.
Cons: Higher computational cost, longer runtime.
Recommended: 50–200.


Mutation rate: 0.01.
Pros: Prevents stagnation, explores new solutions.
Cons: High rates disrupt good solutions, causing instability.
Recommended: 0.001–0.1.


Crossover rate: 0.7.
Pros: Combines good traits, accelerates convergence.
Cons: High rates reduce diversity, may lose unique solutions.
Recommended: 0.5–0.9.


Generations: 200 (configurable via --generations).
Pros: More generations improve optimization.
Cons: Longer runtime, diminishing returns after convergence.
Recommended: 100–500.




Features:
Initialize population with random weights (scaled by 0.01) or load from ./ticker/state.json.
Evaluate fitness (accuracy) using multithreading for efficiency.
Use tournament selection (size 5) to select parents.
Apply uniform crossover to combine parent weights.
Apply Gaussian mutation (std=0.1) to introduce variation.
Implement elitism to preserve the top 10% of networks each generation.
Save the best network per generation to ./ticker/winners/winner_<timestamp>_gen<gen>.json with weights, fitness, and architecture details.
Save training state (population, generation, fitness history) to ./ticker/state.json on interruption (Ctrl+C).
Support resumption of training using the --resume flag.
Delete state.json upon successful completion.



3.6. Backtesting

Evaluate the best model on the validation set.
Compute average accuracy across 1–11 day predictions.
Log backtest accuracy to ./ticker/logs/ticker_log.txt.

3.7. Prediction Output

Generate predictions for 1–11 days using the latest data point.
Highlight Day 5 and Day 10 predictions (e.g., "Up (Day 5)").
Save predictions to ./ticker/predictions_ticker.csv with columns:
Date: Predicted date (YYYY-MM-DD).
Prediction: Up or Down (with Day 5/10 highlighted).
Confidence: Prediction probability (0 to 1).


Example CSV:Date,Prediction,Confidence
2025-07-23,Up,0.75
2025-07-24,Down,0.62
...
2025-07-27,Up (Day 5),0.82
...
2025-08-01,Down (Day 10),0.65
2025-08-02,Up,0.71



3.8. Visualization

Plot the best fitness (accuracy) per generation.
Save the plot to ./ticker/fitness_history_ticker.png.
Use matplotlib with a clear layout (e.g., labeled axes, title, grid).

4. Non-Functional Requirements
4.1. Performance

Accuracy: Target 90%+ accuracy on validation data, with realistic expectations of 60–70% due to market unpredictability.
Runtime: Optimize for an M1 Pro Max with 48GB RAM, using multithreading for fitness evaluation. Expected runtime: ~5–10 minutes per stock for 200 generations.
Resource Usage: Ensure memory and CPU efficiency to handle large datasets and parallel computations.

4.2. Scalability

Support multiple tickers via command-line input (e.g., --ticker HOOD, --ticker AVGO).
Allow easy addition of new technical indicators by modifying data_handler.py (with commented guidance).
Configurable for regression tasks (e.g., predicting percentage price changes) with minimal code changes, guided by comments in data_handler.py and neural_network.py.

4.3. Robustness

Handle partial data for stocks with less than 10 years of history (e.g., HOOD since its 2021 IPO).
Implement comprehensive error handling for:
Data fetching failures (e.g., network issues, invalid tickers).
Indicator computation errors (e.g., insufficient data).
File I/O errors (e.g., permission issues).
Neural network and genetic algorithm exceptions (e.g., invalid weights, convergence issues).


Log all errors and key events (e.g., training progress, interruptions) to ./ticker/logs/ticker_log.txt with timestamps and severity levels.

4.4. Interruptibility

Save training state (population, generation, fitness history) to ./ticker/state.json on interruption (Ctrl+C).
Allow resumption of training from the saved state using the --resume flag.
Ensure state file is deleted upon successful completion to avoid stale data.

5. Technical Requirements
5.1. Programming Language

Use Python 3.13 for all source code.
Adhere to PEP 8 (style) and PEP 257 (docstrings) guidelines for readability and maintainability.

5.2. Dependencies

Use non-deprecated libraries (as of July 2025):
numpy>=2.1.1: Numerical computations for arrays and matrices.
pandas>=2.2.2: Data handling and manipulation.
yfinance>=0.2.40: Fetching historical stock data.
ta>=0.11.0: Computing technical indicators.
matplotlib>=3.9.1: Plotting fitness history.


List all dependencies in requirements.txt for easy installation.

5.3. File Structure

Source Files:
main.py: Entry point, orchestrates the pipeline (data preparation, training, prediction, visualization).
data_handler.py: Handles data fetching, indicator computation, preprocessing, backtesting, and prediction saving.
neural_network.py: Implements the FNN with forward propagation, weight management, evaluation, and prediction.
genetic_algorithm.py: Implements the genetic algorithm with multithreading, selection, crossover, mutation, and state management.
utils.py: Provides utility functions for logging setup and fitness history plotting.
requirements.txt: Lists dependencies for installation.


Output Directory Structure:./ticker/
├── ticker_data.csv           # Raw and processed stock data
├── predictions_ticker.csv    # Predictions for 1–11 days
├── state.json                # Training state for resumption
├── winners/                  # Best models per generation
│   ├── winner_<timestamp>_gen<gen>.json
├── logs/                     # Log files
│   ├── ticker_log.txt
├── fitness_history_ticker.png # Fitness history plot



5.4. Code Quality

Object-Oriented Design: Use classes for DataHandler, NeuralNetwork, and GeneticAlgorithm to encapsulate functionality.
Documentation:
Include detailed docstrings for all classes, methods, and functions.
Describe purpose, arguments, return types, and exceptions.
For genetic algorithm parameters, include pros/cons in the GeneticAlgorithm class docstring.


Type Hints: Use Python type annotations for all function and method signatures to improve code clarity.
Error Handling:
Catch and log exceptions for all critical operations (data fetching, computation, training, file I/O).
Raise meaningful exceptions with descriptive messages.


Modularity: Split functionality across files to ensure maintainability and reusability.
Code Style: Follow PEP 8 for formatting (e.g., 4-space indentation, maximum line length 79) and PEP 257 for docstrings.

6. Constraints

Data Availability: Handle stocks with limited history (e.g., HOOD since 2021) by using all available data.
Platform: Optimize for macOS with an M1 Pro Max (48GB RAM) to ensure efficient computation.
No Deprecated Libraries: Use only current, supported versions of libraries compatible with Python 3.13.
No External Dependencies: Beyond listed libraries, avoid external services or APIs for data or computation.
Accuracy Realism: While targeting 90%+ accuracy, acknowledge that 60–70% is more realistic due to market volatility.

7. Deliverables

Python source files (main.py, data_handler.py, neural_network.py, genetic_algorithm.py, utils.py) implementing the full pipeline.
requirements.txt for dependency installation.
Stock-specific output files in ./ticker/ directories (data, predictions, winners, logs, plots).
This requirements document (requirements.md) detailing specifications and approach.

8. Usage Instructions

Install Dependencies:pip install -r requirements.txt


Run for a Specific Ticker:python main.py --ticker HOOD


Run with Custom Generations:python main.py --ticker HOOD --generations 300


Resume Training:python main.py --ticker HOOD --resume


Interrupt Training: Press Ctrl+C to save the training state and exit.
Output: Check ./ticker/ for:
ticker_data.csv: Raw and processed data.
predictions_ticker.csv: 1–11 day predictions with confidence.
winners/winner_<timestamp>_gen<gen>.json: Best models per generation.
logs/ticker_log.txt: Logs of events and errors.
fitness_history_ticker.png: Fitness history plot.



9. Notes

Accuracy Expectations: The 90%+ accuracy target is ambitious; backtesting will help validate performance, but 60–70% is typical due to market noise.
Scalability: The system is designed to handle multiple tickers sequentially via command-line input. Future enhancements could include parallel processing for multiple tickers.
Extensibility: Comments in data_handler.py and neural_network.py should guide switching to regression tasks or adding new indicators (e.g., Stochastic Oscillator).
Performance: Multithreading optimizes fitness evaluation for the M1 Pro Max, reducing runtime to ~5–10 minutes per stock for 200 generations.
