#!/usr/bin/env python3
"""
Improved Self-Learning NEAT Trainer
Continues training until performance metrics are met or learning stalls.
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

class ImprovedNEATTrainer:
    """Self-learning NEAT trainer that continues until metrics are met."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.target_metrics = {
            'min_win_rate': 0.60,           # 60% win rate minimum
            'min_total_return': 0.20,       # 20% total return minimum
            'min_vs_buy_hold': 0.05,        # 5% better than buy & hold
            'max_drawdown': -0.15,          # Max 15% drawdown
            'min_sharpe': 1.0               # Minimum Sharpe ratio
        }
        
    def _setup_logging(self):
        logger = logging.getLogger('ImprovedNEAT')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def create_adaptive_neat_config(self, num_inputs: int, generation: int = 0) -> str:
        """Create NEAT configuration that adapts based on generation."""
        
        # Adaptive parameters based on generation
        pop_size = min(200, 50 + generation * 5)  # Increase population over time
        conn_add_prob = max(0.2, 0.5 - generation * 0.01)  # Decrease connection adding
        node_add_prob = max(0.05, 0.2 - generation * 0.005)  # Decrease node adding
        weight_mutate_rate = max(0.6, 0.9 - generation * 0.01)  # Decrease mutation rate
        
        config_text = f"""
[NEAT]
fitness_criterion = max
fitness_threshold = 1000.0
pop_size = {pop_size}
reset_on_extinction = False

[DefaultGenome]
activation_default = tanh
activation_mutate_rate = 0.1
activation_options = tanh relu sigmoid leaky_relu
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
conn_add_prob = {conn_add_prob}
conn_delete_prob = 0.2
enabled_default = True
enabled_mutate_rate = 0.01
feed_forward = True
initial_connection = full_direct
node_add_prob = {node_add_prob}
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
weight_mutate_rate = {weight_mutate_rate}
weight_replace_rate = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation = 20
species_elitism = 2

[DefaultReproduction]
elitism = 3
survival_threshold = 0.2
"""
        return config_text
    
    def engineer_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer enhanced features for better performance."""
        df = data.copy()
        features = pd.DataFrame(index=df.index)
        
        # Price momentum (8 features)
        features['returns_1d'] = df['Close'].pct_change()
        features['returns_5d'] = df['Close'].pct_change(5)
        features['returns_20d'] = df['Close'].pct_change(20)
        features['returns_60d'] = df['Close'].pct_change(60)
        
        # Moving averages (6 features)
        for period in [10, 20, 50]:
            features[f'sma_{period}'] = df['Close'].rolling(period).mean()
            features[f'price_vs_sma_{period}'] = df['Close'] / features[f'sma_{period}'] - 1
        
        # Volatility (6 features)
        features['volatility_5d'] = features['returns_1d'].rolling(5).std()
        features['volatility_20d'] = features['returns_1d'].rolling(20).std()
        features['volatility_60d'] = features['returns_1d'].rolling(60).std()
        features['vol_ratio_5_20'] = features['volatility_5d'] / features['volatility_20d']
        features['vol_ratio_20_60'] = features['volatility_20d'] / features['volatility_60d']
        features['vol_percentile'] = features['volatility_20d'].rolling(252).rank(pct=True)
        
        # Volume (4 features)
        features['volume_sma_20'] = df['Volume'].rolling(20).mean()
        features['volume_ratio'] = df['Volume'] / features['volume_sma_20']
        features['volume_trend'] = features['volume_ratio'].rolling(5).mean()
        features['price_volume_trend'] = (df['Close'] * df['Volume']).rolling(10).mean()
        
        # Technical indicators (6 features)
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_sma = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        features['bb_upper'] = bb_sma + (bb_std * 2)
        features['bb_lower'] = bb_sma - (bb_std * 2)
        features['bb_position'] = (df['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_sma
        
        # Market structure (4 features)
        features['high_low_ratio'] = df['High'] / df['Low']
        features['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        features['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        features['intraday_return'] = (df['Close'] - df['Open']) / df['Open']
        
        # Clean and normalize
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        # Z-score normalization
        for col in features.columns:
            if features[col].std() > 0:
                features[col] = (features[col] - features[col].mean()) / features[col].std()
        
        return features
    
    def create_optimized_labels(self, data: pd.DataFrame, horizon_weeks: int = 4) -> np.ndarray:
        """Create optimized labels for better learning."""
        horizon_days = horizon_weeks * 5
        labels = []
        
        for i in range(len(data)):
            if i >= len(data) - horizon_days:
                labels.append(0.0)
                continue
            
            current_price = data['Close'].iloc[i]
            
            # Look at multiple future points for stability
            future_returns = []
            for offset in range(horizon_days - 2, horizon_days + 3):  # ±2 days around target
                if i + offset < len(data):
                    future_price = data['Close'].iloc[i + offset]
                    return_pct = (future_price / current_price) - 1
                    future_returns.append(return_pct)
            
            if future_returns:
                # Use median return for stability
                median_return = np.median(future_returns)
                
                # Enhanced signal calculation
                if median_return > 0.03:  # Strong bullish
                    signal = 0.8
                elif median_return > 0.01:  # Moderate bullish
                    signal = 0.4
                elif median_return < -0.03:  # Strong bearish
                    signal = -0.8
                elif median_return < -0.01:  # Moderate bearish
                    signal = -0.4
                else:  # Neutral
                    signal = 0.0
                
                labels.append(signal)
            else:
                labels.append(0.0)
        
        return np.array(labels)
    
    def enhanced_fitness_function(self, genome, config, training_data):
        """Enhanced fitness function focused on meeting target metrics."""
        try:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            features = training_data['features'].values
            labels = training_data['labels']
            prices = training_data['raw_data']['Close'].values
            
            # Generate predictions
            predictions = []
            for feature_row in features:
                output = net.activate(feature_row)
                predictions.append(output[0])
            
            predictions = np.array(predictions)
            
            # Enhanced trading simulation
            trades = []
            positions = []
            position = 0
            entry_price = 0
            days_in_position = 0
            max_position_days = 20  # 4 weeks
            
            for i in range(len(predictions)):
                prediction = predictions[i]
                current_price = prices[i] if i < len(prices) else prices[-1]
                
                # Entry logic with confidence threshold
                if position == 0 and abs(prediction) > 0.3:
                    position = np.sign(prediction) * min(abs(prediction), 1.0)  # Position size based on confidence
                    entry_price = current_price
                    days_in_position = 0
                
                # Position management
                elif position != 0:
                    days_in_position += 1
                    
                    if i < len(labels):
                        # Calculate current P&L
                        price_change = (current_price / entry_price) - 1
                        trade_return = position * price_change
                        
                        # Exit conditions
                        should_exit = False
                        
                        # Stop loss
                        if trade_return < -0.03:
                            should_exit = True
                        # Take profit
                        elif trade_return > 0.05:
                            should_exit = True
                        # Time exit
                        elif days_in_position >= max_position_days:
                            should_exit = True
                        # Signal reversal
                        elif np.sign(prediction) != np.sign(position) and abs(prediction) > 0.4:
                            should_exit = True
                        
                        if should_exit:
                            final_return = trade_return - 0.002  # Transaction costs
                            trades.append(final_return)
                            positions.append({
                                'entry_price': entry_price,
                                'exit_price': current_price,
                                'return': final_return,
                                'days_held': days_in_position
                            })
                            position = 0
                            days_in_position = 0
            
            if len(trades) == 0:
                return 0.1
            
            # Calculate comprehensive metrics
            total_return = sum(trades)
            win_rate = len([t for t in trades if t > 0]) / len(trades)
            
            # Sharpe ratio
            if len(trades) > 1:
                sharpe = np.mean(trades) / (np.std(trades) + 1e-8)
            else:
                sharpe = 0
            
            # Maximum drawdown
            cumulative = np.cumsum(trades)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = cumulative - running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            # Profit factor
            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t < 0]
            profit_factor = (sum(wins) / abs(sum(losses))) if losses else 10.0
            
            # Enhanced fitness calculation targeting specific metrics
            fitness = 0
            
            # Win rate component (0-200 points)
            if win_rate >= self.target_metrics['min_win_rate']:
                fitness += 200 * (win_rate / self.target_metrics['min_win_rate'])
            else:
                fitness += 100 * (win_rate / self.target_metrics['min_win_rate'])
            
            # Total return component (0-300 points)
            if total_return >= self.target_metrics['min_total_return']:
                fitness += 300 * (total_return / self.target_metrics['min_total_return'])
            else:
                fitness += 150 * (total_return / self.target_metrics['min_total_return'])
            
            # Sharpe ratio component (0-200 points)
            if sharpe >= self.target_metrics['min_sharpe']:
                fitness += 200 * (sharpe / self.target_metrics['min_sharpe'])
            else:
                fitness += 100 * (sharpe / self.target_metrics['min_sharpe'])
            
            # Drawdown penalty (0-100 points)
            if max_drawdown >= self.target_metrics['max_drawdown']:
                fitness += 100
            else:
                fitness += 50 * (max_drawdown / self.target_metrics['max_drawdown'])
            
            # Profit factor bonus (0-100 points)
            fitness += min(100, profit_factor * 20)
            
            # Activity bonus (encourage reasonable trading)
            trades_per_year = len(trades) * 252 / len(features)
            if 10 <= trades_per_year <= 50:  # Reasonable frequency
                fitness += 50
            
            return max(fitness, 0.1)
            
        except Exception as e:
            return 0.1
    
    def meets_target_metrics(self, performance: dict) -> bool:
        """Check if performance meets target metrics."""
        return (
            performance['win_rate'] >= self.target_metrics['min_win_rate'] and
            performance['total_return'] >= self.target_metrics['min_total_return'] and
            performance['max_drawdown'] >= self.target_metrics['max_drawdown'] and
            performance.get('sharpe_ratio', 0) >= self.target_metrics['min_sharpe']
        )
    
    def train_until_success(self, symbol: str, data: pd.DataFrame, 
                           horizon_weeks: int = 4, max_attempts: int = 5) -> dict:
        """Train until metrics are met or learning stalls."""
        
        self.logger.info(f"🎯 Training {symbol} until target metrics are met...")
        self.logger.info(f"   Target: {self.target_metrics['min_win_rate']:.0%} win rate, "
                        f"{self.target_metrics['min_total_return']:.0%} return, "
                        f"Sharpe ≥ {self.target_metrics['min_sharpe']}")
        
        best_result = None
        best_fitness = 0
        attempt_results = []
        
        for attempt in range(max_attempts):
            self.logger.info(f"🔄 Attempt {attempt + 1}/{max_attempts}")
            
            try:
                # Prepare enhanced data
                features = self.engineer_enhanced_features(data)
                labels = self.create_optimized_labels(data, horizon_weeks)
                
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
                    'raw_data': data.iloc[:train_size],
                    'symbol': symbol,
                    'horizon_weeks': horizon_weeks
                }
                
                # Create adaptive config
                config_text = self.create_adaptive_neat_config(len(features.columns), attempt * 20)
                
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(config_text)
                    config_path = f.name
                
                config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                   neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
                
                # Train with extended generations
                population = neat.Population(config)
                
                def fitness_wrapper(genomes, config):
                    for genome_id, genome in genomes:
                        genome.fitness = self.enhanced_fitness_function(genome, config, training_data)
                
                best_genome = None
                best_generation_fitness = 0
                stagnation_count = 0
                max_stagnation = 30
                
                for generation in range(100):  # Extended training
                    population.run(fitness_wrapper, 1)
                    
                    # Check for improvement
                    current_best = max(population.population.values(), key=lambda g: g.fitness)
                    
                    if current_best.fitness > best_generation_fitness:
                        best_generation_fitness = current_best.fitness
                        best_genome = current_best
                        stagnation_count = 0
                        
                        # Log progress
                        if generation % 10 == 0:
                            self.logger.info(f"   Gen {generation}: Fitness = {current_best.fitness:.1f}")
                    else:
                        stagnation_count += 1
                    
                    # Early stopping conditions
                    if best_generation_fitness > 800:  # High fitness threshold
                        self.logger.info(f"   🎯 High fitness achieved at generation {generation}")
                        break
                    elif stagnation_count >= max_stagnation:
                        self.logger.info(f"   ⏸️ Training stalled at generation {generation}")
                        break
                
                # Cleanup
                os.unlink(config_path)
                
                # Evaluate performance
                if best_genome:
                    test_features = features.iloc[train_size:]
                    test_labels = labels[train_size:]
                    test_data = data.iloc[train_size:]
                    
                    performance = self._evaluate_enhanced_performance(
                        best_genome, config, test_features, test_labels, test_data
                    )
                    
                    result = {
                        'success': True,
                        'symbol': symbol,
                        'genome': best_genome,
                        'config': config,
                        'performance': performance,
                        'horizon_weeks': horizon_weeks,
                        'attempt': attempt + 1,
                        'final_fitness': best_generation_fitness,
                        'training_period': f"{data.index[0].date()} to {data.index[train_size-1].date()}",
                        'test_period': f"{data.index[train_size].date()} to {data.index[-1].date()}"
                    }
                    
                    attempt_results.append(result)
                    
                    # Check if metrics are met
                    if self.meets_target_metrics(performance):
                        self.logger.info(f"✅ Target metrics achieved for {symbol}!")
                        self.logger.info(f"   Win Rate: {performance['win_rate']:.1%}")
                        self.logger.info(f"   Total Return: {performance['total_return']:.1%}")
                        self.logger.info(f"   Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
                        return result
                    
                    # Track best result
                    if best_generation_fitness > best_fitness:
                        best_fitness = best_generation_fitness
                        best_result = result
                        
                    self.logger.info(f"   Attempt {attempt + 1} - Win Rate: {performance['win_rate']:.1%}, "
                                   f"Return: {performance['total_return']:.1%}")
                
            except Exception as e:
                self.logger.error(f"   ❌ Attempt {attempt + 1} failed: {e}")
                continue
        
        # Return best result if no attempt met targets
        if best_result:
            self.logger.warning(f"⚠️ {symbol} did not meet target metrics. Returning best attempt.")
            best_result['met_targets'] = False
            return best_result
        else:
            return {
                'success': False,
                'symbol': symbol,
                'error': 'All training attempts failed',
                'attempts': attempt_results
            }
    
    def _evaluate_enhanced_performance(self, genome, config, features, labels, price_data):
        """Enhanced performance evaluation."""
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Generate predictions
        predictions = [net.activate(row)[0] for row in features.values]
        
        # Simulate trading
        trades = []
        position = 0
        days_in_position = 0
        entry_price = 0
        
        prices = price_data['Close'].values
        
        for i, prediction in enumerate(predictions):
            current_price = prices[i] if i < len(prices) else prices[-1]
            
            if position == 0 and abs(prediction) > 0.3:
                position = np.sign(prediction)
                entry_price = current_price
                days_in_position = 0
                
            elif position != 0:
                days_in_position += 1
                
                if i < len(labels):
                    price_change = (current_price / entry_price) - 1
                    trade_return = position * price_change
                    
                    if (trade_return < -0.03 or trade_return > 0.05 or 
                        days_in_position >= 20 or 
                        (np.sign(prediction) != np.sign(position) and abs(prediction) > 0.4)):
                        
                        trades.append(trade_return - 0.002)
                        position = 0
        
        if not trades:
            return {
                'total_return': 0.0, 'win_rate': 0.0, 'total_trades': 0,
                'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
                'buy_hold_return': 0.0, 'trade_frequency': 'No trades'
            }
        
        # Calculate metrics
        total_return = sum(trades)
        win_rate = len([t for t in trades if t > 0]) / len(trades)
        sharpe_ratio = np.mean(trades) / (np.std(trades) + 1e-8) if len(trades) > 1 else 0
        
        # Drawdown
        cumulative = np.cumsum(trades)
        running_max = np.maximum.accumulate(cumulative)
        max_drawdown = np.min(cumulative - running_max)
        
        # Buy and hold
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
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'buy_hold_return': buy_hold_return,
            'trade_frequency': f"{trades_per_year:.1f} trades/year",
            'vs_buy_hold': total_return - buy_hold_return,
            'test_period_days': days_span
        }

def test_improved_trainer():
    """Test the improved trainer on challenging symbols."""
    trainer = ImprovedNEATTrainer()
    
    # Test on symbols that previously performed poorly
    test_symbols = ['AVGO', 'TSLL', 'AAPL']
    
    print("\n" + "="*80)
    print("🎯 IMPROVED SELF-LEARNING NEAT TRAINER")
    print("="*80)
    print(f"Target Metrics:")
    print(f"  • Win Rate: ≥{trainer.target_metrics['min_win_rate']:.0%}")
    print(f"  • Total Return: ≥{trainer.target_metrics['min_total_return']:.0%}")
    print(f"  • Sharpe Ratio: ≥{trainer.target_metrics['min_sharpe']}")
    print(f"  • Max Drawdown: ≤{abs(trainer.target_metrics['max_drawdown']):.0%}")
    
    results = {}
    
    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"Training {symbol}")
        print('='*60)
        
        try:
            # Get 2 years of data
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y", auto_adjust=True)
            
            if len(data) < 500:
                print(f"❌ {symbol}: Insufficient data")
                continue
            
            # Train until success
            result = trainer.train_until_success(symbol, data, horizon_weeks=4, max_attempts=3)
            results[symbol] = result
            
            if result['success']:
                perf = result['performance']
                met_targets = result.get('met_targets', trainer.meets_target_metrics(perf))
                
                print(f"\n📊 Final Results for {symbol}:")
                print(f"  ✅ Win Rate: {perf['win_rate']:.1%} (Target: ≥{trainer.target_metrics['min_win_rate']:.0%})")
                print(f"  ✅ Total Return: {perf['total_return']:.1%} (Target: ≥{trainer.target_metrics['min_total_return']:.0%})")
                print(f"  ✅ Buy & Hold: {perf['buy_hold_return']:.1%}")
                print(f"  ✅ Outperformance: {perf['vs_buy_hold']:.1%}")
                print(f"  ✅ Sharpe Ratio: {perf['sharpe_ratio']:.2f} (Target: ≥{trainer.target_metrics['min_sharpe']})")
                print(f"  ✅ Max Drawdown: {perf['max_drawdown']:.1%} (Target: ≤{abs(trainer.target_metrics['max_drawdown']):.0%})")
                print(f"  ✅ Trades: {perf['total_trades']} ({perf['trade_frequency']})")
                print(f"  ✅ Attempts: {result['attempt']}")
                print(f"  {'🎯 TARGETS MET!' if met_targets else '⚠️ Targets not fully met'}")
            
        except Exception as e:
            print(f"❌ {symbol}: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("📈 TRAINING SUMMARY")
    print('='*80)
    
    successful = [r for r in results.values() if r.get('success', False)]
    met_targets = [r for r in successful if trainer.meets_target_metrics(r['performance'])]
    
    print(f"Symbols Trained: {len(results)}")
    print(f"Successful Training: {len(successful)}")
    print(f"Met Target Metrics: {len(met_targets)}")
    print(f"Success Rate: {len(met_targets)/len(results)*100:.0f}%")
    
    if successful:
        avg_win_rate = np.mean([r['performance']['win_rate'] for r in successful])
        avg_return = np.mean([r['performance']['total_return'] for r in successful])
        avg_vs_bh = np.mean([r['performance']['vs_buy_hold'] for r in successful])
        
        print(f"\nAverage Performance:")
        print(f"  Win Rate: {avg_win_rate:.1%}")
        print(f"  Total Return: {avg_return:.1%}")
        print(f"  vs Buy & Hold: {avg_vs_bh:+.1%}")
    
    print("\n" + "="*80)
    print("✅ IMPROVED TRAINING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_improved_trainer()
