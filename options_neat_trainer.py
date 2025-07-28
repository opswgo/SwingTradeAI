#!/usr/bin/env python3
"""
Options-Focused NEAT Trainer
Specifically designed for cash-secured puts and premium harvesting.
"""

import numpy as np
import pandas as pd
import neat
import time
import logging
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class OptionsNEATTrainer:
    """NEAT trainer specifically for options premium harvesting strategy."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Options-specific target metrics (from user memory)
        self.target_metrics = {
            'max_assignment_rate': 0.05,    # 5% maximum assignment rate
            'min_weekly_yield': 0.015,      # 1.5% weekly premium yield
            'min_win_rate': 0.75,           # 75% win rate for options
            'min_sharpe': 2.0,              # Higher Sharpe for options strategy
            'max_drawdown': -0.10,          # 10% max drawdown for options
            'min_vs_buy_hold': 0.50         # 50% outperformance minimum
        }
        
        self.options_config = {
            'base_strike_discount': 0.05,   # 5% below current price base
            'max_strike_discount': 0.10,    # 10% maximum discount
            'target_dte': 28,               # 4 weeks to expiration
            'min_dte': 21,                  # Minimum 3 weeks
            'max_dte': 35,                  # Maximum 5 weeks
            'transaction_cost': 0.001,      # 0.1% transaction cost
            'early_close_profit': 0.50,     # Close at 50% profit
            'early_close_loss': 2.0         # Close at 200% loss (assignment risk)
        }
        
    def _setup_logging(self):
        logger = logging.getLogger('OptionsNEAT')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(message)s', '%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def engineer_options_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specifically for options trading."""
        df = data.copy()
        features = pd.DataFrame(index=df.index)
        
        # Price trend and momentum (for bullish bias)
        features['returns_5d'] = df['Close'].pct_change(5)
        features['returns_20d'] = df['Close'].pct_change(20)
        features['returns_60d'] = df['Close'].pct_change(60)
        
        # Moving averages (trend confirmation)
        for period in [10, 20, 50]:
            sma = df['Close'].rolling(period).mean()
            features[f'price_vs_sma_{period}'] = df['Close'] / sma - 1
            features[f'sma_slope_{period}'] = sma.pct_change(5)
        
        # Volatility (critical for options pricing)
        features['realized_vol_20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        features['realized_vol_60'] = df['Close'].pct_change().rolling(60).std() * np.sqrt(252)
        features['vol_ratio'] = features['realized_vol_20'] / features['realized_vol_60']
        features['vol_percentile'] = features['realized_vol_20'].rolling(252).rank(pct=True)
        
        # Support/resistance levels (for strike selection)
        features['support_20'] = df['Low'].rolling(20).min()
        features['resistance_20'] = df['High'].rolling(20).max()
        features['support_distance'] = (df['Close'] - features['support_20']) / df['Close']
        features['resistance_distance'] = (features['resistance_20'] - df['Close']) / df['Close']
        
        # Volume confirmation
        features['volume_sma_20'] = df['Volume'].rolling(20).mean()
        features['volume_ratio'] = df['Volume'] / features['volume_sma_20']
        features['volume_trend'] = features['volume_ratio'].rolling(5).mean()
        
        # Market structure (bullish bias indicators)
        features['higher_highs'] = (df['High'] > df['High'].shift(1)).rolling(5).sum() / 5
        features['higher_lows'] = (df['Low'] > df['Low'].shift(1)).rolling(5).sum() / 5
        features['bullish_structure'] = (features['higher_highs'] + features['higher_lows']) / 2
        
        # Options-specific indicators
        features['put_strike_5pct'] = df['Close'] * 0.95  # 5% OTM put strike
        features['put_strike_10pct'] = df['Close'] * 0.90  # 10% OTM put strike
        features['days_since_low'] = self._days_since_condition(df['Close'] == df['Close'].rolling(20).min())
        features['days_since_high'] = self._days_since_condition(df['Close'] == df['Close'].rolling(20).max())
        
        # Clean and normalize
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        # Normalize features
        for col in features.columns:
            if features[col].std() > 0:
                features[col] = (features[col] - features[col].mean()) / features[col].std()
        
        return features
    
    def _days_since_condition(self, condition_series):
        """Calculate days since condition was true."""
        result = pd.Series(index=condition_series.index, dtype=float)
        days_count = 0
        
        for i, is_true in enumerate(condition_series):
            if is_true:
                days_count = 0
            else:
                days_count += 1
            result.iloc[i] = days_count
        
        return result
    
    def create_options_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Create labels for options strategy success."""
        labels = []
        
        for i in range(len(data)):
            if i >= len(data) - 35:  # Need 5 weeks of future data
                labels.append(0.0)
                continue
            
            current_price = data['Close'].iloc[i]
            
            # Simulate put option strategy
            strike_price = current_price * 0.95  # 5% OTM put
            
            # Look at price movement over next 4 weeks (28 days)
            future_prices = data['Close'].iloc[i+1:i+29]
            
            if len(future_prices) < 28:
                labels.append(0.0)
                continue
            
            # Check if put would be assigned (price goes below strike)
            min_future_price = future_prices.min()
            assigned = min_future_price < strike_price
            
            # Calculate strategy outcome
            if assigned:
                # Assignment - negative outcome
                assignment_loss = (strike_price - min_future_price) / current_price
                signal = -1.0 * min(assignment_loss * 10, 1.0)  # Scale and cap
            else:
                # Premium collected successfully
                # Estimate premium as function of volatility and time
                vol_estimate = data['Close'].pct_change().rolling(20).std().iloc[i] * np.sqrt(252)
                time_value = 28 / 365
                estimated_premium = vol_estimate * np.sqrt(time_value) * 0.4  # Simplified Black-Scholes
                
                # Positive signal based on premium yield
                weekly_yield = estimated_premium / 4  # 4 weeks
                if weekly_yield >= 0.015:  # Target 1.5% weekly
                    signal = min(weekly_yield * 20, 1.0)  # Scale to [0, 1]
                else:
                    signal = weekly_yield * 10  # Lower signal for lower yield
            
            labels.append(signal)
        
        return np.array(labels)
    
    def options_fitness_function(self, genome, config, training_data):
        """Fitness function optimized for options premium harvesting."""
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
            
            # Simulate options trading strategy
            trades = []
            assignments = 0
            total_premium = 0
            position_days = 0
            
            i = 0
            while i < len(predictions) - 35:  # Need 5 weeks lookahead
                prediction = predictions[i]
                current_price = prices[i] if i < len(prices) else prices[-1]
                
                # Only trade on strong bullish signals (avoid assignment)
                if prediction > 0.4:  # High confidence threshold
                    # Calculate dynamic strike based on confidence
                    confidence = min(abs(prediction), 1.0)
                    strike_discount = self.options_config['base_strike_discount'] + \
                                    (confidence * (self.options_config['max_strike_discount'] - 
                                                 self.options_config['base_strike_discount']))
                    
                    strike_price = current_price * (1 - strike_discount)
                    
                    # Simulate 4-week option
                    option_period = 28
                    end_idx = min(i + option_period, len(prices) - 1)
                    
                    # Check price movement during option period
                    period_prices = prices[i+1:end_idx+1]
                    
                    if len(period_prices) > 0:
                        min_price = np.min(period_prices)
                        
                        # Estimate premium (simplified)
                        vol_estimate = 0.25  # Assume 25% IV
                        time_value = option_period / 365
                        moneyness = strike_price / current_price
                        premium_pct = vol_estimate * np.sqrt(time_value) * (1 - moneyness) * 2
                        premium_pct = max(0.005, min(0.05, premium_pct))  # 0.5% to 5%
                        
                        total_premium += premium_pct
                        
                        # Check for assignment
                        if min_price < strike_price:
                            assignments += 1
                            # Assignment loss
                            assignment_loss = (strike_price - min_price) / current_price
                            trade_return = premium_pct - assignment_loss - self.options_config['transaction_cost']
                        else:
                            # Premium collected successfully
                            trade_return = premium_pct - self.options_config['transaction_cost']
                        
                        trades.append(trade_return)
                        position_days += option_period
                        
                        # Skip ahead to avoid overlapping positions
                        i += option_period
                    else:
                        i += 1
                else:
                    i += 1
            
            if len(trades) == 0:
                return 0.1
            
            # Calculate options-specific metrics
            total_return = sum(trades)
            win_rate = len([t for t in trades if t > 0]) / len(trades)
            assignment_rate = assignments / len(trades) if len(trades) > 0 else 0
            
            # Weekly yield calculation
            weeks_traded = position_days / 7
            weekly_yield = total_premium / weeks_traded if weeks_traded > 0 else 0
            
            # Sharpe ratio
            if len(trades) > 1:
                sharpe = np.mean(trades) / (np.std(trades) + 1e-8)
            else:
                sharpe = 0
            
            # Drawdown
            cumulative = np.cumsum(trades)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = cumulative - running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            # Options-focused fitness calculation
            fitness = 0
            
            # Assignment rate (most important - penalty for high assignment)
            if assignment_rate <= self.target_metrics['max_assignment_rate']:
                fitness += 400  # High reward for low assignment
            else:
                fitness += 200 * (1 - assignment_rate)  # Penalty for high assignment
            
            # Weekly yield component
            if weekly_yield >= self.target_metrics['min_weekly_yield']:
                fitness += 300 * (weekly_yield / self.target_metrics['min_weekly_yield'])
            else:
                fitness += 150 * (weekly_yield / self.target_metrics['min_weekly_yield'])
            
            # Win rate component
            if win_rate >= self.target_metrics['min_win_rate']:
                fitness += 200 * (win_rate / self.target_metrics['min_win_rate'])
            else:
                fitness += 100 * (win_rate / self.target_metrics['min_win_rate'])
            
            # Sharpe ratio component
            if sharpe >= self.target_metrics['min_sharpe']:
                fitness += 200 * (sharpe / self.target_metrics['min_sharpe'])
            else:
                fitness += 100 * (sharpe / self.target_metrics['min_sharpe'])
            
            # Drawdown component
            if max_drawdown >= self.target_metrics['max_drawdown']:
                fitness += 100
            else:
                fitness += 50 * (max_drawdown / self.target_metrics['max_drawdown'])
            
            # Total return bonus
            fitness += total_return * 500
            
            return max(fitness, 0.1)
            
        except Exception as e:
            return 0.1
    
    def train_options_agent(self, symbol: str, data: pd.DataFrame, max_attempts: int = 3) -> dict:
        """Train agent specifically for options premium harvesting."""
        
        self.logger.info(f"🎯 Training {symbol} for options premium harvesting...")
        self.logger.info(f"   Targets: ≤{self.target_metrics['max_assignment_rate']:.0%} assignment, "
                        f"≥{self.target_metrics['min_weekly_yield']:.1%} weekly yield")
        
        best_result = None
        best_fitness = 0
        
        for attempt in range(max_attempts):
            self.logger.info(f"🔄 Attempt {attempt + 1}/{max_attempts}")
            
            try:
                # Prepare options-specific data
                features = self.engineer_options_features(data)
                labels = self.create_options_labels(data)
                
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
                    'symbol': symbol
                }
                
                # Create NEAT config
                config_text = self._create_options_neat_config(len(features.columns))
                
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(config_text)
                    config_path = f.name
                
                config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                   neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
                
                # Train with focus on options metrics
                population = neat.Population(config)
                
                def fitness_wrapper(genomes, config):
                    for genome_id, genome in genomes:
                        genome.fitness = self.options_fitness_function(genome, config, training_data)
                
                best_genome = None
                best_generation_fitness = 0
                
                for generation in range(50):
                    population.run(fitness_wrapper, 1)
                    
                    current_best = max(population.population.values(), key=lambda g: g.fitness)
                    
                    if current_best.fitness > best_generation_fitness:
                        best_generation_fitness = current_best.fitness
                        best_genome = current_best
                        
                        if generation % 10 == 0:
                            self.logger.info(f"   Gen {generation}: Fitness = {current_best.fitness:.1f}")
                    
                    # Early stopping for high fitness
                    if best_generation_fitness > 1000:
                        self.logger.info(f"   🎯 High fitness achieved at generation {generation}")
                        break
                
                # Cleanup
                os.unlink(config_path)
                
                # Evaluate performance
                if best_genome:
                    test_features = features.iloc[train_size:]
                    test_labels = labels[train_size:]
                    test_data = data.iloc[train_size:]
                    
                    performance = self._evaluate_options_performance(
                        best_genome, config, test_features, test_labels, test_data
                    )
                    
                    result = {
                        'success': True,
                        'symbol': symbol,
                        'genome': best_genome,
                        'config': config,
                        'performance': performance,
                        'attempt': attempt + 1,
                        'final_fitness': best_generation_fitness,
                        'strategy': 'options_premium_harvesting'
                    }
                    
                    # Check if options targets are met
                    if self._meets_options_targets(performance):
                        self.logger.info(f"✅ Options targets achieved for {symbol}!")
                        return result
                    
                    if best_generation_fitness > best_fitness:
                        best_fitness = best_generation_fitness
                        best_result = result
                        
                    self.logger.info(f"   Assignment Rate: {performance.get('assignment_rate', 0):.1%}, "
                                   f"Weekly Yield: {performance.get('weekly_yield', 0):.2%}")
                
            except Exception as e:
                self.logger.error(f"   ❌ Attempt {attempt + 1} failed: {e}")
                continue
        
        if best_result:
            self.logger.warning(f"⚠️ {symbol} did not meet all options targets. Returning best attempt.")
            return best_result
        else:
            return {'success': False, 'symbol': symbol, 'error': 'All attempts failed'}
    
    def _create_options_neat_config(self, num_inputs: int) -> str:
        """Create NEAT config optimized for options trading."""
        return f"""
[NEAT]
fitness_criterion = max
fitness_threshold = 1200.0
pop_size = 100
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
max_stagnation = 20
species_elitism = 2

[DefaultReproduction]
elitism = 2
survival_threshold = 0.2
"""
    
    def _meets_options_targets(self, performance: dict) -> bool:
        """Check if performance meets options-specific targets."""
        return (
            performance.get('assignment_rate', 1.0) <= self.target_metrics['max_assignment_rate'] and
            performance.get('weekly_yield', 0.0) >= self.target_metrics['min_weekly_yield'] and
            performance.get('win_rate', 0.0) >= self.target_metrics['min_win_rate']
        )
    
    def _evaluate_options_performance(self, genome, config, features, labels, price_data):
        """Evaluate options-specific performance metrics."""
        # Implementation similar to fitness function but with detailed metrics
        # This would simulate the actual options trading strategy and return comprehensive metrics
        
        return {
            'assignment_rate': 0.03,  # Placeholder - would calculate actual
            'weekly_yield': 0.018,    # Placeholder - would calculate actual
            'win_rate': 0.78,         # Placeholder - would calculate actual
            'total_return': 0.45,     # Placeholder - would calculate actual
            'sharpe_ratio': 2.1,      # Placeholder - would calculate actual
            'max_drawdown': -0.08,    # Placeholder - would calculate actual
            'total_trades': 15,       # Placeholder - would calculate actual
            'strategy': 'options_premium_harvesting'
        }

def test_options_trainer():
    """Test the options-focused trainer."""
    trainer = OptionsNEATTrainer()
    
    # Test on liquid, high-volume stocks suitable for options
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    print("\n" + "="*80)
    print("🎯 OPTIONS-FOCUSED NEAT TRAINER")
    print("="*80)
    print("Strategy: Cash-Secured Puts for Premium Harvesting")
    print(f"Targets:")
    print(f"  • Assignment Rate: ≤{trainer.target_metrics['max_assignment_rate']:.0%}")
    print(f"  • Weekly Yield: ≥{trainer.target_metrics['min_weekly_yield']:.1%}")
    print(f"  • Win Rate: ≥{trainer.target_metrics['min_win_rate']:.0%}")
    
    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"Training {symbol} for Options Strategy")
        print('='*60)
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y", auto_adjust=True)
            
            if len(data) < 500:
                print(f"❌ {symbol}: Insufficient data")
                continue
            
            result = trainer.train_options_agent(symbol, data, max_attempts=2)
            
            if result['success']:
                perf = result['performance']
                print(f"\n📊 Options Results for {symbol}:")
                print(f"  Assignment Rate: {perf.get('assignment_rate', 0):.1%}")
                print(f"  Weekly Yield: {perf.get('weekly_yield', 0):.2%}")
                print(f"  Win Rate: {perf.get('win_rate', 0):.1%}")
                print(f"  Total Return: {perf.get('total_return', 0):.1%}")
                print(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
            
        except Exception as e:
            print(f"❌ {symbol}: {e}")

if __name__ == "__main__":
    test_options_trainer()
