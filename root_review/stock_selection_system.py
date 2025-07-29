"""
Stock Selection & Prioritization System - NEAT Options
Identify and prioritize top stocks for initial agent training based on performance, volatility, and options liquidity.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockMetrics:
    """Stock analysis metrics for prioritization."""
    symbol: str
    market_cap: float
    avg_volume: float
    volatility: float
    options_volume: float
    price_range: Tuple[float, float]
    sector: str
    performance_1y: float
    sharpe_ratio: float
    max_drawdown: float
    liquidity_score: float
    priority_score: float

class StockSelectionSystem:
    """System for selecting and prioritizing stocks for NEAT agent training."""
    
    def __init__(self):
        """Initialize stock selection system."""
        self.logger = logging.getLogger(__name__)
        
        # S&P 500 top stocks by market cap (updated list)
        self.sp500_top_stocks = [
            # Mega Cap (>$1T)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            
            # Large Cap ($100B - $1T)
            'BRK.B', 'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'CVX', 'MA', 'ABBV',
            'PFE', 'AVGO', 'KO', 'PEP', 'TMO', 'COST', 'WMT', 'MRK', 'BAC', 'ADBE',
            'DIS', 'ABT', 'CRM', 'NFLX', 'ACN', 'VZ', 'CMCSA', 'NKE', 'DHR', 'TXN',
            
            # High Volume Tech/Growth
            'ORCL', 'INTC', 'AMD', 'CRM', 'PYPL', 'QCOM', 'UBER', 'SQ', 'ROKU', 'ZM',
            'PLTR', 'HOOD', 'COIN', 'RBLX', 'SNAP', 'TWTR', 'PINS', 'DOCU', 'ZS', 'CRWD'
        ]
        
        # Sector classifications
        self.sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Consumer Discretionary',
            'NVDA': 'Technology', 'META': 'Technology', 'TSLA': 'Consumer Discretionary', 'BRK.B': 'Financial',
            'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'V': 'Financial', 'PG': 'Consumer Staples',
            'JPM': 'Financial', 'HD': 'Consumer Discretionary', 'CVX': 'Energy', 'MA': 'Financial',
            'ABBV': 'Healthcare', 'PFE': 'Healthcare', 'AVGO': 'Technology', 'KO': 'Consumer Staples',
            'PEP': 'Consumer Staples', 'TMO': 'Healthcare', 'COST': 'Consumer Staples', 'WMT': 'Consumer Staples',
            'MRK': 'Healthcare', 'BAC': 'Financial', 'ADBE': 'Technology', 'DIS': 'Communication Services',
            'ABT': 'Healthcare', 'CRM': 'Technology', 'NFLX': 'Communication Services', 'ACN': 'Technology',
            'VZ': 'Communication Services', 'CMCSA': 'Communication Services', 'NKE': 'Consumer Discretionary',
            'DHR': 'Healthcare', 'TXN': 'Technology', 'ORCL': 'Technology', 'INTC': 'Technology',
            'AMD': 'Technology', 'PYPL': 'Financial', 'QCOM': 'Technology', 'UBER': 'Technology',
            'SQ': 'Financial', 'ROKU': 'Technology', 'ZM': 'Technology', 'PLTR': 'Technology',
            'HOOD': 'Financial', 'COIN': 'Financial', 'RBLX': 'Technology', 'SNAP': 'Technology',
            'TWTR': 'Technology', 'PINS': 'Technology', 'DOCU': 'Technology', 'ZS': 'Technology',
            'CRWD': 'Technology'
        }
        
        # Minimum criteria for stock selection
        self.min_criteria = {
            'market_cap': 5e9,      # $5B minimum market cap
            'avg_volume': 1e6,      # 1M shares daily volume
            'price_min': 20,        # $20 minimum price
            'price_max': 1000,      # $1000 maximum price
            'options_volume': 1000   # Minimum daily options volume
        }
    
    def analyze_stock(self, symbol: str, period: str = "2y") -> Optional[StockMetrics]:
        """Analyze a single stock for selection criteria."""
        
        try:
            self.logger.info(f"Analyzing {symbol}...")
            
            # Get stock data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) < 100:  # Need sufficient data
                self.logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Get basic info
            info = ticker.info
            
            # Calculate metrics
            current_price = hist['Close'].iloc[-1]
            
            # Market cap and volume
            market_cap = info.get('marketCap', 0)
            avg_volume = hist['Volume'].mean()
            
            # Price range
            price_52w_high = hist['High'].rolling(252).max().iloc[-1]
            price_52w_low = hist['Low'].rolling(252).min().iloc[-1]
            
            # Volatility (annualized)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # Performance metrics
            performance_1y = (current_price / hist['Close'].iloc[-252]) - 1 if len(hist) >= 252 else 0
            
            # Sharpe ratio (assuming 4% risk-free rate)
            excess_returns = returns - (0.04 / 252)
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative / running_max) - 1
            max_drawdown = drawdown.min()
            
            # Options volume (simulated - in production would use real options data)
            options_volume = avg_volume * 0.1 * (volatility * 100)  # Rough estimate
            
            # Sector
            sector = self.sector_map.get(symbol, 'Unknown')
            
            # Liquidity score (combination of volume and market cap)
            liquidity_score = np.log10(avg_volume) * np.log10(market_cap) / 100
            
            # Priority score calculation
            priority_score = self._calculate_priority_score(
                market_cap, avg_volume, volatility, options_volume, 
                performance_1y, sharpe_ratio, abs(max_drawdown)
            )
            
            return StockMetrics(
                symbol=symbol,
                market_cap=market_cap,
                avg_volume=avg_volume,
                volatility=volatility,
                options_volume=options_volume,
                price_range=(price_52w_low, price_52w_high),
                sector=sector,
                performance_1y=performance_1y,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                liquidity_score=liquidity_score,
                priority_score=priority_score
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def _calculate_priority_score(self, market_cap: float, avg_volume: float, 
                                volatility: float, options_volume: float,
                                performance_1y: float, sharpe_ratio: float, 
                                max_drawdown: float) -> float:
        """Calculate priority score for stock selection."""
        
        # Normalize metrics (0-100 scale)
        scores = {}
        
        # Market cap score (higher is better, but diminishing returns)
        scores['market_cap'] = min(100, np.log10(market_cap / 1e9) * 20)
        
        # Volume score (higher is better)
        scores['volume'] = min(100, np.log10(avg_volume / 1e6) * 25)
        
        # Volatility score (moderate volatility preferred, 15-35% optimal)
        optimal_vol = 0.25
        vol_penalty = abs(volatility - optimal_vol) / optimal_vol
        scores['volatility'] = max(0, 100 - vol_penalty * 50)
        
        # Options volume score
        scores['options'] = min(100, np.log10(options_volume / 1000) * 20)
        
        # Performance score (positive performance preferred)
        scores['performance'] = min(100, max(0, performance_1y * 100 + 50))
        
        # Sharpe ratio score
        scores['sharpe'] = min(100, max(0, sharpe_ratio * 25 + 50))
        
        # Drawdown score (lower drawdown is better)
        scores['drawdown'] = min(100, max(0, 100 - abs(max_drawdown) * 200))
        
        # Weighted average
        weights = {
            'market_cap': 0.15,
            'volume': 0.20,
            'volatility': 0.15,
            'options': 0.20,
            'performance': 0.10,
            'sharpe': 0.10,
            'drawdown': 0.10
        }
        
        priority_score = sum(scores[metric] * weight for metric, weight in weights.items())
        
        return priority_score
    
    def meets_minimum_criteria(self, metrics: StockMetrics) -> bool:
        """Check if stock meets minimum selection criteria."""
        
        criteria_checks = [
            metrics.market_cap >= self.min_criteria['market_cap'],
            metrics.avg_volume >= self.min_criteria['avg_volume'],
            metrics.price_range[1] >= self.min_criteria['price_min'],
            metrics.price_range[1] <= self.min_criteria['price_max'],
            metrics.options_volume >= self.min_criteria['options_volume']
        ]
        
        return all(criteria_checks)
    
    def analyze_stock_universe(self, symbols: List[str] = None, max_workers: int = 10) -> List[StockMetrics]:
        """Analyze entire stock universe in parallel."""
        
        if symbols is None:
            symbols = self.sp500_top_stocks
        
        self.logger.info(f"Analyzing {len(symbols)} stocks...")
        
        stock_metrics = []
        
        # Parallel analysis
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.analyze_stock, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    metrics = future.result()
                    if metrics and self.meets_minimum_criteria(metrics):
                        stock_metrics.append(metrics)
                        self.logger.info(f"✅ {symbol}: Priority Score {metrics.priority_score:.1f}")
                    else:
                        self.logger.warning(f"❌ {symbol}: Failed criteria or analysis")
                except Exception as e:
                    self.logger.error(f"❌ {symbol}: {str(e)}")
        
        return stock_metrics
    
    def select_top_stocks(self, n: int = 20, sector_diversification: bool = True) -> List[StockMetrics]:
        """Select top N stocks for agent training."""
        
        self.logger.info("🎯 Starting stock selection process...")
        
        # Analyze all stocks
        all_metrics = self.analyze_stock_universe()
        
        # Sort by priority score
        all_metrics.sort(key=lambda x: x.priority_score, reverse=True)
        
        if not sector_diversification:
            return all_metrics[:n]
        
        # Sector diversification (max 25% in any sector)
        selected_stocks = []
        sector_counts = {}
        max_per_sector = max(1, n // 4)  # Max 25% per sector
        
        for metrics in all_metrics:
            sector = metrics.sector
            current_count = sector_counts.get(sector, 0)
            
            if current_count < max_per_sector or len(selected_stocks) < n * 0.8:
                selected_stocks.append(metrics)
                sector_counts[sector] = current_count + 1
                
                if len(selected_stocks) >= n:
                    break
        
        # Fill remaining slots if needed
        if len(selected_stocks) < n:
            remaining = [m for m in all_metrics if m not in selected_stocks]
            selected_stocks.extend(remaining[:n - len(selected_stocks)])
        
        self.logger.info(f"✅ Selected {len(selected_stocks)} stocks for training")
        
        return selected_stocks[:n]
    
    def create_training_schedule(self, selected_stocks: List[StockMetrics], 
                               total_resources: int = 4) -> Dict[str, Dict]:
        """Create training schedule and resource allocation."""
        
        schedule = {
            'training_batches': [],
            'resource_allocation': {},
            'estimated_time': {},
            'priority_order': []
        }
        
        # Sort by priority for training order
        stocks_by_priority = sorted(selected_stocks, key=lambda x: x.priority_score, reverse=True)
        
        # Create batches for parallel training
        batch_size = max(1, total_resources)
        batches = [stocks_by_priority[i:i + batch_size] 
                  for i in range(0, len(stocks_by_priority), batch_size)]
        
        for i, batch in enumerate(batches):
            batch_info = {
                'batch_id': i + 1,
                'symbols': [stock.symbol for stock in batch],
                'estimated_time_hours': 2 + (len(batch) * 0.5),  # Base time + per stock
                'resource_allocation': {
                    stock.symbol: {
                        'cpu_cores': 1,
                        'memory_gb': 2,
                        'priority': stock.priority_score,
                        'generations': 150 if stock.priority_score > 80 else 100
                    }
                    for stock in batch
                }
            }
            schedule['training_batches'].append(batch_info)
        
        # Overall resource allocation
        for stock in stocks_by_priority:
            schedule['resource_allocation'][stock.symbol] = {
                'priority_rank': stocks_by_priority.index(stock) + 1,
                'training_generations': 150 if stock.priority_score > 80 else 100,
                'validation_split': 0.2,
                'test_split': 0.1,
                'expected_accuracy_target': 0.80,
                'profit_target_multiplier': 2.0  # 2x S&P 500
            }
        
        # Priority order
        schedule['priority_order'] = [stock.symbol for stock in stocks_by_priority]
        
        # Estimated total time
        total_time = sum(batch['estimated_time_hours'] for batch in schedule['training_batches'])
        schedule['estimated_total_time_hours'] = total_time
        
        return schedule
    
    def generate_selection_report(self, selected_stocks: List[StockMetrics], 
                                training_schedule: Dict) -> Dict:
        """Generate comprehensive stock selection report."""
        
        report = {
            'selection_date': datetime.now().isoformat(),
            'total_stocks_analyzed': len(self.sp500_top_stocks),
            'stocks_selected': len(selected_stocks),
            'selection_criteria': self.min_criteria,
            'sector_distribution': {},
            'performance_metrics': {},
            'training_schedule': training_schedule,
            'selected_stocks': []
        }
        
        # Sector distribution
        sectors = [stock.sector for stock in selected_stocks]
        for sector in set(sectors):
            report['sector_distribution'][sector] = sectors.count(sector)
        
        # Performance metrics
        priority_scores = [stock.priority_score for stock in selected_stocks]
        volatilities = [stock.volatility for stock in selected_stocks]
        market_caps = [stock.market_cap for stock in selected_stocks]
        
        report['performance_metrics'] = {
            'avg_priority_score': np.mean(priority_scores),
            'min_priority_score': np.min(priority_scores),
            'max_priority_score': np.max(priority_scores),
            'avg_volatility': np.mean(volatilities),
            'avg_market_cap': np.mean(market_caps),
            'total_market_cap': np.sum(market_caps)
        }
        
        # Selected stocks details
        for stock in selected_stocks:
            stock_info = {
                'symbol': stock.symbol,
                'sector': stock.sector,
                'priority_score': stock.priority_score,
                'market_cap': stock.market_cap,
                'avg_volume': stock.avg_volume,
                'volatility': stock.volatility,
                'performance_1y': stock.performance_1y,
                'sharpe_ratio': stock.sharpe_ratio,
                'max_drawdown': stock.max_drawdown,
                'liquidity_score': stock.liquidity_score
            }
            report['selected_stocks'].append(stock_info)
        
        return report
    
    def save_selection_results(self, selected_stocks: List[StockMetrics], 
                             training_schedule: Dict, filename: str = None):
        """Save selection results to file."""
        
        if filename is None:
            filename = f"data/stock_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Generate report
        report = self.generate_selection_report(selected_stocks, training_schedule)
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📄 Selection results saved to: {filename}")
        
        return filename


def run_stock_selection():
    """Run complete stock selection process."""
    
    print("🎯 NEAT Options System - Stock Selection & Prioritization")
    print("=" * 60)
    
    # Initialize system
    selector = StockSelectionSystem()
    
    try:
        # Select top 20 stocks
        print("📊 Analyzing stock universe...")
        selected_stocks = selector.select_top_stocks(n=20, sector_diversification=True)
        
        if not selected_stocks:
            print("❌ No stocks met selection criteria")
            return None
        
        # Create training schedule
        print("📅 Creating training schedule...")
        training_schedule = selector.create_training_schedule(selected_stocks)
        
        # Display results
        print(f"\n✅ Selected {len(selected_stocks)} stocks for agent training:")
        print("-" * 60)
        
        for i, stock in enumerate(selected_stocks, 1):
            print(f"{i:2d}. {stock.symbol:6s} | {stock.sector:20s} | Score: {stock.priority_score:5.1f} | "
                  f"Vol: {stock.volatility:5.1%} | MCap: ${stock.market_cap/1e9:5.1f}B")
        
        print(f"\n📊 Sector Distribution:")
        sector_dist = {}
        for stock in selected_stocks:
            sector_dist[stock.sector] = sector_dist.get(stock.sector, 0) + 1
        
        for sector, count in sorted(sector_dist.items()):
            percentage = count / len(selected_stocks) * 100
            print(f"  {sector:20s}: {count:2d} stocks ({percentage:4.1f}%)")
        
        print(f"\n⏱️  Training Schedule:")
        print(f"  Total batches: {len(training_schedule['training_batches'])}")
        print(f"  Estimated time: {training_schedule['estimated_total_time_hours']:.1f} hours")
        print(f"  Priority order: {', '.join(training_schedule['priority_order'][:10])}...")
        
        # Save results
        filename = selector.save_selection_results(selected_stocks, training_schedule)
        print(f"\n📄 Results saved to: {filename}")
        
        return {
            'selected_stocks': selected_stocks,
            'training_schedule': training_schedule,
            'report_file': filename
        }
        
    except Exception as e:
        print(f"❌ Stock selection failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_stock_selection()
