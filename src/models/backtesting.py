"""
Backtesting Engine for Options Trading Strategies

This module provides comprehensive backtesting capabilities with walk-forward
testing, performance metrics, and user-friendly result visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

# Local imports
from ..core.config import Settings
from ..core.logger import log_user_action, log_performance_metric
from loguru import logger
from .regime_model import RegimeAwareModel, ModelPrediction


@dataclass
class Trade:
    """Structure for individual trade records."""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    signal: str  # 'BUY_CALL', 'BUY_PUT', 'SELL', 'HOLD'
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    pnl: Optional[float]
    return_pct: Optional[float]
    confidence: float
    regime: str
    reason: str  # Trade reasoning
    is_open: bool = True


@dataclass
class BacktestResults:
    """Structure for backtest performance results."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    trades: List[Trade]
    equity_curve: pd.DataFrame
    regime_performance: Dict[str, Dict[str, float]]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float


class BacktestingEngine:
    """
    Comprehensive backtesting engine for options trading strategies.
    
    Provides walk-forward testing, performance analysis, and detailed
    trade-by-trade results with regime-aware performance breakdown.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = RegimeAwareModel(settings)
        
        # Backtesting parameters
        self.initial_capital = 100000  # $100k starting capital
        self.position_size = 0.02  # 2% risk per trade
        self.commission = 2.0  # $2 per options contract
        self.slippage = 0.001  # 0.1% slippage
        
        # Options parameters
        self.options_multiplier = 100  # Standard options contract size
        self.min_days_to_expiry = 7  # Minimum days before expiry to hold
        self.max_days_to_expiry = 45  # Maximum days to expiry when entering
        
        # Risk management
        self.max_position_size = 0.05  # Max 5% of capital per trade
        self.stop_loss_pct = 0.50  # 50% stop loss on options
        self.take_profit_pct = 2.00  # 200% take profit on options
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the backtesting engine."""
        try:
            logger.info("📈 Initializing Backtesting Engine...")
            
            # Initialize model
            model_result = await self.model.initialize()
            if not model_result['success']:
                raise Exception(f"Model initialization failed: {model_result.get('error')}")
            
            logger.info("✅ Backtesting Engine initialized successfully")
            return {
                'success': True,
                'initial_capital': self.initial_capital,
                'position_size': self.position_size,
                'model_ready': model_result['is_trained']
            }
            
        except Exception as e:
            logger.error(f"❌ Backtesting engine initialization failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def run_backtest(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime,
        walk_forward: bool = True
    ) -> BacktestResults:
        """
        Run comprehensive backtest with walk-forward analysis.
        
        Args:
            symbol: Stock symbol to backtest
            start_date: Backtest start date
            end_date: Backtest end date
            walk_forward: Whether to use walk-forward testing
            
        Returns:
            Detailed backtest results
        """
        log_user_action("run_backtest", {
            "symbol": symbol, 
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "walk_forward": walk_forward
        })
        
        try:
            logger.info(f"🚀 Starting backtest for {symbol} ({start_date.date()} to {end_date.date()})")
            
            # Prepare historical data
            historical_data = await self._prepare_backtest_data(symbol, start_date, end_date)
            if historical_data.empty:
                raise ValueError(f"No historical data available for {symbol}")
            
            # Initialize tracking variables
            trades = []
            equity_curve = []
            current_capital = self.initial_capital
            open_positions = []
            
            # Walk-forward backtesting
            if walk_forward:
                results = await self._walk_forward_backtest(
                    symbol, historical_data, start_date, end_date
                )
            else:
                results = await self._simple_backtest(
                    symbol, historical_data, start_date, end_date
                )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(results['trades'], results['equity_curve'])
            
            # Ensure equity curve exists
            logger.info(f"📊 Equity curve status: {len(results['equity_curve'])} data points, empty={results['equity_curve'].empty}")
            if results['equity_curve'].empty:
                # Create minimal equity curve if none exists
                logger.info("🔧 Creating minimal equity curve as fallback")
                results['equity_curve'] = pd.DataFrame({
                    'capital': [self.initial_capital],
                    'daily_return': [0.0]
                }, index=[start_date])
                logger.info(f"✅ Fallback equity curve created: {len(results['equity_curve'])} data points")
            
            # Create backtest results
            backtest_results = BacktestResults(
                total_trades=len(results['trades']),
                winning_trades=len([t for t in results['trades'] if t.pnl and t.pnl > 0]),
                losing_trades=len([t for t in results['trades'] if t.pnl and t.pnl < 0]),
                win_rate=performance_metrics['win_rate'],
                total_return=performance_metrics['total_return'],
                annualized_return=performance_metrics['annualized_return'],
                max_drawdown=performance_metrics['max_drawdown'],
                sharpe_ratio=performance_metrics['sharpe_ratio'],
                sortino_ratio=performance_metrics['sortino_ratio'],
                profit_factor=performance_metrics['profit_factor'],
                avg_win=performance_metrics['avg_win'],
                avg_loss=performance_metrics['avg_loss'],
                trades=results['trades'],
                equity_curve=results['equity_curve'],
                regime_performance=performance_metrics['regime_performance'],
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                final_capital=results['equity_curve']['capital'].iloc[-1] if not results['equity_curve'].empty else self.initial_capital
            )
            
            # Log results
            log_performance_metric("backtest_total_return", performance_metrics['total_return'])
            log_performance_metric("backtest_win_rate", performance_metrics['win_rate'])
            log_performance_metric("backtest_sharpe_ratio", performance_metrics['sharpe_ratio'])
            
            logger.info(f"🎉 Backtest complete: {performance_metrics['total_return']:.2%} return, {performance_metrics['win_rate']:.1%} win rate")
            
            log_user_action("run_backtest", {
                "symbol": symbol,
                "total_return": performance_metrics['total_return'],
                "win_rate": performance_metrics['win_rate'],
                "total_trades": len(results['trades'])
            })
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"❌ Backtest failed for {symbol}: {str(e)}")
            log_user_action("run_backtest", {"symbol": symbol, "error": str(e)}, success=False)
            raise
    
    async def _prepare_backtest_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Prepare historical data for backtesting."""
        try:
            # Get historical features from feature store
            from ..features.feature_store import FeatureStore
            feature_store = FeatureStore(self.settings)
            await feature_store.initialize()
            
            # Calculate required lookback for features
            lookback_days = (end_date - start_date).days + 100  # Extra buffer for indicators
            
            features_result = await feature_store.retrieve_features(symbol, limit=lookback_days)
            features_df = features_result.get('data', pd.DataFrame()) if features_result.get('success') else pd.DataFrame()
            
            if features_df.empty:
                logger.warning(f"⚠️ No stored features for {symbol}, attempting to fetch real market data...")
                
                # Try to fetch and calculate real market data
                from ..data.pipeline import DataPipeline
                pipeline = DataPipeline(self.settings)
                await pipeline.initialize()
                
                try:
                    # Fetch real historical data
                    historical_data, _ = await pipeline.fetch_historical_data(symbol, period="2y")
                    if historical_data.empty:
                        raise ValueError(f"No real market data available for {symbol}")
                    
                    # Calculate technical indicators on real data
                    from ..features.technical_analysis import TechnicalAnalysis
                    technical_analyzer = TechnicalAnalysis()
                    enhanced_data, _ = technical_analyzer.calculate_all_indicators(historical_data)
                    
                    # Add regime detection to real data
                    from ..features.market_regime import MarketRegimeDetector
                    regime_detector = MarketRegimeDetector()
                    regime_result = await regime_detector.detect_current_regime(enhanced_data)
                    enhanced_data['regime'] = regime_result.get('regime', 'bull_trending')
                    enhanced_data['regime_confidence'] = regime_result.get('confidence', 0.8)
                    
                    # Store the real features for future use
                    await feature_store.store_features(symbol, enhanced_data)
                    
                    logger.info(f"✅ Successfully fetched and processed real market data for {symbol}")
                    return enhanced_data
                    
                except Exception as e:
                    logger.error(f"❌ Failed to fetch real market data for {symbol}: {str(e)}")
                    raise ValueError(f"Cannot proceed with backtesting - no real market data available for {symbol}. Error: {str(e)}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare backtest data: {str(e)}")
            raise
    
    async def _walk_forward_backtest(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Perform walk-forward backtesting."""
        logger.info("🔄 Running walk-forward backtest...")
        
        trades = []
        equity_curve = []
        current_capital = self.initial_capital
        
        # Walk-forward parameters
        training_window = 252  # 1 year training window
        rebalance_frequency = 21  # Retrain every 21 days (monthly)
        
        # Create walk-forward windows
        test_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        test_dates = test_dates[test_dates.isin(data.index)]
        
        for i, test_date in enumerate(test_dates):
            if i % rebalance_frequency == 0:  # Retrain model
                training_end = test_date - timedelta(days=1)
                training_start = training_end - timedelta(days=training_window)
                
                # Train model on historical data
                if not self.model.is_trained:
                    logger.info(f"🎯 Training model for period ending {training_end.date()}")
                    await self.model.train_model(symbol, lookback_days=training_window)
            
            # Make prediction for test date
            if test_date in data.index:
                test_data = data.loc[[test_date]]
                
                try:
                    prediction = await self.model.predict(symbol, explain=True)
                    
                    # Simulate trade execution
                    trade_result = await self._simulate_trade(
                        symbol, test_date, test_data, prediction, current_capital
                    )
                    
                    if trade_result:
                        trades.append(trade_result)
                        current_capital += trade_result.pnl or 0
                    
                    # Record equity curve
                    equity_curve.append({
                        'date': test_date,
                        'capital': current_capital,
                        'daily_return': (current_capital / self.initial_capital - 1) if current_capital > 0 else -1
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ Prediction failed for {test_date}: {str(e)}")
                    continue
        
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df.set_index('date', inplace=True)
        
        return {
            'trades': trades,
            'equity_curve': equity_df
        }
    
    async def _simple_backtest(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Perform simple backtesting without walk-forward."""
        logger.info("📊 Running simple backtest...")
        
        # Train model once on all historical data
        if not self.model.is_trained:
            logger.info("🎯 Training model on historical data...")
            await self.model.train_model(symbol, lookback_days=500)
        
        trades = []
        equity_curve = []
        current_capital = self.initial_capital
        
        # Filter data for backtest period
        backtest_data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        for date, row in backtest_data.iterrows():
            try:
                # Make prediction
                prediction = await self.model.predict(symbol, explain=True)
                
                # Simulate trade
                trade_result = await self._simulate_trade(
                    symbol, date, pd.DataFrame([row]), prediction, current_capital
                )
                
                if trade_result:
                    trades.append(trade_result)
                    current_capital += trade_result.pnl or 0
                
                # Record equity curve
                equity_curve.append({
                    'date': date,
                    'capital': current_capital,
                    'daily_return': (current_capital / self.initial_capital - 1) if current_capital > 0 else -1
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Simulation failed for {date}: {str(e)}")
                continue
        
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df.set_index('date', inplace=True)
        
        return {
            'trades': trades,
            'equity_curve': equity_df
        }
    
    async def _simulate_trade(
        self, 
        symbol: str, 
        date: datetime, 
        data: pd.DataFrame, 
        prediction: ModelPrediction, 
        current_capital: float
    ) -> Optional[Trade]:
        """Simulate individual trade execution."""
        if prediction.signal == 'HOLD' or prediction.confidence < 0.6:
            return None
        
        # Get current price
        current_price = data['Close'].iloc[0] if 'Close' in data.columns else 100.0
        
        # Calculate position size
        risk_amount = current_capital * self.position_size
        
        # Simulate options pricing (simplified)
        if prediction.signal == 'BUY_CALL':
            # Call option premium (simplified Black-Scholes approximation)
            strike_price = current_price * 1.02  # 2% OTM
            option_premium = current_price * 0.03 * prediction.confidence  # Simplified premium
            contracts = int(risk_amount / (option_premium * self.options_multiplier))
            
        elif prediction.signal == 'BUY_PUT':
            # Put option premium
            strike_price = current_price * 0.98  # 2% OTM
            option_premium = current_price * 0.03 * prediction.confidence
            contracts = int(risk_amount / (option_premium * self.options_multiplier))
            
        else:
            return None
        
        if contracts <= 0:
            return None
        
        # Calculate entry cost
        entry_cost = contracts * option_premium * self.options_multiplier
        entry_cost += contracts * self.commission  # Add commission
        
        # Simulate holding period (5-10 days)
        holding_days = np.random.randint(5, 11)
        exit_date = date + timedelta(days=holding_days)
        
        # Simulate exit price (simplified)
        price_change = np.random.normal(prediction.expected_return, 0.05)  # Add noise
        exit_price = current_price * (1 + price_change)
        
        # Calculate option value at exit (simplified)
        if prediction.signal == 'BUY_CALL':
            intrinsic_value = max(0, exit_price - strike_price)
            time_decay = 0.8  # Assume 20% time decay
            exit_option_value = intrinsic_value + (option_premium - intrinsic_value) * time_decay
        else:  # BUY_PUT
            intrinsic_value = max(0, strike_price - exit_price)
            time_decay = 0.8
            exit_option_value = intrinsic_value + (option_premium - intrinsic_value) * time_decay
        
        # Calculate P&L
        exit_value = contracts * exit_option_value * self.options_multiplier
        exit_cost = contracts * self.commission  # Exit commission
        
        pnl = exit_value - entry_cost - exit_cost
        return_pct = pnl / entry_cost if entry_cost > 0 else 0
        
        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_date=date,
            exit_date=exit_date,
            signal=prediction.signal,
            entry_price=option_premium,
            exit_price=exit_option_value,
            quantity=contracts,
            pnl=pnl,
            return_pct=return_pct,
            confidence=prediction.confidence,
            regime=prediction.regime,
            reason=prediction.explanation,
            is_open=False
        )
        
        return trade
    
    def _calculate_performance_metrics(self, trades: List[Trade], equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return self._empty_metrics()
        
        # Basic metrics
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Return metrics
        total_pnl = sum([t.pnl for t in trades if t.pnl])
        total_return = total_pnl / self.initial_capital
        
        # Risk metrics
        if not equity_curve.empty and len(equity_curve) > 1:
            returns = equity_curve['daily_return'].pct_change().dropna()
            
            # Annualized return
            days = len(equity_curve)
            annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
            
            # Drawdown
            cumulative = (1 + equity_curve['daily_return']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Sharpe ratio
            if len(returns) > 1 and returns.std() > 0:
                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            else:
                sharpe_ratio = 0
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 1 and negative_returns.std() > 0:
                sortino_ratio = (returns.mean() * 252) / (negative_returns.std() * np.sqrt(252))
            else:
                sortino_ratio = 0
        else:
            annualized_return = 0
            max_drawdown = 0
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Profit factor
        gross_profit = sum([t.pnl for t in winning_trades]) if winning_trades else 0
        gross_loss = abs(sum([t.pnl for t in losing_trades])) if losing_trades else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Regime performance
        regime_performance = {}
        for regime in set([t.regime for t in trades]):
            regime_trades = [t for t in trades if t.regime == regime]
            regime_wins = [t for t in regime_trades if t.pnl and t.pnl > 0]
            
            regime_performance[regime] = {
                'trades': len(regime_trades),
                'win_rate': len(regime_wins) / len(regime_trades) if regime_trades else 0,
                'avg_return': np.mean([t.return_pct for t in regime_trades if t.return_pct]) if regime_trades else 0,
                'total_pnl': sum([t.pnl for t in regime_trades if t.pnl])
            }
        
        return {
            'win_rate': win_rate,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'regime_performance': regime_performance
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            'win_rate': 0,
            'total_return': 0,
            'annualized_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'regime_performance': {
                'bull_trending': {
                    'trades': 0,
                    'win_rate': 0,
                    'avg_return': 0,
                    'total_pnl': 0
                }
            }
        }
    
    def generate_backtest_report(self, results: BacktestResults) -> str:
        """Generate user-friendly backtest report."""
        report = f"""
📈 BACKTEST RESULTS SUMMARY
{'='*50}

📊 PERFORMANCE OVERVIEW
• Total Return: {results.total_return:.2%}
• Annualized Return: {results.annualized_return:.2%}
• Win Rate: {results.win_rate:.1%}
• Total Trades: {results.total_trades}

💰 TRADE BREAKDOWN
• Winning Trades: {results.winning_trades}
• Losing Trades: {results.losing_trades}
• Average Win: ${results.avg_win:.2f}
• Average Loss: ${results.avg_loss:.2f}
• Profit Factor: {results.profit_factor:.2f}

⚠️ RISK METRICS
• Maximum Drawdown: {results.max_drawdown:.2%}
• Sharpe Ratio: {results.sharpe_ratio:.2f}
• Sortino Ratio: {results.sortino_ratio:.2f}

💼 CAPITAL PROGRESSION
• Initial Capital: ${results.initial_capital:,.2f}
• Final Capital: ${results.final_capital:,.2f}
• Net Profit: ${results.final_capital - results.initial_capital:,.2f}

🌍 REGIME PERFORMANCE
"""
        
        for regime, metrics in results.regime_performance.items():
            report += f"• {regime.title()}: {metrics['win_rate']:.1%} win rate, {metrics['trades']} trades\n"
        
        report += f"""
📅 PERIOD: {results.start_date.date()} to {results.end_date.date()}

{'='*50}
"""
        
        return report
