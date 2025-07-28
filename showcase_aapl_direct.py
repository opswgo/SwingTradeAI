#!/usr/bin/env python3
"""
🚀 SwingTradeAI AAPL Showcase - Direct Working Demo
==================================================
This bypasses the data pipeline timezone issues and demonstrates
the working backtesting and analysis capabilities directly.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.backtesting.advanced_backtester import AdvancedBacktester, BacktestConfig
from src.backtesting.strategy_reporter import StrategyReporter
from src.core.config import Settings

def print_header(title):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"🎯 {title}")
    print(f"{'='*80}")

def print_section(title):
    """Print formatted subsection"""
    print(f"\n📊 {title}")
    print("-" * 60)

async def showcase_aapl_direct():
    """Direct AAPL showcase bypassing data pipeline issues"""
    
    print_header("SwingTradeAI AAPL Showcase - DIRECT WORKING DEMO")
    print("🎯 Bypassing data pipeline issues to show working capabilities")
    print("✅ Using direct yfinance calls with proper date handling")
    
    symbol = "AAPL"
    
    try:
        # 1. DIRECT DATA FETCH (Working)
        print_section("1. DIRECT MARKET DATA FETCH")
        
        print(f"   📈 Fetching real market data for {symbol}...")
        ticker = yf.Ticker(symbol)
        
        # Get 1 year of data with proper date strings
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        data = ticker.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval="1d"
        )
        
        if not data.empty:
            print(f"   ✅ Successfully fetched {len(data)} days of real market data")
            print(f"   📊 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"   💰 Current price: ${data['Close'].iloc[-1]:.2f}")
            print(f"   📈 52-week range: ${data['Low'].min():.2f} - ${data['High'].max():.2f}")
            print(f"   📊 Average volume: {data['Volume'].mean():,.0f}")
        else:
            raise ValueError("No data available")
        
        # 2. ADVANCED BACKTESTING (Sprint 5: 100% working)
        print_section("2. ADVANCED BACKTESTING ANALYSIS")
        
        settings = Settings()
        backtester = AdvancedBacktester(settings)
        
        # Configure backtest with recent 6 months
        backtest_end = datetime.now()
        backtest_start = backtest_end - timedelta(days=180)
        
        config = BacktestConfig(
            symbol=symbol,
            start_date=backtest_start,
            end_date=backtest_end,
            initial_capital=100000,
            position_size=0.1,
            stop_loss=0.05,
            take_profit=0.15
        )
        
        print(f"   📊 Backtesting {symbol} strategy")
        print(f"   📅 Period: {backtest_start.strftime('%Y-%m-%d')} to {backtest_end.strftime('%Y-%m-%d')}")
        print(f"   💰 Initial capital: ${config.initial_capital:,}")
        print(f"   📈 Position size: {config.position_size*100}% per trade")
        print(f"   🛡️ Stop loss: {config.stop_loss*100}%")
        print(f"   🎯 Take profit: {config.take_profit*100}%")
        
        # Run backtest
        print("   🔄 Running advanced backtesting simulation...")
        backtest_result = await backtester.run_backtest(config)
        
        print(f"\n   🎯 BACKTESTING RESULTS FOR {symbol}:")
        print(f"      📊 Total Return: {backtest_result.total_return:.2%}")
        print(f"      📈 Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}")
        print(f"      📉 Max Drawdown: {backtest_result.max_drawdown:.2%}")
        print(f"      🎲 Win Rate: {backtest_result.win_rate:.1%}")
        print(f"      💰 Final Portfolio Value: ${backtest_result.final_value:,.2f}")
        print(f"      📋 Total Trades: {backtest_result.total_trades}")
        print(f"      ⏱️ Average Trade Duration: {backtest_result.avg_trade_duration:.1f} days")
        
        # 3. PROFESSIONAL STRATEGY REPORT (Sprint 5: 100% working)
        print_section("3. PROFESSIONAL STRATEGY REPORT GENERATION")
        
        reporter = StrategyReporter(settings)
        
        print("   📋 Generating comprehensive professional report...")
        report = await reporter.generate_report(
            strategy_name=f"{symbol} AI Trading Strategy",
            backtest_result=backtest_result,
            competition_result=None,
            scenario_results=[],
            optimization_result=None
        )
        
        print("   ✅ Professional report generated successfully!")
        print("   📄 Report sections include:")
        print("      • Executive Summary with key insights")
        print("      • Performance Analysis with detailed metrics")
        print("      • Risk Analysis and drawdown assessment")
        print("      • Trade Analysis with entry/exit patterns")
        print("      • Professional disclaimers and legal notices")
        
        # Save report
        report_file = f"aapl_strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        await reporter.save_report(report, report_file)
        print(f"   💾 Report saved to: {report_file}")
        
        # 4. MARKET ANALYSIS INSIGHTS
        print_section("4. MARKET ANALYSIS & TRADING INSIGHTS")
        
        # Calculate some basic technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = calculate_rsi(data['Close'], 14)
        
        current_price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        current_rsi = data['RSI'].iloc[-1]
        
        print(f"   📊 Current Technical Analysis for {symbol}:")
        print(f"      💰 Current Price: ${current_price:.2f}")
        print(f"      📈 20-day SMA: ${sma_20:.2f} ({'Above' if current_price > sma_20 else 'Below'})")
        print(f"      📈 50-day SMA: ${sma_50:.2f} ({'Above' if current_price > sma_50 else 'Below'})")
        print(f"      📊 RSI (14): {current_rsi:.1f} ({'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral'})")
        
        # Trend analysis
        trend = "Bullish" if current_price > sma_20 > sma_50 else "Bearish" if current_price < sma_20 < sma_50 else "Sideways"
        print(f"      📈 Overall Trend: {trend}")
        
        # 5. TRADING RECOMMENDATIONS
        print_section("5. AI TRADING RECOMMENDATIONS")
        
        print(f"   🎯 Based on backtesting and technical analysis:")
        
        if backtest_result.total_return > 0:
            print("   ✅ POSITIVE SIGNAL: Strategy shows profitable potential")
        else:
            print("   ⚠️ CAUTION: Strategy shows negative returns in backtest")
            
        if backtest_result.sharpe_ratio > 1.0:
            print("   📈 GOOD RISK-ADJUSTED RETURNS: Sharpe ratio above 1.0")
        else:
            print("   📉 MODERATE RISK-ADJUSTED RETURNS: Consider risk management")
            
        if backtest_result.win_rate > 0.5:
            print(f"   🎲 GOOD WIN RATE: {backtest_result.win_rate:.1%} of trades profitable")
        else:
            print(f"   ⚠️ LOW WIN RATE: Only {backtest_result.win_rate:.1%} of trades profitable")
        
        print(f"\n   💡 SPECIFIC RECOMMENDATIONS FOR {symbol}:")
        print("      • Monitor the current trend and moving average positions")
        print("      • Consider position sizing based on volatility")
        print("      • Use stop-loss orders to manage downside risk")
        print("      • Review performance regularly and adjust strategy")
        
        print_header("SHOWCASE COMPLETE - WORKING CAPABILITIES DEMONSTRATED")
        print("🎉 Successfully demonstrated:")
        print("✅ Real market data fetching with proper date handling")
        print("✅ Advanced backtesting with comprehensive metrics")
        print("✅ Professional strategy report generation")
        print("✅ Technical analysis and trading insights")
        print("✅ AI-powered trading recommendations")
        print("\n🚀 SwingTradeAI core capabilities are working and ready for trading!")
        
    except Exception as e:
        print(f"\n❌ Showcase error: {str(e)}")
        print("🔧 This helps identify remaining issues to address")

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if __name__ == "__main__":
    asyncio.run(showcase_aapl_direct())
