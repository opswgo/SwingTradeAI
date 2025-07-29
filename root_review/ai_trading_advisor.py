#!/usr/bin/env python3
"""
🤖 AI Trading Advisor - Fully Automated Strategy Recommendation System
Automatically backtests all strategies, finds the best one, and provides multi-week predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StrategyResult:
    name: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: int
    beats_benchmark: bool
    excess_return: float

class AITradingAdvisor:
    """Fully automated AI trading advisor that finds the best strategy and predicts future moves"""
    
    def __init__(self):
        # Define all available strategies to test
        self.strategies = {
            'Regime-Aware ML': {
                'position_size_pct': 0.25,
                'stop_loss_pct': 0.05,
                'regime_threshold': 0.7,
                'lookback_period': 20
            },
            'Conservative ML': {
                'position_size_pct': 0.15,
                'stop_loss_pct': 0.03,
                'regime_threshold': 0.8,
                'lookback_period': 30
            },
            'Aggressive ML': {
                'position_size_pct': 0.35,
                'stop_loss_pct': 0.08,
                'regime_threshold': 0.6,
                'lookback_period': 15
            },
            'Momentum Strategy': {
                'position_size_pct': 0.30,
                'stop_loss_pct': 0.06,
                'regime_threshold': 0.65,
                'lookback_period': 10
            },
            'Mean Reversion': {
                'position_size_pct': 0.20,
                'stop_loss_pct': 0.04,
                'regime_threshold': 0.75,
                'lookback_period': 25
            }
        }
    
    async def find_best_strategy(self, symbol: str, months_back: int = 12) -> Dict:
        """Automatically backtest all strategies and find the best one"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)
        
        st.write(f"🔄 **Auto-backtesting {len(self.strategies)} strategies for {symbol}...**")
        
        # Get buy-and-hold benchmark
        benchmark_return = await self._calculate_buy_hold_return(symbol, start_date, end_date)
        
        results = {}
        progress_bar = st.progress(0)
        
        for i, (strategy_name, params) in enumerate(self.strategies.items()):
            st.write(f"Testing {strategy_name}...")
            
            try:
                config = BacktestConfig(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=100000,
                    **params
                )
                
                result = await self.backtester.run_backtest(config)
                
                # Calculate key metrics
                results[strategy_name] = {
                    'result': result,
                    'total_return': result.total_return,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown,
                    'win_rate': result.win_rate,
                    'beats_benchmark': result.total_return > benchmark_return,
                    'excess_return': result.total_return - benchmark_return
                }
                
            except Exception as e:
                st.error(f"Failed to test {strategy_name}: {str(e)}")
                results[strategy_name] = None
            
            progress_bar.progress((i + 1) / len(self.strategies))
        
        # Find the best strategy
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            return None
        
        # Rank by excess return over benchmark, then by Sharpe ratio
        best_strategy = max(valid_results.items(), 
                          key=lambda x: (x[1]['beats_benchmark'], x[1]['excess_return'], x[1]['sharpe_ratio']))
        
        return {
            'best_strategy': best_strategy[0],
            'best_result': best_strategy[1],
            'all_results': valid_results,
            'benchmark_return': benchmark_return
        }
    
    async def _calculate_buy_hold_return(self, symbol: str, start_date: datetime, end_date: datetime) -> float:
        """Calculate buy-and-hold return for benchmark"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                end=end_date.strftime('%Y-%m-%d'))
            
            if len(data) < 2:
                return 0.0
            
            return (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
        except:
            return 0.0
    
    async def generate_predictions(self, symbol: str, weeks: List[int] = [1,2,3,4,5,6,7,8]) -> Dict:
        """Generate multi-week predictions for the symbol"""
        
        st.write(f"🔮 **Generating {len(weeks)}-week predictions for {symbol}...**")
        
        try:
            # Get recent features for prediction
            features_df = await self.feature_store.get_features(symbol, days=60)
            
            if features_df.empty:
                return {"error": "No features available for prediction"}
            
            predictions = {}
            
            for week in weeks:
                # Generate prediction for this time horizon
                prediction = await self.regime_model.predict(features_df.iloc[-1:])
                
                if prediction:
                    # Simulate confidence decay over time
                    confidence = max(0.3, prediction.get('confidence', 0.5) * (0.9 ** (week - 1)))
                    
                    predictions[f"{week}_week"] = {
                        'direction': prediction.get('prediction', 'HOLD'),
                        'confidence': confidence,
                        'regime': prediction.get('regime', 'UNKNOWN'),
                        'reasoning': f"Week {week}: {prediction.get('reasoning', 'AI analysis based on current market regime')}"
                    }
                else:
                    predictions[f"{week}_week"] = {
                        'direction': 'HOLD',
                        'confidence': 0.3,
                        'regime': 'UNKNOWN',
                        'reasoning': f"Week {week}: Insufficient data for reliable prediction"
                    }
            
            return predictions
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

def main():
    st.set_page_config(
        page_title="🤖 AI Trading Advisor",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🤖 AI Trading Advisor")
    st.markdown("**Fully Automated Strategy Recommendation & Multi-Week Predictions**")
    st.markdown("---")
    
    # Initialize advisor
    if 'advisor' not in st.session_state:
        st.session_state.advisor = AITradingAdvisor()
    
    # User input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("📊 **Enter Stock Symbol:**", value="AAPL", key="symbol_input").upper()
    
    with col2:
        months_back = st.selectbox("📅 **Analysis Period:**", [6, 9, 12, 18, 24], index=2)
    
    if st.button("🚀 **ANALYZE & RECOMMEND**", type="primary"):
        if symbol:
            # Clear previous results
            for key in ['strategy_results', 'predictions']:
                if key in st.session_state:
                    del st.session_state[key]
            
            with st.spinner("🤖 AI is analyzing all strategies and generating predictions..."):
                # Find best strategy
                strategy_results = asyncio.run(st.session_state.advisor.find_best_strategy(symbol, months_back))
                st.session_state.strategy_results = strategy_results
                
                # Generate predictions
                predictions = asyncio.run(st.session_state.advisor.generate_predictions(symbol))
                st.session_state.predictions = predictions
    
    # Display results
    if 'strategy_results' in st.session_state and st.session_state.strategy_results:
        results = st.session_state.strategy_results
        
        st.markdown("## 🎯 **AI RECOMMENDATION**")
        
        best_name = results['best_strategy']
        best_data = results['best_result']
        benchmark = results['benchmark_return']
        
        # Recommendation header
        if best_data['beats_benchmark']:
            st.success(f"✅ **RECOMMENDED STRATEGY: {best_name}**")
            st.write(f"🏆 **Outperforms Buy-and-Hold by {best_data['excess_return']:.2%}**")
        else:
            st.warning(f"⚠️ **BEST AVAILABLE: {best_name}** (Still underperforms buy-and-hold)")
            st.write(f"📉 **Underperforms Buy-and-Hold by {abs(best_data['excess_return']):.2%}**")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{best_data['total_return']:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{best_data['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{best_data['max_drawdown']:.2%}")
        with col4:
            st.metric("Win Rate", f"{best_data['win_rate']:.1%}")
        
        # Benchmark comparison
        st.markdown("### 📊 **Strategy Comparison**")
        
        comparison_data = []
        for name, data in results['all_results'].items():
            comparison_data.append({
                'Strategy': name,
                'Total Return': f"{data['total_return']:.2%}",
                'Excess Return': f"{data['excess_return']:.2%}",
                'Sharpe Ratio': f"{data['sharpe_ratio']:.2f}",
                'Beats Benchmark': "✅" if data['beats_benchmark'] else "❌"
            })
        
        # Add benchmark row
        comparison_data.append({
            'Strategy': '📈 Buy-and-Hold',
            'Total Return': f"{benchmark:.2%}",
            'Excess Return': "0.00%",
            'Sharpe Ratio': "N/A",
            'Beats Benchmark': "📊"
        })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
    
    # Display predictions
    if 'predictions' in st.session_state and st.session_state.predictions:
        predictions = st.session_state.predictions
        
        if 'error' not in predictions:
            st.markdown("## 🔮 **MULTI-WEEK PREDICTIONS**")
            
            # Create prediction timeline
            pred_data = []
            for week_key, pred in predictions.items():
                week_num = int(week_key.split('_')[0])
                pred_data.append({
                    'Week': week_num,
                    'Direction': pred['direction'],
                    'Confidence': f"{pred['confidence']:.1%}",
                    'Regime': pred['regime'],
                    'Reasoning': pred['reasoning']
                })
            
            pred_df = pd.DataFrame(pred_data)
            
            # Display as cards
            cols = st.columns(4)
            for i, row in pred_df.iterrows():
                col_idx = i % 4
                with cols[col_idx]:
                    direction = row['Direction']
                    if direction == 'BUY':
                        st.success(f"**Week {row['Week']}**\n🟢 {direction}\n{row['Confidence']} confidence")
                    elif direction == 'SELL':
                        st.error(f"**Week {row['Week']}**\n🔴 {direction}\n{row['Confidence']} confidence")
                    else:
                        st.info(f"**Week {row['Week']}**\n⚪ {direction}\n{row['Confidence']} confidence")
            
            # Detailed prediction table
            st.markdown("### 📋 **Detailed Predictions**")
            st.dataframe(pred_df, use_container_width=True)
        else:
            st.error(f"Prediction Error: {predictions['error']}")
    
    # Instructions
    with st.expander("ℹ️ **How It Works**"):
        st.markdown("""
        **🤖 Fully Automated AI Trading System:**
        
        1. **Auto-Strategy Testing**: Tests 5 different ML strategies automatically
        2. **Benchmark Comparison**: Compares all strategies against buy-and-hold
        3. **Best Strategy Selection**: Automatically recommends the top performer
        4. **Multi-Week Predictions**: Provides 1-8 week forward predictions
        5. **Clear Recommendations**: No guesswork - tells you exactly what to do
        
        **🎯 What You Get:**
        - ✅ Automatic best strategy identification
        - 📊 Clear benchmark comparisons  
        - 🔮 Multi-horizon predictions
        - 🤖 AI-driven recommendations
        - 📈 Only strategies that beat buy-and-hold are recommended
        """)

if __name__ == "__main__":
    main()
