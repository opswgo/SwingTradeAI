#!/usr/bin/env python3
"""
SwingTradeAI - Advanced Backtesting & Model Competition Interface
User Acceptance: User can run advanced backtesting, compare strategies, and get professional reports
"""

import streamlit as st
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.backtesting.advanced_backtester import AdvancedBacktester, BacktestConfig
from src.backtesting.model_competition import ModelCompetition, StrategyConfig
from src.backtesting.scenario_analyzer import ScenarioAnalyzer, MarketScenario
from src.backtesting.strategy_reporter import StrategyReporter
from src.core.config import Settings
from src.models.regime_model import RegimeAwareModel as RegimeModel
from src.features.market_regime import MarketRegimeDetector

st.set_page_config(
    page_title="SwingTradeAI - Advanced Backtesting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedBacktestingInterface:
    """User-facing interface for advanced backtesting and model competition"""
    
    def __init__(self):
        self.settings = Settings()
        self.backtester = AdvancedBacktester(self.settings)
        self.competition = ModelCompetition(self.settings)
        self.scenario_analyzer = ScenarioAnalyzer(self.settings)
        self.reporter = StrategyReporter(self.settings)
        
    def run(self):
        """Main interface"""
        
        st.title("🚀 SwingTradeAI - Advanced Backtesting & Model Competition")
        st.markdown("**Professional-grade strategy analysis with regime detection and model comparison**")
        
        # Sidebar controls
        with st.sidebar:
            st.header("🎯 Analysis Configuration")
            
            # Symbol selection
            symbol = st.text_input("Symbol", value="AAPL", help="Stock symbol to analyze")
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
            with col2:
                end_date = st.date_input("End Date", value=datetime.now())
            
            # Analysis type
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Single Strategy Backtest", "Strategy Competition", "Scenario Analysis", "Portfolio Optimization"],
                help="Choose the type of analysis to perform"
            )
            
            # Strategy parameters
            st.subheader("📈 Strategy Parameters")
            
            lookback_period = st.slider("Lookback Period (days)", 5, 50, 20)
            regime_threshold = st.slider("Regime Confidence Threshold", 0.5, 0.9, 0.7)
            position_size = st.slider("Position Size (%)", 10, 100, 25)
            stop_loss = st.slider("Stop Loss (%)", 1, 20, 5)
            
            # Run analysis button
            run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        # Main content area
        if run_analysis:
            with st.spinner("🔄 Running advanced analysis..."):
                if analysis_type == "Single Strategy Backtest":
                    self._run_single_backtest(symbol, start_date, end_date, lookback_period, regime_threshold, position_size, stop_loss)
                elif analysis_type == "Strategy Competition":
                    self._run_strategy_competition(symbol, start_date, end_date)
                elif analysis_type == "Scenario Analysis":
                    self._run_scenario_analysis(symbol, start_date, end_date)
                elif analysis_type == "Portfolio Optimization":
                    self._run_portfolio_optimization(symbol, start_date, end_date)
        else:
            self._show_welcome_screen()
    
    def _show_welcome_screen(self):
        """Show welcome screen with feature overview"""
        
        st.markdown("## 🎯 Advanced Backtesting Features")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Regime Detection", "Real-time", help="Market regime classification with confidence scores")
        
        with col2:
            st.metric("🏆 Strategy Competition", "Multi-model", help="Compare multiple strategies side-by-side")
        
        with col3:
            st.metric("🎲 Monte Carlo", "1000+ sims", help="Statistical validation with Monte Carlo simulation")
        
        with col4:
            st.metric("📈 Professional Reports", "Export ready", help="Institutional-quality strategy reports")
        
        st.markdown("---")
        
        # Feature descriptions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 **Single Strategy Backtest**
            - Regime-specific performance analysis
            - Real-time P&L tracking
            - Monte Carlo validation
            - Professional risk metrics
            - Trade-by-trade analysis
            """)
            
            st.markdown("""
            ### 🎲 **Scenario Analysis**
            - Market crash scenarios
            - Volatility spike testing
            - Bull/bear market analysis
            - Stress testing
            - Risk assessment
            """)
        
        with col2:
            st.markdown("""
            ### 🏆 **Strategy Competition**
            - Multi-strategy comparison
            - Statistical significance testing
            - Hyperparameter optimization
            - Ensemble model creation
            - Performance rankings
            """)
            
            st.markdown("""
            ### 📊 **Portfolio Optimization**
            - Mean-variance optimization
            - Risk parity allocation
            - Diversification analysis
            - Constraint optimization
            - Efficient frontier
            """)
        
        st.markdown("---")
        st.info("👆 **Configure your analysis in the sidebar and click 'Run Analysis' to get started!**")
    
    def _run_single_backtest(self, symbol, start_date, end_date, lookback_period, regime_threshold, position_size, stop_loss):
        """Run single strategy backtest"""
        
        try:
            # Create backtest configuration
            config = BacktestConfig(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000,
                position_size_pct=position_size / 100,
                stop_loss_pct=stop_loss / 100,
                regime_threshold=regime_threshold,
                lookback_period=lookback_period
            )
            
            # Run backtest
            result = asyncio.run(self.backtester.run_backtest(config))
            
            # Display results
            st.success(f"✅ Backtest completed for {symbol}")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Return", f"{result.total_return:.2f}%", 
                         delta=f"{result.total_return - result.benchmark_comparison.get('benchmark_return', 0):.2f}% vs benchmark")
            
            with col2:
                st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}",
                         delta="Excellent" if result.sharpe_ratio > 2 else "Good" if result.sharpe_ratio > 1 else "Moderate")
            
            with col3:
                st.metric("Max Drawdown", f"{result.max_drawdown:.2f}%",
                         delta="Low Risk" if result.max_drawdown < 10 else "Moderate Risk" if result.max_drawdown < 20 else "High Risk")
            
            with col4:
                st.metric("Win Rate", f"{result.win_rate:.1f}%",
                         delta=f"{result.total_trades} trades")
            
            # Performance chart
            st.subheader("📈 Performance Analysis")
            
            if not result.equity_curve.empty:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve['portfolio_value'],
                    mode='lines',
                    name='Strategy',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Add benchmark if available
                if 'benchmark_value' in result.equity_curve.columns:
                    fig.add_trace(go.Scatter(
                        x=result.equity_curve.index,
                        y=result.equity_curve['benchmark_value'],
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='#ff7f0e', width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Regime performance
            if result.regime_performance:
                st.subheader("🎯 Regime-Specific Performance")
                
                regime_data = []
                for regime_name, regime_perf in result.regime_performance.items():
                    regime_data.append({
                        'Market Regime': regime_name.replace('_', ' ').title(),
                        'Trades': regime_perf.total_trades,
                        'Win Rate (%)': f"{regime_perf.win_rate:.1f}%",
                        'Avg Return (%)': f"{regime_perf.avg_return:.2f}%",
                        'Sharpe Ratio': f"{regime_perf.sharpe_ratio:.2f}",
                        'Max Drawdown (%)': f"{regime_perf.max_drawdown:.2f}%"
                    })
                
                regime_df = pd.DataFrame(regime_data)
                st.dataframe(regime_df, use_container_width=True)
            
            # Trade analysis
            if not result.trade_history.empty:
                st.subheader("📋 Trade Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # P&L distribution
                    if 'pnl' in result.trade_history.columns:
                        fig = px.histogram(
                            result.trade_history,
                            x='pnl',
                            title="P&L Distribution",
                            nbins=20
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Holding period analysis
                    if 'holding_days' in result.trade_history.columns:
                        fig = px.box(
                            result.trade_history,
                            y='holding_days',
                            title="Holding Period Distribution"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Trade history table
                st.subheader("📊 Recent Trades")
                display_trades = result.trade_history.tail(10).copy()
                if not display_trades.empty:
                    # Format for display
                    for col in ['entry_date', 'exit_date']:
                        if col in display_trades.columns:
                            display_trades[col] = pd.to_datetime(display_trades[col]).dt.strftime('%Y-%m-%d')
                    
                    st.dataframe(display_trades, use_container_width=True)
            
            # Generate professional report
            st.subheader("📄 Professional Report")
            
            if st.button("📋 Generate Comprehensive Report", type="secondary"):
                with st.spinner("📊 Generating professional report..."):
                    report = asyncio.run(self.reporter.generate_comprehensive_report(result))
                    
                    st.success(f"✅ Report generated: {report.report_id}")
                    
                    # Display executive summary
                    st.markdown(report.executive_summary)
                    
                    # Download link
                    st.download_button(
                        label="📥 Download Full Report (JSON)",
                        data=json.dumps({
                            'report_id': report.report_id,
                            'executive_summary': report.executive_summary,
                            'key_metrics': {
                                'total_return': result.total_return,
                                'sharpe_ratio': result.sharpe_ratio,
                                'max_drawdown': result.max_drawdown,
                                'win_rate': result.win_rate
                            }
                        }, indent=2),
                        file_name=f"{report.report_id}.json",
                        mime="application/json"
                    )
            
        except Exception as e:
            st.error(f"❌ Backtest failed: {str(e)}")
            st.exception(e)
    
    def _run_strategy_competition(self, symbol, start_date, end_date):
        """Run strategy competition"""
        
        try:
            st.subheader("🏆 Strategy Competition")
            
            # Define strategies to compete
            strategies = [
                StrategyConfig(
                    name="Momentum Strategy",
                    description="Trend-following momentum strategy",
                    parameters={"lookback": 20, "threshold": 0.02},
                    regime_weights={"bull": 1.0, "bear": 0.3, "sideways": 0.5},
                    risk_settings={"stop_loss": 0.05, "position_size": 0.1},
                    entry_rules=["momentum > threshold"],
                    exit_rules=["stop_loss or take_profit"]
                ),
                StrategyConfig(
                    name="Mean Reversion Strategy",
                    description="Counter-trend mean reversion strategy",
                    parameters={"lookback": 14, "oversold": 30, "overbought": 70},
                    regime_weights={"bull": 0.5, "bear": 1.0, "sideways": 0.8},
                    risk_settings={"stop_loss": 0.03, "position_size": 0.08},
                    entry_rules=["rsi < oversold or rsi > overbought"],
                    exit_rules=["rsi normalized or stop_loss"]
                ),
                StrategyConfig(
                    name="ML Regime Strategy",
                    description="AI-powered regime-aware strategy",
                    parameters={"confidence_threshold": 0.7, "regime_lookback": 30},
                    regime_weights={"bull": 0.9, "bear": 0.9, "sideways": 0.6},
                    risk_settings={"stop_loss": 0.04, "position_size": 0.12},
                    entry_rules=["ml_confidence > threshold"],
                    exit_rules=["regime_change or stop_loss"]
                )
            ]
            
            # Run competition
            competition_result = asyncio.run(
                self.competition.run_competition(symbol, start_date, end_date, strategies)
            )
            
            st.success(f"✅ Competition completed with {len(strategies)} strategies")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🥇 Best Strategy", competition_result.best_strategy)
            
            with col2:
                st.metric("📊 Strategies Tested", len(strategies))
            
            with col3:
                best_sharpe = competition_result.performance_matrix['Sharpe Ratio'].max()
                st.metric("🎯 Best Sharpe Ratio", f"{best_sharpe:.2f}")
            
            # Performance matrix
            st.subheader("📊 Performance Comparison")
            st.dataframe(competition_result.performance_matrix, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            for rec in competition_result.recommendations:
                st.info(f"• {rec}")
            
        except Exception as e:
            st.error(f"❌ Strategy competition failed: {str(e)}")
            st.exception(e)
    
    def _run_scenario_analysis(self, symbol, start_date, end_date):
        """Run scenario analysis"""
        
        try:
            st.subheader("🎲 Scenario Analysis")
            
            # Define scenarios
            scenarios = [
                MarketScenario(
                    name="Market Crash",
                    description="20% market decline over 30 days",
                    market_shock=-0.20,
                    volatility_multiplier=2.0,
                    duration_days=30
                ),
                MarketScenario(
                    name="Volatility Spike",
                    description="VIX increases to 40+",
                    market_shock=0.0,
                    volatility_multiplier=3.0,
                    duration_days=14
                ),
                MarketScenario(
                    name="Bull Market",
                    description="15% market rally over 60 days",
                    market_shock=0.15,
                    volatility_multiplier=0.8,
                    duration_days=60
                )
            ]
            
            # Run scenario analysis
            scenario_results = {}
            for scenario in scenarios:
                result = asyncio.run(
                    self.scenario_analyzer.analyze_scenario(symbol, start_date, end_date, scenario)
                )
                scenario_results[scenario.name] = result
            
            st.success(f"✅ Analyzed {len(scenarios)} market scenarios")
            
            # Display scenario impacts
            scenario_data = []
            for scenario_name, scenario_result in scenario_results.items():
                scenario_data.append({
                    'Scenario': scenario_result.scenario_name,
                    'Return Impact (%)': f"{scenario_result.impact_metrics.get('return_change_pct', 0):.2f}%",
                    'Volatility Impact (%)': f"{scenario_result.impact_metrics.get('volatility_change_pct', 0):.2f}%",
                    'Max Drawdown Impact (%)': f"{scenario_result.impact_metrics.get('drawdown_change_pct', 0):.2f}%",
                    'Overall Assessment': scenario_result.impact_metrics.get('overall_impact', 'Moderate')
                })
            
            scenario_df = pd.DataFrame(scenario_data)
            st.dataframe(scenario_df, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Risk Management Recommendations")
            for scenario_name, scenario_result in scenario_results.items():
                for rec in scenario_result.recommendations:
                    st.info(f"**{scenario_name}**: {rec}")
            
        except Exception as e:
            st.error(f"❌ Scenario analysis failed: {str(e)}")
            st.exception(e)
    
    def _run_portfolio_optimization(self, symbol, start_date, end_date):
        """Run portfolio optimization"""
        
        try:
            st.subheader("📊 Portfolio Optimization")
            
            # For demo, use a few symbols
            symbols = [symbol, "SPY", "QQQ", "IWM"]
            
            # Run optimization
            optimization_result = asyncio.run(
                self.scenario_analyzer.optimize_portfolio(symbols, start_date, end_date)
            )
            
            st.success("✅ Portfolio optimization completed")
            
            # Display optimal allocation
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expected Return", f"{optimization_result.expected_return:.2f}%")
            
            with col2:
                st.metric("Expected Volatility", f"{optimization_result.expected_volatility:.2f}%")
            
            with col3:
                st.metric("Sharpe Ratio", f"{optimization_result.sharpe_ratio:.2f}")
            
            with col4:
                st.metric("Diversification Ratio", f"{optimization_result.diversification_ratio:.2f}")
            
            # Allocation chart
            allocation_data = []
            for symbol_name, weight in optimization_result.optimal_weights.items():
                allocation_data.append({
                    'Symbol': symbol_name,
                    'Weight': weight * 100
                })
            
            allocation_df = pd.DataFrame(allocation_data)
            
            fig = px.pie(
                allocation_df,
                values='Weight',
                names='Symbol',
                title="Optimal Portfolio Allocation"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Allocation table
            st.subheader("📋 Detailed Allocation")
            allocation_df['Weight (%)'] = allocation_df['Weight'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(allocation_df[['Symbol', 'Weight (%)']], use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Portfolio optimization failed: {str(e)}")
            st.exception(e)

def main():
    """Main application entry point"""
    
    try:
        interface = AdvancedBacktestingInterface()
        interface.run()
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
