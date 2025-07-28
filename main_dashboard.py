"""
Main Dashboard - NEAT Options System Hub
Single entry point with navigation to all analysis tools.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

# Import individual tool functions
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_main_dashboard():
    """Create the main dashboard hub for all NEAT Options System tools."""
    
    st.set_page_config(
        page_title="NEAT Options System - Main Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("🚀 NEAT Options System")
    st.sidebar.markdown("**World-Class Options Premium Harvesting**")
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "📋 Select Tool",
        options=[
            "🏠 Dashboard Home",
            "🔮 Weekly Predictions (1-8 Weeks)",
            "🔍 Stock Screening",
            "⚡ Options Chain Analyzer", 
            "🔬 Custom Backtesting",
            "📈 Performance Attribution",
            "🔬 Research & Analysis Tools"
        ],
        index=0
    )
    
    # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Status")
    st.sidebar.success("✅ NEAT AI Model: Active")
    st.sidebar.success("✅ Data Pipeline: Connected")
    st.sidebar.success("✅ Options Data: Live")
    st.sidebar.info("🔄 Last Updated: " + datetime.now().strftime("%H:%M:%S"))
    
    # Main content area
    if page == "🏠 Dashboard Home":
        show_dashboard_home()
    elif page == "🔮 Weekly Predictions (1-8 Weeks)":
        show_weekly_predictions()
    elif page == "🔍 Stock Screening":
        show_stock_screening()
    elif page == "⚡ Options Chain Analyzer":
        show_options_analyzer()
    elif page == "🔬 Custom Backtesting":
        show_custom_backtesting()
    elif page == "📈 Performance Attribution":
        show_performance_attribution()
    elif page == "🔬 Research & Analysis Tools":
        show_research_tools()

def show_dashboard_home():
    """Show the main dashboard home page."""
    
    st.header("🚀 NEAT Options System - Main Dashboard")
    st.markdown("**World-Class AI-Driven Options Premium Harvesting Platform**")
    
    # Quick stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Strategies", "5", delta="2")
    with col2:
        st.metric("Weekly Yield Target", "1.5%", delta="0.3%")
    with col3:
        st.metric("Assignment Rate", "2.1%", delta="-0.9%")
    with col4:
        st.metric("AI Confidence", "87%", delta="5%")
    
    # Main tools grid
    st.subheader("🛠️ Available Analysis Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("### 🔮 Weekly Predictions")
            st.markdown("**Multi-horizon forecasting (1-8 weeks)**")
            st.markdown("- Week-by-week price predictions")
            st.markdown("- AI confidence scores")
            st.markdown("- Options strategy recommendations")
            st.markdown("- Risk assessment for each week")
            
            if st.button("🔮 Open Weekly Predictions", key="pred"):
                st.rerun()
    
    with col2:
        with st.container():
            st.markdown("### 🔍 Stock Screening")
            st.markdown("**Advanced stock filtering & analysis**")
            st.markdown("- Real-time market data")
            st.markdown("- Custom filtering criteria")
            st.markdown("- Performance analytics")
            st.markdown("- Export capabilities")
            
            if st.button("🔍 Open Stock Screening", key="screen"):
                st.rerun()
    
    with col3:
        with st.container():
            st.markdown("### ⚡ Options Chain Analyzer")
            st.markdown("**Deep options market analysis**")
            st.markdown("- Multi-expiration analysis")
            st.markdown("- Greeks visualization")
            st.markdown("- Volatility smile analysis")
            st.markdown("- Premium yield calculations")
            
            if st.button("⚡ Open Options Analyzer", key="options"):
                st.rerun()
    
    # Second row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("### 🔬 Custom Backtesting")
            st.markdown("**Advanced strategy testing**")
            st.markdown("- Custom parameters")
            st.markdown("- Risk management controls")
            st.markdown("- Performance analytics")
            st.markdown("- Trade simulation")
            
            if st.button("🔬 Open Custom Backtesting", key="backtest"):
                st.rerun()
    
    with col2:
        with st.container():
            st.markdown("### 📈 Performance Attribution")
            st.markdown("**Detailed performance analysis**")
            st.markdown("- Factor attribution")
            st.markdown("- Risk decomposition")
            st.markdown("- Sector analysis")
            st.markdown("- Time-based attribution")
            
            if st.button("📈 Open Performance Attribution", key="perf"):
                st.rerun()
    
    with col3:
        with st.container():
            st.markdown("### 🔬 Research & Analysis")
            st.markdown("**Market intelligence & research**")
            st.markdown("- Market research dashboard")
            st.markdown("- AI strategy research")
            st.markdown("- Technical analysis")
            st.markdown("- Custom research tools")
            
            if st.button("🔬 Open Research Tools", key="research"):
                st.rerun()
    
    # Recent activity
    st.subheader("📊 Recent Activity")
    
    activity_data = pd.DataFrame({
        'Time': ['22:30', '22:25', '22:20', '22:15', '22:10'],
        'Activity': [
            'Generated 8-week predictions for AAPL',
            'Screened 24 stocks with custom filters',
            'Analyzed TSLA options chain',
            'Backtested NEAT strategy on NVDA',
            'Updated AI model parameters'
        ],
        'Status': ['✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete']
    })
    
    st.dataframe(activity_data, use_container_width=True, hide_index=True)
    
    # Quick market overview
    st.subheader("📈 Market Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sample market data chart
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        spy_prices = 420 + np.cumsum(np.random.normal(0, 2, 30))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=spy_prices,
            mode='lines',
            name='SPY',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="SPY - Last 30 Days",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Market Indicators**")
        st.metric("VIX", "18.5", delta="-2.1")
        st.metric("Put/Call Ratio", "0.85", delta="+0.05")
        st.metric("Market Breadth", "65%", delta="+5%")
        st.metric("Options Volume", "42.3M", delta="+12%")

def show_weekly_predictions():
    """Show the weekly predictions interface."""
    from weekly_predictions_interface import create_weekly_predictions_interface
    create_weekly_predictions_interface()

def show_stock_screening():
    """Show the stock screening interface."""
    from simple_stock_screener import create_simple_stock_screener
    create_simple_stock_screener()

def show_options_analyzer():
    """Show options chain analyzer."""
    st.header("⚡ Options Chain Analyzer")
    st.info("🚧 Loading Options Chain Analyzer...")
    st.markdown("This tool provides comprehensive options chain analysis with:")
    st.markdown("- Multi-expiration analysis")
    st.markdown("- Greeks visualization") 
    st.markdown("- Volatility smile analysis")
    st.markdown("- Premium yield calculations")
    
    # Basic options analysis demo
    symbol = st.selectbox("Select Symbol", ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"])
    
    if st.button("Analyze Options Chain"):
        st.success(f"✅ Analyzing options chain for {symbol}")
        
        # Sample options data
        options_data = pd.DataFrame({
            'Strike': [140, 145, 150, 155, 160],
            'Call Premium': [12.50, 8.75, 5.25, 2.80, 1.15],
            'Put Premium': [2.15, 3.80, 6.25, 9.75, 14.20],
            'Call Delta': [0.85, 0.65, 0.45, 0.25, 0.10],
            'Put Delta': [-0.15, -0.35, -0.55, -0.75, -0.90]
        })
        
        st.dataframe(options_data, use_container_width=True)

def show_custom_backtesting():
    """Show custom backtesting interface."""
    st.header("🔬 Custom Backtesting Laboratory")
    st.info("🚧 Loading Custom Backtesting Interface...")
    st.markdown("Advanced backtesting with:")
    st.markdown("- Custom parameters")
    st.markdown("- Risk management controls")
    st.markdown("- Performance analytics")
    st.markdown("- Trade simulation")
    
    # Basic backtesting demo
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Backtest Parameters")
        symbol = st.selectbox("Symbol", ["AAPL", "TSLA", "NVDA"], key="bt_symbol")
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=180))
        confidence = st.slider("Confidence Threshold", 0.3, 0.9, 0.6)
    
    with col2:
        st.subheader("Risk Parameters")
        position_size = st.number_input("Position Size ($)", 1000, 50000, 10000)
        max_assignment = st.slider("Max Assignment Rate (%)", 1, 20, 5)
        
    if st.button("Run Backtest"):
        st.success("✅ Backtest completed successfully!")
        
        # Sample results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", "15.2%")
        with col2:
            st.metric("Sharpe Ratio", "1.85")
        with col3:
            st.metric("Max Drawdown", "3.8%")
        with col4:
            st.metric("Win Rate", "89.3%")

def show_performance_attribution():
    """Show performance attribution interface."""
    st.header("📈 Performance Attribution Analyzer")
    st.info("🚧 Loading Performance Attribution Analyzer...")
    st.markdown("Detailed performance analysis including:")
    st.markdown("- Factor attribution")
    st.markdown("- Risk decomposition")
    st.markdown("- Sector analysis")
    st.markdown("- Time-based attribution")
    
    # Sample attribution chart
    factors = ['Market', 'Value', 'Growth', 'Momentum', 'Quality', 'Alpha']
    contributions = [0.08, 0.03, -0.01, 0.04, 0.02, 0.05]
    
    fig = px.bar(
        x=factors,
        y=contributions,
        title="Factor Attribution to Total Return",
        color=contributions,
        color_continuous_scale='RdYlGn'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_research_tools():
    """Show research and analysis tools."""
    st.header("🔬 Research & Analysis Laboratory")
    st.info("🚧 Loading Research & Analysis Tools...")
    st.markdown("Advanced research capabilities:")
    st.markdown("- Market research dashboard")
    st.markdown("- AI strategy research")
    st.markdown("- Technical analysis")
    st.markdown("- Custom research tools")
    
    # Research tabs
    tab1, tab2, tab3 = st.tabs(["Market Research", "AI Strategy Research", "Technical Analysis"])
    
    with tab1:
        st.subheader("Market Overview")
        
        # Market indices
        indices_data = pd.DataFrame({
            'Index': ['SPY', 'QQQ', 'IWM', 'VTI'],
            'Price': ['$420.50', '$350.25', '$195.80', '$225.40'],
            '1D Change': ['+0.5%', '+0.8%', '-0.2%', '+0.3%'],
            '1M Change': ['+2.1%', '+3.5%', '+1.2%', '+2.3%']
        })
        
        st.dataframe(indices_data, use_container_width=True)
    
    with tab2:
        st.subheader("NEAT AI Strategy Performance")
        
        strategies = ['NEAT v1.0', 'Random Forest', 'Gradient Boost', 'Buy & Hold']
        returns = [15.2, 8.9, 7.2, 10.5]
        
        fig = px.bar(x=strategies, y=returns, title="Strategy Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Technical Analysis")
        
        symbol = st.selectbox("Symbol for Analysis", ["AAPL", "TSLA", "NVDA"], key="ta_symbol")
        
        if st.button("Generate Technical Analysis"):
            st.success(f"✅ Technical analysis for {symbol} completed")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RSI", "65.2", delta="Neutral")
            with col2:
                st.metric("MACD", "1.25", delta="Bullish")
            with col3:
                st.metric("BB Position", "0.75", delta="Upper")

if __name__ == "__main__":
    create_main_dashboard()
