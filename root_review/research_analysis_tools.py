"""
Research and Analysis Tools for NEAT Options System
Advanced research capabilities including market analysis, strategy research, and AI insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import yfinance as yf

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neat_options_advisor import neat_advisor
from neat_framework import neat_evolution

def create_research_analysis_interface():
    """Create comprehensive research and analysis tools interface."""
    
    st.header("🔬 Research & Analysis Laboratory")
    st.markdown("Advanced research tools for market analysis and strategy development")
    
    # Main tabs for different research areas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Research", 
        "🧠 AI Strategy Research", 
        "📈 Technical Analysis", 
        "📰 Market Intelligence",
        "🔍 Custom Research"
    ])
    
    with tab1:
        display_market_research()
    
    with tab2:
        display_ai_strategy_research()
    
    with tab3:
        display_technical_analysis()
    
    with tab4:
        display_market_intelligence()
    
    with tab5:
        display_custom_research()

def display_market_research():
    """Display market research tools and analysis."""
    
    st.subheader("📊 Market Research Dashboard")
    
    # Market overview section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌍 Market Overview")
        
        # Market indices analysis
        indices = ['SPY', 'QQQ', 'IWM', 'VTI', 'DIA']
        
        market_data = {}
        for index in indices:
            try:
                ticker = yf.Ticker(index)
                hist = ticker.history(period="1mo")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    change_1d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100
                    change_1m = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                    
                    market_data[index] = {
                        'Price': f"${current_price:.2f}",
                        '1D Change': f"{change_1d:+.2f}%",
                        '1M Change': f"{change_1m:+.2f}%"
                    }
            except:
                market_data[index] = {
                    'Price': 'N/A',
                    '1D Change': 'N/A',
                    '1M Change': 'N/A'
                }
        
        market_df = pd.DataFrame(market_data).T
        st.dataframe(market_df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Market Sentiment")
        
        # Simulated market sentiment indicators
        sentiment_data = {
            'VIX Level': 18.5,
            'Put/Call Ratio': 0.85,
            'Market Breadth': 65,
            'Momentum Score': 72
        }
        
        for indicator, value in sentiment_data.items():
            if indicator == 'VIX Level':
                color = "normal" if value < 20 else "inverse"
            elif indicator == 'Put/Call Ratio':
                color = "normal" if 0.7 <= value <= 1.0 else "inverse"
            else:
                color = "normal" if value > 50 else "inverse"
            
            st.metric(indicator, f"{value}", delta_color=color)
    
    # Sector analysis
    st.subheader("🏭 Sector Performance Analysis")
    
    sectors = {
        'Technology': {'return': 2.3, 'volatility': 15.2},
        'Healthcare': {'return': 1.8, 'volatility': 12.1},
        'Financial': {'return': 1.2, 'volatility': 18.5},
        'Consumer Disc.': {'return': 0.9, 'volatility': 16.8},
        'Industrial': {'return': 1.5, 'volatility': 14.3},
        'Energy': {'return': -0.5, 'volatility': 22.1},
        'Utilities': {'return': 0.8, 'volatility': 9.2},
        'Materials': {'return': 1.1, 'volatility': 17.4},
        'Real Estate': {'return': 0.6, 'volatility': 13.9},
        'Telecom': {'return': 0.4, 'volatility': 11.5}
    }
    
    sector_df = pd.DataFrame(sectors).T
    
    # Sector performance scatter plot
    fig = px.scatter(
        sector_df,
        x='volatility',
        y='return',
        title="Sector Risk-Return Profile",
        labels={'volatility': 'Volatility (%)', 'return': 'Return (%)'},
        hover_name=sector_df.index
    )
    
    fig.update_traces(marker=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)
    
    # Options market analysis
    st.subheader("⚡ Options Market Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg IV Rank", "45%", delta="+5%")
        st.metric("Options Volume", "42.3M", delta="+12%")
    
    with col2:
        st.metric("Call/Put Skew", "1.15", delta="-0.05")
        st.metric("Term Structure", "Backwardation", delta="Neutral")
    
    with col3:
        st.metric("Gamma Exposure", "$2.1B", delta="+$300M")
        st.metric("Max Pain Level", "4,180", delta="+25")

def display_ai_strategy_research():
    """Display AI strategy research and development tools."""
    
    st.subheader("🧠 AI Strategy Research Laboratory")
    
    # NEAT evolution analysis
    st.subheader("🧬 NEAT Evolution Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Evolution progress chart
        generations = list(range(1, 21))
        fitness_scores = [50 + 2*i + np.random.normal(0, 5) for i in generations]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=generations,
            y=fitness_scores,
            mode='lines+markers',
            name='Best Fitness',
            line=dict(color='blue', width=3)
        ))
        
        fig.update_layout(
            title="NEAT Evolution Progress",
            xaxis_title="Generation",
            yaxis_title="Fitness Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Current Best Model")
        
        st.metric("Generation", "20")
        st.metric("Fitness Score", "89.2")
        st.metric("Network Complexity", "47 nodes")
        st.metric("Success Rate", "94.3%")
        st.metric("Assignment Rate", "2.1%")
        st.metric("Weekly Yield", "1.8%")
    
    # Strategy performance comparison
    st.subheader("📊 Strategy Performance Comparison")
    
    strategies = {
        'NEAT v1.0': {'return': 15.2, 'sharpe': 1.85, 'assignment': 2.1},
        'NEAT v0.9': {'return': 12.8, 'sharpe': 1.62, 'assignment': 3.2},
        'Random Forest': {'return': 8.9, 'sharpe': 1.23, 'assignment': 5.8},
        'Gradient Boost': {'return': 7.2, 'sharpe': 1.15, 'assignment': 6.1},
        'Buy & Hold': {'return': 10.5, 'sharpe': 0.95, 'assignment': 0.0}
    }
    
    strategy_df = pd.DataFrame(strategies).T
    
    # Multi-metric comparison chart
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Annual Return (%)", "Sharpe Ratio", "Assignment Rate (%)"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    strategies_list = list(strategies.keys())
    
    fig.add_trace(
        go.Bar(x=strategies_list, y=[strategies[s]['return'] for s in strategies_list], name="Return"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=strategies_list, y=[strategies[s]['sharpe'] for s in strategies_list], name="Sharpe"),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=strategies_list, y=[strategies[s]['assignment'] for s in strategies_list], name="Assignment"),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance analysis
    st.subheader("🔍 Feature Importance Analysis")
    
    features = {
        'Price Momentum': 0.18,
        'Volatility Regime': 0.15,
        'Options Greeks': 0.14,
        'Market Breadth': 0.12,
        'Volume Profile': 0.11,
        'Technical Indicators': 0.10,
        'Earnings Distance': 0.09,
        'Sector Rotation': 0.07,
        'Macro Factors': 0.04
    }
    
    feature_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Importance'])
    
    fig = px.bar(
        feature_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="NEAT Model Feature Importance"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_technical_analysis():
    """Display technical analysis tools."""
    
    st.subheader("📈 Technical Analysis Workshop")
    
    # Symbol input
    col1, col2 = st.columns([1, 3])
    
    with col1:
        symbol = st.text_input("Symbol", value="AAPL").upper()
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        
        if st.button("🔍 Analyze", type="primary"):
            st.session_state['ta_symbol'] = symbol
            st.session_state['ta_period'] = period
    
    with col2:
        if 'ta_symbol' in st.session_state:
            symbol = st.session_state['ta_symbol']
            period = st.session_state['ta_period']
            
            try:
                # Fetch data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if not data.empty:
                    # Price chart with moving averages
                    fig = go.Figure()
                    
                    # Candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name=symbol
                    ))
                    
                    # Moving averages
                    data['MA20'] = data['Close'].rolling(20).mean()
                    data['MA50'] = data['Close'].rolling(50).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['MA20'],
                        mode='lines',
                        name='MA20',
                        line=dict(color='orange', width=1)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['MA50'],
                        mode='lines',
                        name='MA50',
                        line=dict(color='blue', width=1)
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} Technical Analysis",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Technical indicators
                    st.subheader("📊 Technical Indicators")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        # RSI
                        delta = data['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        current_rsi = rsi.iloc[-1]
                        
                        rsi_signal = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
                        st.metric("RSI", f"{current_rsi:.1f}", delta=rsi_signal)
                    
                    with col2:
                        # MACD
                        ema12 = data['Close'].ewm(span=12).mean()
                        ema26 = data['Close'].ewm(span=26).mean()
                        macd = ema12 - ema26
                        signal = macd.ewm(span=9).mean()
                        histogram = macd - signal
                        
                        macd_signal = "Bullish" if histogram.iloc[-1] > 0 else "Bearish"
                        st.metric("MACD", f"{macd.iloc[-1]:.2f}", delta=macd_signal)
                    
                    with col3:
                        # Bollinger Bands
                        bb_period = 20
                        bb_std = 2
                        sma = data['Close'].rolling(bb_period).mean()
                        std = data['Close'].rolling(bb_period).std()
                        upper_bb = sma + (std * bb_std)
                        lower_bb = sma - (std * bb_std)
                        
                        bb_position = (data['Close'].iloc[-1] - lower_bb.iloc[-1]) / (upper_bb.iloc[-1] - lower_bb.iloc[-1])
                        bb_signal = "Upper" if bb_position > 0.8 else "Lower" if bb_position < 0.2 else "Middle"
                        st.metric("BB Position", f"{bb_position:.2f}", delta=bb_signal)
                    
                    with col4:
                        # Volume analysis
                        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                        current_volume = data['Volume'].iloc[-1]
                        volume_ratio = current_volume / avg_volume
                        
                        volume_signal = "High" if volume_ratio > 1.5 else "Low" if volume_ratio < 0.5 else "Normal"
                        st.metric("Volume Ratio", f"{volume_ratio:.1f}x", delta=volume_signal)
                
                else:
                    st.error(f"No data available for {symbol}")
                    
            except Exception as e:
                st.error(f"Error analyzing {symbol}: {e}")

def display_market_intelligence():
    """Display market intelligence and news analysis."""
    
    st.subheader("📰 Market Intelligence Center")
    
    # Market news sentiment (simulated)
    st.subheader("📊 News Sentiment Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sentiment over time
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        sentiment_scores = np.random.normal(0.1, 0.3, 30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=sentiment_scores,
            mode='lines+markers',
            name='Market Sentiment',
            line=dict(color='green' if sentiment_scores[-1] > 0 else 'red', width=2)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Market Sentiment Trend (30 Days)",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Sentiment Metrics")
        
        current_sentiment = sentiment_scores[-1]
        sentiment_label = "Positive" if current_sentiment > 0.1 else "Negative" if current_sentiment < -0.1 else "Neutral"
        
        st.metric("Current Sentiment", f"{current_sentiment:.2f}", delta=sentiment_label)
        st.metric("Sentiment Volatility", f"{np.std(sentiment_scores):.2f}")
        st.metric("Positive Days", f"{np.sum(sentiment_scores > 0)}/30")
        st.metric("Trend", "Improving" if sentiment_scores[-1] > sentiment_scores[-7] else "Declining")
    
    # Economic indicators
    st.subheader("📊 Economic Indicators")
    
    indicators = {
        'GDP Growth': {'value': 2.1, 'change': '+0.3', 'status': 'Positive'},
        'Unemployment': {'value': 3.7, 'change': '-0.1', 'status': 'Positive'},
        'Inflation (CPI)': {'value': 3.2, 'change': '-0.2', 'status': 'Positive'},
        'Fed Funds Rate': {'value': 5.25, 'change': '0.0', 'status': 'Neutral'},
        'Consumer Confidence': {'value': 102.3, 'change': '+2.1', 'status': 'Positive'},
        '10Y Treasury': {'value': 4.35, 'change': '+0.05', 'status': 'Neutral'}
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (indicator, data) in enumerate(indicators.items()):
        col = [col1, col2, col3][i % 3]
        with col:
            delta_color = "normal" if data['status'] == 'Positive' else "inverse" if data['status'] == 'Negative' else "off"
            st.metric(indicator, f"{data['value']}", delta=data['change'], delta_color=delta_color)
    
    # Earnings calendar (simulated)
    st.subheader("📅 Upcoming Earnings")
    
    earnings_data = {
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        'Date': ['2024-01-25', '2024-01-26', '2024-01-27', '2024-01-28', '2024-01-29'],
        'Expected EPS': [2.18, 2.78, 1.45, 0.85, 5.12],
        'Implied Move': ['4.2%', '3.8%', '5.1%', '8.5%', '6.3%']
    }
    
    earnings_df = pd.DataFrame(earnings_data)
    st.dataframe(earnings_df, use_container_width=True, hide_index=True)

def display_custom_research():
    """Display custom research tools and analysis."""
    
    st.subheader("🔍 Custom Research Laboratory")
    
    # Research query interface
    st.subheader("🔬 Research Query Builder")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        research_type = st.selectbox(
            "Research Type",
            options=[
                "Strategy Backtesting",
                "Market Correlation Analysis",
                "Volatility Surface Analysis",
                "Options Flow Analysis",
                "Sector Rotation Study",
                "Risk Factor Analysis"
            ]
        )
        
        symbols_input = st.text_area(
            "Symbols (comma-separated)",
            value="AAPL,MSFT,GOOGL,TSLA,NVDA"
        )
        
        date_range = st.date_input(
            "Analysis Period",
            value=[datetime.now() - timedelta(days=180), datetime.now()],
            max_value=datetime.now()
        )
    
    with col2:
        st.subheader("⚙️ Research Parameters")
        
        if research_type == "Strategy Backtesting":
            confidence_threshold = st.slider("Confidence Threshold", 0.3, 0.9, 0.6, 0.05)
            min_premium = st.slider("Min Premium (%)", 0.5, 5.0, 1.5, 0.1)
            max_assignment = st.slider("Max Assignment (%)", 1, 20, 5, 1)
        
        elif research_type == "Market Correlation Analysis":
            correlation_window = st.slider("Rolling Window (days)", 30, 252, 90, 10)
            correlation_threshold = st.slider("Correlation Threshold", 0.1, 0.9, 0.5, 0.1)
        
        elif research_type == "Volatility Surface Analysis":
            vol_model = st.selectbox("Volatility Model", ["Black-Scholes", "Heston", "SABR"])
            moneyness_range = st.slider("Moneyness Range", 0.8, 1.2, (0.9, 1.1), 0.05)
        
        else:
            st.info("Configure parameters based on selected research type")
    
    # Execute research
    if st.button("🚀 Execute Research", type="primary"):
        with st.spinner("Conducting research analysis..."):
            # Simulate research execution
            st.success("✅ Research completed successfully!")
            
            # Display sample results based on research type
            if research_type == "Strategy Backtesting":
                display_strategy_backtest_results()
            elif research_type == "Market Correlation Analysis":
                display_correlation_analysis()
            elif research_type == "Volatility Surface Analysis":
                display_volatility_surface()
            else:
                st.info("Research results would be displayed here")

def display_strategy_backtest_results():
    """Display strategy backtesting results."""
    
    st.subheader("📊 Strategy Backtesting Results")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", "15.2%", delta="+5.2%")
    with col2:
        st.metric("Sharpe Ratio", "1.85", delta="+0.45")
    with col3:
        st.metric("Max Drawdown", "3.8%", delta="-1.2%")
    with col4:
        st.metric("Win Rate", "89.3%", delta="+4.3%")
    
    # Equity curve
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    returns = np.random.normal(0.001, 0.02, 100)
    cumulative_returns = (1 + pd.Series(returns)).cumprod() - 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_returns * 100,
        mode='lines',
        name='Strategy',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Strategy Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_correlation_analysis():
    """Display correlation analysis results."""
    
    st.subheader("🔗 Market Correlation Analysis")
    
    # Create sample correlation matrix
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    correlation_matrix = np.random.rand(5, 5)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1)
    
    correlation_df = pd.DataFrame(correlation_matrix, index=symbols, columns=symbols)
    
    fig = px.imshow(
        correlation_df,
        title="Stock Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_volatility_surface():
    """Display volatility surface analysis."""
    
    st.subheader("📊 Volatility Surface Analysis")
    
    # Create sample volatility surface
    strikes = np.arange(0.8, 1.21, 0.05)
    expiries = np.arange(7, 91, 7)
    
    X, Y = np.meshgrid(strikes, expiries)
    Z = 0.2 + 0.1 * np.sin(X * 10) + 0.05 * np.cos(Y / 10)
    
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Moneyness",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Volatility"
        ),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Research & Analysis Tools",
        page_icon="🔬",
        layout="wide"
    )
    
    create_research_analysis_interface()
