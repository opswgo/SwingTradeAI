"""
Stock Screening Interface for NEAT Options System
Advanced filtering and analysis tools for options trading candidates.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import NEAT components with error handling
try:
    from neat_options_advisor import neat_advisor
    from neat_framework import neat_evolution
    NEAT_AVAILABLE = True
except ImportError as e:
    st.error(f"NEAT components not available: {e}")
    st.info("Please ensure neat-python is installed: pip install neat-python")
    NEAT_AVAILABLE = False
    
    # Create mock objects for development
    class MockNeatAdvisor:
        sp500_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    class MockNeatEvolution:
        @staticmethod
        def create_feature_vector(symbol, date):
            return None
    
    neat_advisor = MockNeatAdvisor()
    neat_evolution = MockNeatEvolution()

def create_stock_screener():
    """Create comprehensive stock screening interface."""
    
    st.header("🔍 Advanced Stock Screening")
    st.markdown("Filter and analyze stocks for optimal options trading opportunities")
    
    # Screening parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Market Cap & Volume")
        min_market_cap = st.selectbox(
            "Minimum Market Cap",
            options=[1e9, 5e9, 10e9, 25e9, 50e9, 100e9],
            index=1,
            format_func=lambda x: f"${x/1e9:.0f}B"
        )
        
        min_avg_volume = st.number_input(
            "Min Average Volume (millions)",
            min_value=0.1,
            max_value=100.0,
            value=1.0,
            step=0.1
        ) * 1e6
        
        min_price = st.number_input(
            "Minimum Stock Price ($)",
            min_value=1.0,
            max_value=1000.0,
            value=20.0,
            step=1.0
        )
        
        max_price = st.number_input(
            "Maximum Stock Price ($)",
            min_value=10.0,
            max_value=2000.0,
            value=500.0,
            step=10.0
        )
    
    with col2:
        st.subheader("📈 Technical Criteria")
        rsi_range = st.slider(
            "RSI Range",
            min_value=0,
            max_value=100,
            value=(30, 70),
            help="Relative Strength Index range"
        )
        
        volatility_range = st.slider(
            "Implied Volatility Range (%)",
            min_value=10,
            max_value=100,
            value=(15, 60),
            help="Acceptable implied volatility range"
        )
        
        trend_filter = st.selectbox(
            "Trend Direction",
            options=["Any", "Bullish", "Bearish", "Sideways"],
            index=1,
            help="Filter by overall trend direction"
        )
        
        momentum_filter = st.selectbox(
            "Momentum",
            options=["Any", "Strong Up", "Weak Up", "Neutral", "Weak Down", "Strong Down"],
            index=0
        )
    
    with col3:
        st.subheader("⚖️ Options Criteria")
        min_open_interest = st.number_input(
            "Min Open Interest",
            min_value=0,
            max_value=10000,
            value=100,
            step=50,
            help="Minimum open interest for liquidity"
        )
        
        bid_ask_spread_max = st.slider(
            "Max Bid-Ask Spread (%)",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum acceptable bid-ask spread"
        )
        
        exclude_earnings = st.checkbox(
            "Exclude Earnings (8 weeks)",
            value=True,
            help="Exclude stocks with earnings in next 8 weeks"
        )
        
        sector_filter = st.multiselect(
            "Sectors",
            options=["Technology", "Healthcare", "Financial", "Consumer", "Industrial", "Energy", "Utilities", "Materials", "Real Estate", "Communication"],
            default=[],
            help="Filter by specific sectors (empty = all sectors)"
        )
    
    # Run screening button
    if st.button("🚀 Run Advanced Screening", type="primary"):
        with st.spinner("Screening stocks with advanced criteria..."):
            try:
                # Get base universe
                symbols = neat_advisor.sp500_symbols
                
                # Apply screening filters
                screened_stocks = screen_stocks_advanced(
                    symbols=symbols,
                    min_market_cap=min_market_cap,
                    min_avg_volume=min_avg_volume,
                    min_price=min_price,
                    max_price=max_price,
                    rsi_range=rsi_range,
                    volatility_range=volatility_range,
                    trend_filter=trend_filter,
                    momentum_filter=momentum_filter,
                    min_open_interest=min_open_interest,
                    bid_ask_spread_max=bid_ask_spread_max,
                    exclude_earnings=exclude_earnings,
                    sector_filter=sector_filter
                )
                
                if screened_stocks:
                    st.success(f"✅ Found {len(screened_stocks)} stocks meeting criteria")
                    
                    # Display results
                    display_screening_results(screened_stocks)
                    
                    # Store in session state for other tools
                    st.session_state['screened_stocks'] = screened_stocks
                    
                else:
                    st.warning("⚠️ No stocks meet the specified criteria. Try relaxing some filters.")
                    
            except Exception as e:
                st.error(f"❌ Screening failed: {e}")

def screen_stocks_advanced(symbols, min_market_cap, min_avg_volume, min_price, max_price,
                          rsi_range, volatility_range, trend_filter, momentum_filter,
                          min_open_interest, bid_ask_spread_max, exclude_earnings, sector_filter):
    """Apply advanced screening criteria to stock list."""
    
    screened_stocks = []
    
    for symbol in symbols:
        try:
            # Get stock data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="3mo")
            
            if hist.empty:
                continue
            
            # Basic filters
            market_cap = info.get('marketCap', 0)
            if market_cap < min_market_cap:
                continue
            
            current_price = hist['Close'].iloc[-1]
            if current_price < min_price or current_price > max_price:
                continue
            
            avg_volume = hist['Volume'].tail(20).mean()
            if avg_volume < min_avg_volume:
                continue
            
            # Technical analysis
            rsi = calculate_rsi(hist['Close'])
            if rsi < rsi_range[0] or rsi > rsi_range[1]:
                continue
            
            # Trend analysis
            if trend_filter != "Any":
                trend = analyze_trend(hist)
                if trend != trend_filter.lower():
                    continue
            
            # Momentum analysis
            if momentum_filter != "Any":
                momentum = analyze_momentum(hist)
                if momentum != momentum_filter.lower().replace(" ", "_"):
                    continue
            
            # Sector filter
            if sector_filter:
                sector = info.get('sector', '')
                if not any(s.lower() in sector.lower() for s in sector_filter):
                    continue
            
            # Calculate additional metrics
            volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
            
            # Store stock data
            stock_data = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'market_cap': market_cap,
                'current_price': current_price,
                'avg_volume': avg_volume,
                'rsi': rsi,
                'volatility': volatility,
                'trend': analyze_trend(hist),
                'momentum': analyze_momentum(hist),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'pe_ratio': info.get('trailingPE', 0),
                'beta': info.get('beta', 1.0)
            }
            
            screened_stocks.append(stock_data)
            
        except Exception as e:
            continue
    
    return screened_stocks

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

def analyze_trend(hist):
    """Analyze overall trend direction."""
    if len(hist) < 20:
        return "neutral"
    
    sma_20 = hist['Close'].rolling(20).mean()
    sma_50 = hist['Close'].rolling(50).mean() if len(hist) >= 50 else sma_20
    
    current_price = hist['Close'].iloc[-1]
    sma_20_current = sma_20.iloc[-1]
    sma_50_current = sma_50.iloc[-1]
    
    if current_price > sma_20_current > sma_50_current:
        return "bullish"
    elif current_price < sma_20_current < sma_50_current:
        return "bearish"
    else:
        return "sideways"

def analyze_momentum(hist):
    """Analyze momentum strength."""
    if len(hist) < 10:
        return "neutral"
    
    returns_5d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-6] - 1) * 100
    returns_20d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-21] - 1) * 100 if len(hist) >= 21 else returns_5d
    
    if returns_5d > 5 and returns_20d > 10:
        return "strong_up"
    elif returns_5d > 2 and returns_20d > 5:
        return "weak_up"
    elif returns_5d < -5 and returns_20d < -10:
        return "strong_down"
    elif returns_5d < -2 and returns_20d < -5:
        return "weak_down"
    else:
        return "neutral"

def display_screening_results(screened_stocks):
    """Display screening results in organized format."""
    
    # Convert to DataFrame
    df = pd.DataFrame(screened_stocks)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stocks", len(df))
    with col2:
        avg_market_cap = df['market_cap'].mean() / 1e9
        st.metric("Avg Market Cap", f"${avg_market_cap:.1f}B")
    with col3:
        avg_rsi = df['rsi'].mean()
        st.metric("Avg RSI", f"{avg_rsi:.1f}")
    with col4:
        avg_volatility = df['volatility'].mean()
        st.metric("Avg Volatility", f"{avg_volatility:.1f}%")
    
    # Sector distribution
    st.subheader("📊 Sector Distribution")
    sector_counts = df['sector'].value_counts()
    fig_sector = px.pie(
        values=sector_counts.values,
        names=sector_counts.index,
        title="Stocks by Sector"
    )
    st.plotly_chart(fig_sector, use_container_width=True)
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Format display dataframe
    display_df = df.copy()
    display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B")
    display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
    display_df['avg_volume'] = display_df['avg_volume'].apply(lambda x: f"{x/1e6:.1f}M")
    display_df['rsi'] = display_df['rsi'].apply(lambda x: f"{x:.1f}")
    display_df['volatility'] = display_df['volatility'].apply(lambda x: f"{x:.1f}%")
    display_df['dividend_yield'] = display_df['dividend_yield'].apply(lambda x: f"{x:.1f}%")
    display_df['pe_ratio'] = display_df['pe_ratio'].apply(lambda x: f"{x:.1f}" if x > 0 else "N/A")
    display_df['beta'] = display_df['beta'].apply(lambda x: f"{x:.2f}")
    
    # Select columns for display
    display_columns = [
        'symbol', 'company_name', 'sector', 'current_price', 'market_cap',
        'avg_volume', 'rsi', 'volatility', 'trend', 'momentum', 'dividend_yield', 'pe_ratio', 'beta'
    ]
    
    display_df = display_df[display_columns]
    display_df.columns = [
        'Symbol', 'Company', 'Sector', 'Price', 'Market Cap',
        'Avg Volume', 'RSI', 'Volatility', 'Trend', 'Momentum', 'Div Yield', 'P/E', 'Beta'
    ]
    
    # Add sorting options
    sort_by = st.selectbox(
        "Sort by:",
        options=['Symbol', 'Market Cap', 'Price', 'RSI', 'Volatility', 'Div Yield'],
        index=1
    )
    
    if sort_by in ['Market Cap', 'Price', 'RSI', 'Volatility', 'Div Yield']:
        # Convert back to numeric for sorting
        if sort_by == 'Market Cap':
            sort_values = df['market_cap']
        elif sort_by == 'Price':
            sort_values = df['current_price']
        elif sort_by == 'RSI':
            sort_values = df['rsi']
        elif sort_by == 'Volatility':
            sort_values = df['volatility']
        elif sort_by == 'Div Yield':
            sort_values = df['dividend_yield']
        
        sorted_indices = sort_values.sort_values(ascending=False).index
        display_df = display_df.iloc[sorted_indices]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Export options
    st.subheader("📤 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate NEAT Recommendations"):
            symbols = df['symbol'].tolist()[:10]  # Top 10 for performance
            with st.spinner("Generating NEAT recommendations..."):
                recommendations = neat_advisor.get_recommendations(symbols=symbols)
                if recommendations:
                    st.session_state['neat_recommendations'] = recommendations
                    st.success(f"✅ Generated {len(recommendations)} NEAT recommendations")
                else:
                    st.warning("⚠️ No NEAT recommendations generated")
    
    with col2:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv_data,
            file_name=f"stock_screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    st.set_page_config(
        page_title="Stock Screening Interface",
        page_icon="🔍",
        layout="wide"
    )
    
    create_stock_screener()
