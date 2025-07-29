"""
Simple Stock Screening Interface - Standalone Version
No NEAT dependencies required - works immediately.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

def create_simple_stock_screener():
    """Create a simple stock screening interface that works immediately."""
    
    st.set_page_config(
        page_title="Stock Screening Interface",
        page_icon="🔍",
        layout="wide"
    )
    
    st.header("🔍 Stock Screening Interface")
    st.markdown("Advanced stock screening with real-time data")
    
    # Sidebar for filters
    st.sidebar.header("🔧 Screening Filters")
    
    # Predefined stock universe (updated with current ticker symbols)
    popular_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
        'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'SPOT',
        'BLOCK', 'ZM', 'DOCU', 'ROKU', 'TWLO', 'OKTA', 'SNOW', 'PLTR'
    ]
    
    # Market cap filter
    st.sidebar.subheader("💰 Market Cap")
    market_cap_min = st.sidebar.slider(
        "Minimum Market Cap ($B)",
        min_value=0,
        max_value=500,
        value=10,
        step=5
    )
    
    # Price filter
    st.sidebar.subheader("💵 Price Range")
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=0,
        max_value=1000,
        value=(20, 500),
        step=10
    )
    
    # Volume filter
    st.sidebar.subheader("📊 Volume")
    min_volume = st.sidebar.slider(
        "Minimum Daily Volume (M)",
        min_value=0,
        max_value=100,
        value=1,
        step=1
    ) * 1000000
    
    # Performance filter
    st.sidebar.subheader("📈 Performance")
    performance_period = st.sidebar.selectbox(
        "Performance Period",
        options=["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
        index=1
    )
    
    performance_min = st.sidebar.slider(
        "Minimum Performance (%)",
        min_value=-50,
        max_value=100,
        value=-10,
        step=5
    )
    
    # Volatility filter
    volatility_max = st.sidebar.slider(
        "Maximum Volatility (%)",
        min_value=10,
        max_value=100,
        value=50,
        step=5
    )
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🎯 Screening Controls")
        
        if st.button("🔍 Run Screen", type="primary"):
            with st.spinner("Screening stocks..."):
                results = run_stock_screen(
                    popular_stocks,
                    market_cap_min,
                    price_range,
                    min_volume,
                    performance_period,
                    performance_min,
                    volatility_max
                )
                st.session_state['screening_results'] = results
        
        st.markdown("---")
        
        # Quick stats
        if 'screening_results' in st.session_state:
            results = st.session_state['screening_results']
            st.metric("Stocks Screened", len(popular_stocks))
            st.metric("Stocks Passed", len(results))
            st.metric("Pass Rate", f"{len(results)/len(popular_stocks)*100:.1f}%")
    
    with col1:
        st.subheader("📊 Screening Results")
        
        if 'screening_results' in st.session_state:
            results = st.session_state['screening_results']
            
            if len(results) > 0:
                # Display results table
                display_df = results.copy()
                display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
                display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: f"${x:.1f}B")
                display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{x/1000000:.1f}M")
                display_df['Performance'] = display_df['Performance'].apply(lambda x: f"{x:+.1f}%")
                display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Performance chart
                st.subheader("📈 Performance Comparison")
                
                fig = px.scatter(
                    results,
                    x='Volatility',
                    y='Performance',
                    size='Market Cap',
                    color='Performance',
                    hover_name='Symbol',
                    title="Risk-Return Profile",
                    labels={'Volatility': 'Volatility (%)', 'Performance': f'{performance_period} Performance (%)'},
                    color_continuous_scale='RdYlGn'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Export functionality
                st.subheader("💾 Export Results")
                
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"stock_screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("No stocks passed the screening criteria. Try adjusting the filters.")
        else:
            st.info("Click 'Run Screen' to start screening stocks with your selected criteria.")
            
            # Show sample data
            st.subheader("📋 Sample Stocks")
            sample_data = pd.DataFrame({
                'Symbol': popular_stocks[:10],
                'Name': [
                    'Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.', 'Amazon.com Inc.',
                    'Tesla Inc.', 'NVIDIA Corp.', 'Meta Platforms', 'Netflix Inc.',
                    'Advanced Micro Devices', 'Intel Corp.'
                ]
            })
            st.dataframe(sample_data, use_container_width=True, hide_index=True)

def run_stock_screen(symbols, market_cap_min, price_range, min_volume, performance_period, performance_min, volatility_max):
    """Run stock screening with specified criteria."""
    
    results = []
    
    # Map performance period to days
    period_map = {
        "1 Week": 7,
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365
    }
    
    days = period_map[performance_period]
    
    for symbol in symbols:
        try:
            # Fetch stock data
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Get historical data
            hist = ticker.history(period="1y")
            
            if hist.empty or len(hist) < days:
                continue
            
            # Calculate metrics
            current_price = hist['Close'].iloc[-1]
            volume = hist['Volume'].iloc[-1]
            
            # Market cap (in billions)
            market_cap = info.get('marketCap', 0) / 1e9
            
            # Performance calculation
            if len(hist) >= days:
                past_price = hist['Close'].iloc[-days]
                performance = (current_price - past_price) / past_price * 100
            else:
                performance = 0
            
            # Volatility calculation (30-day)
            if len(hist) >= 30:
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.tail(30).std() * np.sqrt(252) * 100
            else:
                volatility = 0
            
            # Apply filters
            if (market_cap >= market_cap_min and
                price_range[0] <= current_price <= price_range[1] and
                volume >= min_volume and
                performance >= performance_min and
                volatility <= volatility_max):
                
                results.append({
                    'Symbol': symbol,
                    'Name': info.get('longName', symbol),
                    'Price': current_price,
                    'Market Cap': market_cap,
                    'Volume': volume,
                    'Performance': performance,
                    'Volatility': volatility,
                    'Sector': info.get('sector', 'Unknown')
                })
                
        except Exception as e:
            # Log the error for debugging but continue processing other stocks
            if "delisted" in str(e).lower() or "no data found" in str(e).lower():
                st.sidebar.warning(f"⚠️ {symbol}: Data unavailable (possibly delisted)")
            continue
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    create_simple_stock_screener()
