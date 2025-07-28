"""
Options Chain Analyzer for NEAT Options System
Advanced options analysis and visualization tools.
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
from scipy.stats import norm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import NEAT components with error handling
try:
    from neat_options_advisor import neat_advisor
    from neat_framework import neat_evolution
    NEAT_AVAILABLE = True
except ImportError as e:
    NEAT_AVAILABLE = False
    
    # Create mock objects for development
    class MockNeatAdvisor:
        sp500_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        network = None
    
    class MockNeatEvolution:
        @staticmethod
        def create_feature_vector(symbol, date):
            return None
    
    neat_advisor = MockNeatAdvisor()
    neat_evolution = MockNeatEvolution()

def create_options_analyzer():
    """Create comprehensive options chain analysis interface."""
    
    st.header("📊 Options Chain Analyzer")
    st.markdown("Deep analysis of options chains with Greeks, volatility, and probability calculations")
    
    # Symbol input
    col1, col2 = st.columns([1, 3])
    
    with col1:
        symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter stock symbol for options analysis"
        ).upper()
    
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type",
            options=["Full Chain Analysis", "Put Focus", "Call Focus", "Volatility Analysis", "Greeks Analysis"],
            index=1,
            help="Select type of analysis to perform"
        )
    
    if st.button("🔍 Analyze Options Chain", type="primary"):
        with st.spinner(f"Analyzing options chain for {symbol}..."):
            try:
                # Get stock and options data
                ticker = yf.Ticker(symbol)
                stock_info = ticker.info
                current_price = ticker.history(period="1d")['Close'].iloc[-1]
                
                # Get options expirations
                expirations = ticker.options
                
                if not expirations:
                    st.error(f"❌ No options data available for {symbol}")
                    return
                
                # Display stock info
                display_stock_info(symbol, stock_info, current_price)
                
                # Analyze options for multiple expirations
                analyze_options_chain(ticker, symbol, current_price, expirations, analysis_type)
                
            except Exception as e:
                st.error(f"❌ Error analyzing {symbol}: {e}")

def display_stock_info(symbol, info, current_price):
    """Display current stock information."""
    
    st.subheader(f"📈 {symbol} Stock Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
        st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B")
    
    with col2:
        st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}")
        st.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}")
    
    with col3:
        st.metric("Volume", f"{info.get('volume', 0):,}")
        st.metric("Avg Volume", f"{info.get('averageVolume', 0):,}")
    
    with col4:
        st.metric("Beta", f"{info.get('beta', 0):.2f}")
        st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.1f}" if info.get('trailingPE') else "N/A")

def analyze_options_chain(ticker, symbol, current_price, expirations, analysis_type):
    """Analyze options chain for multiple expirations."""
    
    st.subheader("🎯 Options Chain Analysis")
    
    # Select expirations to analyze (focus on 1-8 weeks out)
    target_expirations = []
    for exp in expirations[:8]:  # First 8 expirations
        exp_date = datetime.strptime(exp, '%Y-%m-%d')
        days_to_exp = (exp_date - datetime.now()).days
        if 7 <= days_to_exp <= 56:  # 1-8 weeks
            target_expirations.append(exp)
    
    if not target_expirations:
        target_expirations = expirations[:3]  # Fallback to first 3
    
    # Tabs for different expirations
    tabs = st.tabs([f"Exp: {exp}" for exp in target_expirations])
    
    for i, (tab, expiration) in enumerate(zip(tabs, target_expirations)):
        with tab:
            analyze_single_expiration(ticker, symbol, current_price, expiration, analysis_type)

def analyze_single_expiration(ticker, symbol, current_price, expiration, analysis_type):
    """Analyze options for a single expiration date."""
    
    try:
        # Get options chain
        options_chain = ticker.option_chain(expiration)
        calls = options_chain.calls
        puts = options_chain.puts
        
        if calls.empty and puts.empty:
            st.warning(f"No options data for expiration {expiration}")
            return
        
        # Calculate days to expiration
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        days_to_exp = (exp_date - datetime.now()).days
        
        st.markdown(f"**Expiration: {expiration} ({days_to_exp} days)**")
        
        # Enhanced options analysis
        if not puts.empty:
            puts = enhance_options_data(puts, current_price, days_to_exp, 'put')
        if not calls.empty:
            calls = enhance_options_data(calls, current_price, days_to_exp, 'call')
        
        # Analysis based on type
        if analysis_type == "Full Chain Analysis":
            display_full_chain_analysis(calls, puts, current_price, days_to_exp)
        elif analysis_type == "Put Focus":
            display_put_analysis(puts, current_price, days_to_exp)
        elif analysis_type == "Call Focus":
            display_call_analysis(calls, current_price, days_to_exp)
        elif analysis_type == "Volatility Analysis":
            display_volatility_analysis(calls, puts, current_price)
        elif analysis_type == "Greeks Analysis":
            display_greeks_analysis(calls, puts, current_price)
        
    except Exception as e:
        st.error(f"Error analyzing expiration {expiration}: {e}")

def enhance_options_data(options_df, current_price, days_to_exp, option_type):
    """Enhance options dataframe with additional calculations."""
    
    if options_df.empty:
        return options_df
    
    df = options_df.copy()
    
    # Calculate additional metrics
    df['moneyness'] = df['strike'] / current_price
    df['time_to_exp'] = days_to_exp / 365.0
    
    # Calculate intrinsic and time value
    if option_type == 'put':
        df['intrinsic_value'] = np.maximum(df['strike'] - current_price, 0)
    else:
        df['intrinsic_value'] = np.maximum(current_price - df['strike'], 0)
    
    df['time_value'] = df['lastPrice'] - df['intrinsic_value']
    df['time_value'] = np.maximum(df['time_value'], 0)  # Can't be negative
    
    # Calculate bid-ask spread
    df['bid_ask_spread'] = df['ask'] - df['bid']
    df['bid_ask_spread_pct'] = (df['bid_ask_spread'] / df['lastPrice']) * 100
    
    # Calculate premium yield for cash-secured puts
    if option_type == 'put':
        df['premium_yield'] = (df['lastPrice'] / df['strike']) * 100
        df['weekly_yield'] = df['premium_yield'] * (52 * 7 / days_to_exp) if days_to_exp > 0 else 0
        df['assignment_prob'] = calculate_assignment_probability(df['strike'], current_price, df['impliedVolatility'], days_to_exp, option_type)
    
    # Calculate break-even points
    if option_type == 'put':
        df['breakeven'] = df['strike'] - df['lastPrice']
    else:
        df['breakeven'] = df['strike'] + df['lastPrice']
    
    return df

def calculate_assignment_probability(strike, current_price, iv, days_to_exp, option_type):
    """Calculate probability of assignment using Black-Scholes."""
    
    if days_to_exp <= 0 or iv <= 0:
        return 0.0
    
    try:
        # Risk-free rate (approximate)
        r = 0.05
        
        # Time to expiration in years
        T = days_to_exp / 365.0
        
        # Calculate d2 from Black-Scholes
        d2 = (np.log(current_price / strike) + (r - 0.5 * iv**2) * T) / (iv * np.sqrt(T))
        
        if option_type == 'put':
            # Probability that stock price < strike at expiration
            prob = norm.cdf(-d2)
        else:
            # Probability that stock price > strike at expiration
            prob = norm.cdf(d2)
        
        return prob
    except:
        return 0.0

def display_full_chain_analysis(calls, puts, current_price, days_to_exp):
    """Display comprehensive options chain analysis."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📉 PUT OPTIONS")
        if not puts.empty:
            # Filter puts near the money
            puts_filtered = puts[
                (puts['moneyness'] >= 0.85) & 
                (puts['moneyness'] <= 1.05) &
                (puts['volume'] > 0)
            ].sort_values('strike', ascending=False)
            
            if not puts_filtered.empty:
                display_puts_table(puts_filtered.head(10))
            else:
                st.info("No liquid puts near the money")
        else:
            st.info("No put options data available")
    
    with col2:
        st.markdown("### 📈 CALL OPTIONS")
        if not calls.empty:
            # Filter calls near the money
            calls_filtered = calls[
                (calls['moneyness'] >= 0.95) & 
                (calls['moneyness'] <= 1.15) &
                (calls['volume'] > 0)
            ].sort_values('strike')
            
            if not calls_filtered.empty:
                display_calls_table(calls_filtered.head(10))
            else:
                st.info("No liquid calls near the money")
        else:
            st.info("No call options data available")
    
    # Volatility smile
    if not puts.empty or not calls.empty:
        st.markdown("### 📊 Implied Volatility Smile")
        plot_volatility_smile(calls, puts, current_price)

def display_put_analysis(puts, current_price, days_to_exp):
    """Display detailed put options analysis."""
    
    if puts.empty:
        st.info("No put options data available")
        return
    
    # Focus on cash-secured put opportunities
    st.markdown("### 🎯 Cash-Secured Put Opportunities")
    
    # Filter for good put selling opportunities
    good_puts = puts[
        (puts['moneyness'] >= 0.90) &  # 10% or less OTM
        (puts['moneyness'] <= 0.98) &  # Not too far ITM
        (puts['volume'] > 10) &        # Decent volume
        (puts['openInterest'] > 50) &  # Good open interest
        (puts['bid_ask_spread_pct'] < 10)  # Reasonable spread
    ].sort_values('weekly_yield', ascending=False)
    
    if not good_puts.empty:
        st.markdown("**🏆 Top Cash-Secured Put Opportunities:**")
        display_puts_table(good_puts.head(10))
        
        # Visualization
        fig = go.Figure()
        
        # Add scatter plot of puts
        fig.add_trace(go.Scatter(
            x=good_puts['strike'],
            y=good_puts['weekly_yield'],
            mode='markers',
            marker=dict(
                size=good_puts['volume'] / 10,
                color=good_puts['assignment_prob'],
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Assignment Prob")
            ),
            text=good_puts.apply(lambda row: 
                f"Strike: ${row['strike']:.2f}<br>"
                f"Premium: ${row['lastPrice']:.2f}<br>"
                f"Weekly Yield: {row['weekly_yield']:.2f}%<br>"
                f"Assignment Prob: {row['assignment_prob']:.1%}<br>"
                f"Volume: {row['volume']}", axis=1),
            hovertemplate='%{text}<extra></extra>',
            name='Put Options'
        ))
        
        # Add current price line
        fig.add_vline(x=current_price, line_dash="dash", line_color="red", 
                     annotation_text=f"Current Price: ${current_price:.2f}")
        
        fig.update_layout(
            title="Cash-Secured Put Opportunities",
            xaxis_title="Strike Price ($)",
            yaxis_title="Weekly Yield (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("⚠️ No suitable cash-secured put opportunities found")
    
    # Risk analysis
    st.markdown("### ⚖️ Risk Analysis")
    
    if not good_puts.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_yield = good_puts['weekly_yield'].mean()
            st.metric("Avg Weekly Yield", f"{avg_yield:.2f}%")
        
        with col2:
            avg_assignment = good_puts['assignment_prob'].mean()
            st.metric("Avg Assignment Prob", f"{avg_assignment:.1%}")
        
        with col3:
            max_yield = good_puts['weekly_yield'].max()
            st.metric("Max Weekly Yield", f"{max_yield:.2f}%")

def display_calls_table(calls_df):
    """Display formatted calls table."""
    
    display_df = calls_df[[
        'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest',
        'impliedVolatility', 'delta', 'gamma', 'theta'
    ]].copy()
    
    # Format columns
    display_df['lastPrice'] = display_df['lastPrice'].apply(lambda x: f"${x:.2f}")
    display_df['bid'] = display_df['bid'].apply(lambda x: f"${x:.2f}")
    display_df['ask'] = display_df['ask'].apply(lambda x: f"${x:.2f}")
    display_df['impliedVolatility'] = display_df['impliedVolatility'].apply(lambda x: f"{x:.1%}")
    display_df['delta'] = display_df['delta'].apply(lambda x: f"{x:.3f}")
    display_df['gamma'] = display_df['gamma'].apply(lambda x: f"{x:.3f}")
    display_df['theta'] = display_df['theta'].apply(lambda x: f"{x:.3f}")
    
    display_df.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'OI', 'IV', 'Delta', 'Gamma', 'Theta']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def display_puts_table(puts_df):
    """Display formatted puts table with cash-secured put metrics."""
    
    display_df = puts_df[[
        'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest',
        'impliedVolatility', 'premium_yield', 'weekly_yield', 'assignment_prob'
    ]].copy()
    
    # Format columns
    display_df['lastPrice'] = display_df['lastPrice'].apply(lambda x: f"${x:.2f}")
    display_df['bid'] = display_df['bid'].apply(lambda x: f"${x:.2f}")
    display_df['ask'] = display_df['ask'].apply(lambda x: f"${x:.2f}")
    display_df['impliedVolatility'] = display_df['impliedVolatility'].apply(lambda x: f"{x:.1%}")
    display_df['premium_yield'] = display_df['premium_yield'].apply(lambda x: f"{x:.2f}%")
    display_df['weekly_yield'] = display_df['weekly_yield'].apply(lambda x: f"{x:.2f}%")
    display_df['assignment_prob'] = display_df['assignment_prob'].apply(lambda x: f"{x:.1%}")
    
    display_df.columns = ['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'OI', 'IV', 'Yield', 'Weekly', 'Assign%']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def display_call_analysis(calls, current_price, days_to_exp):
    """Display detailed call options analysis."""
    
    if calls.empty:
        st.info("No call options data available")
        return
    
    st.markdown("### 📈 Call Options Analysis")
    
    # Filter for active calls
    active_calls = calls[
        (calls['volume'] > 0) &
        (calls['openInterest'] > 0)
    ].sort_values('volume', ascending=False)
    
    if not active_calls.empty:
        display_calls_table(active_calls.head(15))
        
        # Call volume analysis
        st.markdown("### 📊 Call Volume Analysis")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=active_calls['strike'],
            y=active_calls['volume'],
            name='Volume',
            marker_color='green'
        ))
        
        fig.add_vline(x=current_price, line_dash="dash", line_color="red",
                     annotation_text=f"Current: ${current_price:.2f}")
        
        fig.update_layout(
            title="Call Option Volume by Strike",
            xaxis_title="Strike Price ($)",
            yaxis_title="Volume",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_volatility_analysis(calls, puts, current_price):
    """Display implied volatility analysis."""
    
    st.markdown("### 📊 Implied Volatility Analysis")
    
    # Combine calls and puts
    all_options = []
    
    if not calls.empty:
        calls_data = calls[['strike', 'impliedVolatility', 'volume']].copy()
        calls_data['type'] = 'Call'
        all_options.append(calls_data)
    
    if not puts.empty:
        puts_data = puts[['strike', 'impliedVolatility', 'volume']].copy()
        puts_data['type'] = 'Put'
        all_options.append(puts_data)
    
    if all_options:
        combined_df = pd.concat(all_options, ignore_index=True)
        combined_df = combined_df[combined_df['volume'] > 0]  # Only liquid options
        
        if not combined_df.empty:
            plot_volatility_smile(calls, puts, current_price)
            
            # IV statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_iv = combined_df['impliedVolatility'].mean()
                st.metric("Average IV", f"{avg_iv:.1%}")
            
            with col2:
                max_iv = combined_df['impliedVolatility'].max()
                st.metric("Max IV", f"{max_iv:.1%}")
            
            with col3:
                min_iv = combined_df['impliedVolatility'].min()
                st.metric("Min IV", f"{min_iv:.1%}")

def display_greeks_analysis(calls, puts, current_price):
    """Display Greeks analysis."""
    
    st.markdown("### 🔢 Greeks Analysis")
    
    # Combine options data
    all_options = []
    
    if not calls.empty:
        calls_subset = calls[['strike', 'delta', 'gamma', 'theta', 'vega', 'volume']].copy()
        calls_subset['type'] = 'Call'
        all_options.append(calls_subset)
    
    if not puts.empty:
        puts_subset = puts[['strike', 'delta', 'gamma', 'theta', 'vega', 'volume']].copy()
        puts_subset['type'] = 'Put'
        all_options.append(puts_subset)
    
    if all_options:
        combined_df = pd.concat(all_options, ignore_index=True)
        combined_df = combined_df[combined_df['volume'] > 0]
        
        if not combined_df.empty:
            # Greeks visualization
            fig = go.Figure()
            
            # Delta plot
            calls_data = combined_df[combined_df['type'] == 'Call']
            puts_data = combined_df[combined_df['type'] == 'Put']
            
            if not calls_data.empty:
                fig.add_trace(go.Scatter(
                    x=calls_data['strike'],
                    y=calls_data['delta'],
                    mode='markers+lines',
                    name='Call Delta',
                    line=dict(color='green')
                ))
            
            if not puts_data.empty:
                fig.add_trace(go.Scatter(
                    x=puts_data['strike'],
                    y=puts_data['delta'],
                    mode='markers+lines',
                    name='Put Delta',
                    line=dict(color='red')
                ))
            
            fig.add_vline(x=current_price, line_dash="dash", line_color="blue",
                         annotation_text=f"Current: ${current_price:.2f}")
            
            fig.update_layout(
                title="Delta by Strike Price",
                xaxis_title="Strike Price ($)",
                yaxis_title="Delta",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def plot_volatility_smile(calls, puts, current_price):
    """Plot implied volatility smile."""
    
    fig = go.Figure()
    
    # Plot calls
    if not calls.empty:
        calls_filtered = calls[calls['volume'] > 0]
        if not calls_filtered.empty:
            fig.add_trace(go.Scatter(
                x=calls_filtered['strike'],
                y=calls_filtered['impliedVolatility'],
                mode='markers+lines',
                name='Calls',
                line=dict(color='green'),
                marker=dict(size=calls_filtered['volume']/50)
            ))
    
    # Plot puts
    if not puts.empty:
        puts_filtered = puts[puts['volume'] > 0]
        if not puts_filtered.empty:
            fig.add_trace(go.Scatter(
                x=puts_filtered['strike'],
                y=puts_filtered['impliedVolatility'],
                mode='markers+lines',
                name='Puts',
                line=dict(color='red'),
                marker=dict(size=puts_filtered['volume']/50)
            ))
    
    # Add current price line
    fig.add_vline(x=current_price, line_dash="dash", line_color="blue",
                 annotation_text=f"Current: ${current_price:.2f}")
    
    fig.update_layout(
        title="Implied Volatility Smile",
        xaxis_title="Strike Price ($)",
        yaxis_title="Implied Volatility",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Options Chain Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    create_options_analyzer()
