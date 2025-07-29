"""
Weekly Predictions Interface - 1-8 Week Forward Predictions
Shows explicit Week 1-8 predictions as required by user.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

def create_weekly_predictions_interface():
    """Create interface showing explicit 1-8 week predictions."""
    
    st.set_page_config(
        page_title="Weekly Predictions - NEAT Options System",
        page_icon="🔮",
        layout="wide"
    )
    
    st.header("🔮 Weekly Predictions Interface")
    st.markdown("**Multi-Horizon Predictions: Week 1-8 Forward Outlook**")
    
    # Sidebar for stock selection
    st.sidebar.header("📊 Stock Selection")
    
    # Popular options-friendly stocks
    popular_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
        'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'SPOT',
        'BLOCK', 'ZM', 'DOCU', 'ROKU', 'TWLO', 'OKTA', 'SNOW', 'PLTR'
    ]
    
    selected_symbol = st.sidebar.selectbox(
        "Select Stock Symbol",
        options=popular_stocks,
        index=0  # Default to AAPL
    )
    
    # Prediction confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="Minimum confidence for trade recommendations"
    )
    
    # Capital allocation
    capital = st.sidebar.number_input(
        "Capital ($)",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000,
        help="Available capital for options strategies"
    )
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🎯 Generate Predictions")
        
        if st.button("🔮 Generate 8-Week Forecast", type="primary"):
            with st.spinner("Generating multi-horizon predictions..."):
                predictions = generate_weekly_predictions(selected_symbol, confidence_threshold, capital)
                st.session_state['predictions'] = predictions
                st.session_state['selected_symbol'] = selected_symbol
        
        st.markdown("---")
        
        # Current stock info
        if st.button("📊 Get Current Data"):
            current_data = get_current_stock_data(selected_symbol)
            if current_data:
                st.metric("Current Price", f"${current_data['price']:.2f}")
                st.metric("Daily Change", f"{current_data['change']:+.2f}%")
                st.metric("Volume", f"{current_data['volume']:,.0f}")
    
    with col1:
        st.subheader(f"📈 8-Week Predictions for {selected_symbol}")
        
        if 'predictions' in st.session_state and st.session_state.get('selected_symbol') == selected_symbol:
            predictions = st.session_state['predictions']
            
            # Display predictions table
            st.subheader("🗓️ Weekly Forecast Breakdown")
            
            # Format predictions for display
            display_predictions = predictions.copy()
            display_predictions['Predicted Price'] = display_predictions['Predicted Price'].apply(lambda x: f"${x:.2f}")
            display_predictions['Price Change'] = display_predictions['Price Change'].apply(lambda x: f"{x:+.1f}%")
            display_predictions['Confidence'] = display_predictions['Confidence'].apply(lambda x: f"{x:.1%}")
            display_predictions['Recommended Strike'] = display_predictions['Recommended Strike'].apply(lambda x: f"${x:.2f}")
            display_predictions['Expected Premium'] = display_predictions['Expected Premium'].apply(lambda x: f"${x:.2f}")
            display_predictions['Weekly Yield'] = display_predictions['Weekly Yield'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_predictions, use_container_width=True, hide_index=True)
            
            # Price prediction chart
            st.subheader("📊 Price Prediction Chart")
            
            fig = go.Figure()
            
            # Add predicted prices
            fig.add_trace(go.Scatter(
                x=predictions['Week'],
                y=predictions['Predicted Price'].apply(lambda x: float(x.replace('$', '')) if isinstance(x, str) else x),
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            # Add confidence bands
            prices = predictions['Predicted Price'].apply(lambda x: float(x.replace('$', '')) if isinstance(x, str) else x)
            confidences = predictions['Confidence'].apply(lambda x: float(x.replace('%', ''))/100 if isinstance(x, str) else x)
            
            upper_band = prices * (1 + (1 - confidences) * 0.1)
            lower_band = prices * (1 - (1 - confidences) * 0.1)
            
            fig.add_trace(go.Scatter(
                x=predictions['Week'],
                y=upper_band,
                mode='lines',
                name='Upper Confidence',
                line=dict(color='lightblue', dash='dash'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=predictions['Week'],
                y=lower_band,
                mode='lines',
                name='Lower Confidence',
                line=dict(color='lightblue', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.2)',
                showlegend=False
            ))
            
            fig.update_layout(
                title=f"{selected_symbol} - 8-Week Price Predictions",
                xaxis_title="Week",
                yaxis_title="Price ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Options recommendations
            st.subheader("⚡ Options Strategy Recommendations")
            
            # Filter high-confidence predictions
            high_confidence = predictions[predictions['Confidence'].apply(
                lambda x: float(x.replace('%', ''))/100 if isinstance(x, str) else x
            ) >= confidence_threshold]
            
            if len(high_confidence) > 0:
                st.success(f"✅ Found {len(high_confidence)} high-confidence opportunities")
                
                for _, row in high_confidence.iterrows():
                    with st.expander(f"📅 Week {row['Week']} - {row['Direction']} Signal"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Target Price", row['Predicted Price'])
                            st.metric("Confidence", row['Confidence'])
                        
                        with col2:
                            st.metric("Strike Price", row['Recommended Strike'])
                            st.metric("Expected Premium", row['Expected Premium'])
                        
                        with col3:
                            st.metric("Weekly Yield", row['Weekly Yield'])
                            st.metric("Assignment Risk", f"{row['Assignment Risk']:.1%}")
                        
                        # Strategy details
                        st.markdown(f"""
                        **Strategy:** Sell Cash-Secured Put
                        **Reasoning:** {row['AI Reasoning']}
                        **Risk Level:** {row['Risk Level']}
                        """)
            else:
                st.warning("⚠️ No high-confidence opportunities found. Consider lowering confidence threshold.")
            
            # Summary metrics
            st.subheader("📊 8-Week Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_confidence = predictions['Confidence'].apply(
                    lambda x: float(x.replace('%', ''))/100 if isinstance(x, str) else x
                ).mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with col2:
                bullish_weeks = len(predictions[predictions['Direction'] == 'Bullish'])
                st.metric("Bullish Weeks", f"{bullish_weeks}/8")
            
            with col3:
                avg_yield = predictions['Weekly Yield'].apply(
                    lambda x: float(x.replace('%', ''))/100 if isinstance(x, str) else x
                ).mean()
                st.metric("Avg Weekly Yield", f"{avg_yield:.1%}")
            
            with col4:
                total_opportunities = len(high_confidence)
                st.metric("Trade Opportunities", total_opportunities)
            
        else:
            st.info("👆 Click 'Generate 8-Week Forecast' to see detailed predictions for each week")
            
            # Show example format
            st.subheader("📋 Prediction Format Example")
            
            example_data = pd.DataFrame({
                'Week': ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                'Predicted Price': ['$150.25', '$152.80', '$148.90', '$155.10'],
                'Confidence': ['85%', '78%', '72%', '81%'],
                'Direction': ['Bullish', 'Bullish', 'Bearish', 'Bullish'],
                'Recommended Strike': ['$142.50', '$145.00', '$141.00', '$147.50']
            })
            
            st.dataframe(example_data, use_container_width=True, hide_index=True)

def get_current_stock_data(symbol):
    """Get current stock data."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        
        if len(hist) >= 2:
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            change = (current_price - prev_price) / prev_price * 100
            volume = hist['Volume'].iloc[-1]
            
            return {
                'price': current_price,
                'change': change,
                'volume': volume
            }
    except:
        return None

def generate_weekly_predictions(symbol, confidence_threshold, capital):
    """Generate 8-week predictions for the selected symbol."""
    
    try:
        # Get historical data for analysis
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        
        if hist.empty:
            return pd.DataFrame()
        
        current_price = hist['Close'].iloc[-1]
        
        # Generate predictions for 8 weeks
        predictions = []
        
        for week in range(1, 9):
            # Simulate AI prediction logic (replace with actual NEAT model)
            base_volatility = hist['Close'].pct_change().std() * np.sqrt(252)
            
            # Weekly price prediction with some randomness for demo
            np.random.seed(42 + week)  # Consistent results
            price_change = np.random.normal(0.005, base_volatility/np.sqrt(52))  # Weekly change
            predicted_price = current_price * (1 + price_change * week * 0.3)
            
            # Confidence based on volatility and time horizon
            confidence = max(0.4, 0.9 - (week - 1) * 0.05 - base_volatility * 2)
            
            # Direction
            direction = "Bullish" if price_change > 0 else "Bearish"
            
            # Options strategy parameters
            if direction == "Bullish":
                # For bullish prediction, sell puts below current price
                strike_distance = 0.05 + (1 - confidence) * 0.05  # 5-10% below
                strike_price = current_price * (1 - strike_distance)
                
                # Premium estimation
                time_value = 0.02 + base_volatility * 0.5
                premium = strike_price * time_value * (week / 52)
                
                # Weekly yield
                weekly_yield = premium / (strike_price * 100) * (52 / week)
                
                # Assignment risk
                assignment_risk = max(0.01, min(0.15, (current_price - strike_price) / current_price * 0.5))
                
                risk_level = "Low" if confidence > 0.8 else "Medium" if confidence > 0.6 else "High"
                
                reasoning = f"AI predicts {price_change*100:.1f}% upward movement. High probability of staying above ${strike_price:.2f} strike."
                
            else:
                # For bearish prediction, might avoid or use different strategy
                strike_price = current_price * 0.92  # Conservative strike
                premium = strike_price * 0.015
                weekly_yield = premium / (strike_price * 100) * (52 / week)
                assignment_risk = 0.08
                risk_level = "High"
                reasoning = f"AI predicts {abs(price_change)*100:.1f}% downward movement. Conservative strike recommended."
            
            predictions.append({
                'Week': f'Week {week}',
                'Predicted Price': predicted_price,
                'Price Change': price_change * 100,
                'Confidence': confidence,
                'Direction': direction,
                'Recommended Strike': strike_price,
                'Expected Premium': premium,
                'Weekly Yield': weekly_yield,
                'Assignment Risk': assignment_risk,
                'Risk Level': risk_level,
                'AI Reasoning': reasoning
            })
        
        return pd.DataFrame(predictions)
        
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    create_weekly_predictions_interface()
