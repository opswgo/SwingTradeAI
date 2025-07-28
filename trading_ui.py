#!/usr/bin/env python3
"""
Highly Usable NEAT Trading UI
Interactive interface for training, validation, and predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import json
import os
from pathlib import Path
import sys

# Add the current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from practical_neat_system import PracticalNEATTrader
except ImportError:
    st.error("Could not import PracticalNEATTrader. Please ensure practical_neat_system.py is available.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="NEAT Options Trading System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'trader' not in st.session_state:
    st.session_state.trader = PracticalNEATTrader()
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'training_results' not in st.session_state:
    st.session_state.training_results = {}

def load_models():
    """Load any previously saved models."""
    models_file = Path("trained_models.json")
    if models_file.exists():
        try:
            with open(models_file, 'r') as f:
                data = json.load(f)
                st.session_state.training_results = data.get('results', {})
                # Note: NEAT genomes can't be serialized easily, so we'll retrain as needed
        except Exception as e:
            st.sidebar.warning(f"Could not load saved models: {e}")

def save_models():
    """Save training results (metadata only)."""
    models_file = Path("trained_models.json")
    try:
        data = {
            'results': st.session_state.training_results,
            'last_updated': datetime.now().isoformat()
        }
        with open(models_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.sidebar.warning(f"Could not save models: {e}")

def train_symbol(symbol: str, horizon_weeks: int = 4):
    """Train a model for a specific symbol."""
    with st.spinner(f"Training {symbol} for {horizon_weeks}-week options..."):
        try:
            # Get data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y", auto_adjust=True)
            
            if len(data) < 500:
                st.error(f"Insufficient data for {symbol} (only {len(data)} days)")
                return None
            
            # Train model
            result = st.session_state.trader.train_weekly_agent(symbol, data, horizon_weeks)
            
            if result['success']:
                # Store model and results
                st.session_state.trader.models[symbol] = result
                st.session_state.trained_models[symbol] = result
                st.session_state.training_results[symbol] = result['performance']
                save_models()
                return result
            else:
                st.error(f"Training failed for {symbol}: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            st.error(f"Error training {symbol}: {e}")
            return None

def create_prediction_chart(symbol: str, weeks_ahead: int = 8):
    """Create a Bollinger-band style prediction chart."""
    try:
        # Get recent data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="6mo", auto_adjust=True)
        
        if len(data) == 0:
            st.error(f"No data available for {symbol}")
            return None, None
        
        if symbol not in st.session_state.trader.models:
            st.warning(f"No trained model for {symbol}. Please train first.")
            return None, None
        
        # Get model
        model_info = st.session_state.trader.models[symbol]
        
        # Generate predictions for next 1-8 weeks
        predictions = []
        confidence_bands = []
        
        # Engineer features once
        features = st.session_state.trader.engineer_features(data)
        if len(features) == 0:
            st.error(f"Could not engineer features for {symbol}")
            return None, None
            
        current_features = features.iloc[-1].values
        
        import neat
        net = neat.nn.FeedForwardNetwork.create(
            model_info['genome'], model_info['config']
        )
        
        current_price = data['Close'].iloc[-1]
        
        for week in range(1, weeks_ahead + 1):
            # Get prediction
            prediction = net.activate(current_features)[0]
            
            # Convert prediction to price estimate
            predicted_change = prediction * 0.05 * week  # Scale by week
            predicted_price = current_price * (1 + predicted_change)
            
            predictions.append({
                'week': week,
                'predicted_price': predicted_price,
                'confidence': abs(prediction),
                'signal': prediction
            })
            
            # Create confidence bands
            confidence = abs(prediction)
            band_width = confidence * 0.03 * current_price * week
            confidence_bands.append({
                'week': week,
                'upper_band': predicted_price + band_width,
                'lower_band': predicted_price - band_width
            })
        
        # Create chart
        fig = go.Figure()
        
        # Historical prices
        recent_data = data.tail(30)
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2)
        ))
        
        # Future dates
        last_date = data.index[-1]
        future_dates = [last_date + timedelta(weeks=i) for i in range(1, weeks_ahead + 1)]
        
        # Prediction line
        pred_prices = [p['predicted_price'] for p in predictions]
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=pred_prices,
            mode='lines+markers',
            name='AI Predictions',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Confidence bands
        upper_band = [b['upper_band'] for b in confidence_bands]
        lower_band = [b['lower_band'] for b in confidence_bands]
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=upper_band,
            mode='lines',
            name='Upper Confidence',
            line=dict(color='rgba(255,0,0,0.3)', width=1),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lower_band,
            mode='lines',
            name='Lower Confidence',
            line=dict(color='rgba(255,0,0,0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            showlegend=False
        ))
        
        # Current price line
        current_price = data['Close'].iloc[-1]
        fig.add_hline(
            y=current_price,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Current: ${current_price:.2f}"
        )
        
        fig.update_layout(
            title=f"{symbol} - AI Predictions (Next {weeks_ahead} Weeks)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500
        )
        
        return fig, predictions
        
    except Exception as e:
        st.error(f"Error creating prediction chart: {e}")
        return None, None

def create_validation_chart(symbol: str, test_date: str, weeks_ahead: int = 4):
    """Create actual vs predicted validation chart."""
    try:
        test_date_dt = pd.to_datetime(test_date).tz_localize(None)  # Remove timezone
        
        # Get historical data
        ticker = yf.Ticker(symbol)
        start_date = test_date_dt - timedelta(days=180)
        end_date = test_date_dt + timedelta(weeks=weeks_ahead + 4)
        
        data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        if len(data) == 0:
            st.error(f"No data available for {symbol} around {test_date}")
            return None, None
        
        # Remove timezone from data index for comparison
        data.index = data.index.tz_localize(None)
        
        if test_date_dt not in data.index:
            # Find nearest date
            try:
                nearest_idx = data.index.get_indexer([test_date_dt], method='nearest')[0]
                test_date_dt = data.index[nearest_idx]
            except (IndexError, ValueError):
                st.error(f"Could not find suitable date near {test_date}")
                return None, None
        
        # Get validation result
        validation = st.session_state.trader.validate_historical_prediction(
            symbol, test_date, weeks_ahead
        )
        
        if 'error' in validation:
            st.error(f"Validation error: {validation['error']}")
            return None, None
        
        # Create chart
        fig = go.Figure()
        
        # Historical data before test date
        test_idx = data.index.get_loc(test_date_dt)
        before_data = data.iloc[:test_idx+1]
        after_data = data.iloc[test_idx:]
        
        # Historical prices
        fig.add_trace(go.Scatter(
            x=before_data.index,
            y=before_data['Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue', width=2)
        ))
        
        # Actual future prices
        fig.add_trace(go.Scatter(
            x=after_data.index,
            y=after_data['Close'],
            mode='lines',
            name='Actual Price',
            line=dict(color='green', width=2)
        ))
        
        # Prediction point
        predicted_direction = validation['predicted_direction']
        test_price = validation['test_price']
        
        # Show prediction as an arrow or marker
        color = 'red' if predicted_direction == 'UP' else 'orange' if predicted_direction == 'DOWN' else 'gray'
        
        fig.add_trace(go.Scatter(
            x=[test_date_dt],
            y=[test_price],
            mode='markers',
            name=f'AI Predicted: {predicted_direction}',
            marker=dict(size=15, color=color, symbol='diamond')
        ))
        
        # Outcome point
        if 'outcome_price' in validation:
            outcome_date = test_date_dt + timedelta(weeks=weeks_ahead)
            outcome_price = validation['outcome_price']
            actual_direction = validation['actual_direction']
            
            outcome_color = 'green' if actual_direction == 'UP' else 'red' if actual_direction == 'DOWN' else 'gray'
            
            fig.add_trace(go.Scatter(
                x=[outcome_date],
                y=[outcome_price],
                mode='markers',
                name=f'Actual: {actual_direction}',
                marker=dict(size=15, color=outcome_color, symbol='star')
            ))
        
        # Add vertical line at test date
        fig.add_vline(
            x=test_date_dt,
            line_dash="dash",
            line_color="gray",
            annotation_text="Prediction Date"
        )
        
        # Result annotation
        correct = validation['prediction_correct']
        result_text = "✅ CORRECT" if correct else "❌ WRONG"
        
        fig.update_layout(
            title=f"{symbol} - Validation: {result_text} ({test_date})",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500
        )
        
        return fig, validation
        
    except Exception as e:
        st.error(f"Error creating validation chart: {e}")
        return None, None

# Load saved models
load_models()

# Main UI
st.title("🎯 NEAT Options Trading System")
st.markdown("**Highly Usable Interface for Training, Validation, and Predictions**")

# Sidebar for controls
st.sidebar.header("🔧 Controls")

# Multi-symbol training
st.sidebar.subheader("📚 Train New Models")
new_symbols = st.sidebar.text_input(
    "Enter symbols (comma-separated)",
    placeholder="AAPL, MSFT, GOOGL",
    help="Enter stock symbols separated by commas"
)

horizon_weeks = st.sidebar.selectbox(
    "Training Horizon (weeks)",
    [2, 4, 6, 8],
    index=1,
    help="How many weeks ahead to predict"
)

if st.sidebar.button("🚀 Train Models"):
    if new_symbols:
        symbols = [s.strip().upper() for s in new_symbols.split(',')]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Training {symbol}...")
            result = train_symbol(symbol, horizon_weeks)
            
            if result:
                st.success(f"✅ {symbol} trained successfully!")
            
            progress_bar.progress((i + 1) / len(symbols))
        
        status_text.text("Training complete!")
        st.rerun()

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predictions", "🧪 Validation", "📈 Statistics"])

with tab1:
    st.header("📊 Training Dashboard")
    
    if st.session_state.training_results:
        # Success rate overview
        results_df = pd.DataFrame(st.session_state.training_results).T
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_win_rate = results_df['win_rate'].mean()
            st.metric("Average Win Rate", f"{avg_win_rate:.1%}")
        
        with col2:
            profitable_models = len(results_df[results_df['total_return'] > 0])
            total_models = len(results_df)
            st.metric("Profitable Models", f"{profitable_models}/{total_models}")
        
        with col3:
            avg_trades = results_df['total_trades'].mean()
            st.metric("Avg Trades/Model", f"{avg_trades:.0f}")
        
        with col4:
            best_return = results_df['total_return'].max()
            st.metric("Best Return", f"{best_return:.1%}")
        
        # Detailed results table
        st.subheader("📋 Model Performance")
        
        display_df = results_df.copy()
        display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x:.1%}")
        display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.1%}")
        display_df['buy_hold_return'] = display_df['buy_hold_return'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            display_df[['total_return', 'win_rate', 'total_trades', 'buy_hold_return', 'trade_frequency']],
            use_container_width=True
        )
        
    else:
        st.info("No trained models yet. Use the sidebar to train some models!")

with tab2:
    st.header("🔮 AI Predictions (Next 1-8 Weeks)")
    
    if st.session_state.trained_models:
        # Symbol selection
        available_symbols = list(st.session_state.trained_models.keys())
        selected_symbol = st.selectbox("Select Symbol", available_symbols)
        
        prediction_weeks = st.slider("Prediction Horizon (weeks)", 1, 8, 4)
        
        if st.button("Generate Prediction Chart"):
            try:
                fig, predictions = create_prediction_chart(selected_symbol, prediction_weeks)
                
                if fig and predictions:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show prediction details
                    st.subheader("📋 Prediction Details")
                    pred_df = pd.DataFrame(predictions)
                    pred_df['predicted_price'] = pred_df['predicted_price'].apply(lambda x: f"${x:.2f}")
                    pred_df['confidence'] = pred_df['confidence'].apply(lambda x: f"{x:.2f}")
                    
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Current recommendation
                    try:
                        current_pred = st.session_state.trader.predict_next_weeks(selected_symbol, 4)
                        if 'error' not in current_pred:
                            st.success(f"**Recommendation**: {current_pred['recommendation']}")
                            st.info(f"**Confidence**: {current_pred['confidence']}")
                        else:
                            st.error(f"Recommendation error: {current_pred['error']}")
                    except Exception as e:
                        st.error(f"Error generating recommendation: {e}")
                else:
                    st.error("Could not generate prediction chart. Please check if the model is trained and data is available.")
            except Exception as e:
                st.error(f"Error generating prediction chart: {e}")
    else:
        st.info("No trained models available. Please train some models first.")

with tab3:
    st.header("🧪 Historical Validation")
    
    if st.session_state.trained_models:
        col1, col2 = st.columns(2)
        
        with col1:
            val_symbol = st.selectbox("Symbol", list(st.session_state.trained_models.keys()), key="val_symbol")
        
        with col2:
            val_weeks = st.selectbox("Weeks Ahead", [2, 4, 6, 8], index=1, key="val_weeks")
        
        # Preset interesting dates
        st.subheader("🎯 Test Interesting Market Periods")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📉 Before 2022 Correction"):
                test_date = "2022-01-01"
                try:
                    result = create_validation_chart(val_symbol, test_date, val_weeks)
                    if result and len(result) == 2:
                        fig, validation = result
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            if validation:
                                st.json(validation)
                        else:
                            st.error("Could not create validation chart")
                    else:
                        st.error("Validation chart returned invalid data")
                except Exception as e:
                    st.error(f"Error creating validation chart: {e}")
        
        with col2:
            if st.button("📈 Before 2023 Bull Run"):
                test_date = "2023-01-01"
                try:
                    result = create_validation_chart(val_symbol, test_date, val_weeks)
                    if result and len(result) == 2:
                        fig, validation = result
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            if validation:
                                st.json(validation)
                        else:
                            st.error("Could not create validation chart")
                    else:
                        st.error("Validation chart returned invalid data")
                except Exception as e:
                    st.error(f"Error creating validation chart: {e}")
        
        with col3:
            if st.button("🔄 Before 2024 Recovery"):
                test_date = "2024-01-01"
                try:
                    result = create_validation_chart(val_symbol, test_date, val_weeks)
                    if result and len(result) == 2:
                        fig, validation = result
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            if validation:
                                st.json(validation)
                        else:
                            st.error("Could not create validation chart")
                    else:
                        st.error("Validation chart returned invalid data")
                except Exception as e:
                    st.error(f"Error creating validation chart: {e}")
        
        # Custom date selection
        st.subheader("📅 Custom Date Testing")
        custom_date = st.date_input("Select Test Date", value=datetime(2024, 3, 1))
        
        if st.button("🧪 Run Custom Validation"):
            try:
                result = create_validation_chart(val_symbol, custom_date.strftime('%Y-%m-%d'), val_weeks)
                if result and len(result) == 2:
                    fig, validation = result
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        if validation:
                            st.json(validation)
                    else:
                        st.error("Could not create validation chart")
                else:
                    st.error("Validation chart returned invalid data")
            except Exception as e:
                st.error(f"Error creating custom validation chart: {e}")
    else:
        st.info("No trained models available for validation.")

with tab4:
    st.header("📈 Performance Statistics")
    
    if st.session_state.training_results:
        results_df = pd.DataFrame(st.session_state.training_results).T
        
        # Performance distribution
        fig_dist = px.histogram(
            results_df,
            x='total_return',
            title="Distribution of Model Returns",
            nbins=10
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Win rate vs return scatter with symbol labels
        results_df_reset = results_df.reset_index()
        results_df_reset['symbol'] = results_df_reset['index']
        
        # Create color and symbol mapping for accessibility
        symbols = results_df_reset['symbol'].unique()
        colors = px.colors.qualitative.Set1[:len(symbols)]
        symbol_shapes = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 
                        'triangle-down', 'star', 'hexagon'][:len(symbols)]
        
        fig_scatter = go.Figure()
        
        for i, symbol in enumerate(symbols):
            symbol_data = results_df_reset[results_df_reset['symbol'] == symbol]
            fig_scatter.add_trace(go.Scatter(
                x=symbol_data['win_rate'],
                y=symbol_data['total_return'],
                mode='markers',
                name=symbol,
                marker=dict(
                    size=12,
                    color=colors[i % len(colors)],
                    symbol=symbol_shapes[i % len(symbol_shapes)],
                    line=dict(width=2, color='black')
                ),
                hovertemplate=f'<b>{symbol}</b><br>' +
                             'Win Rate: %{x:.1%}<br>' +
                             'Total Return: %{y:.1%}<br>' +
                             '<extra></extra>'
            ))
        
        fig_scatter.update_layout(
            title="Win Rate vs Total Return by Symbol",
            xaxis_title="Win Rate",
            yaxis_title="Total Return",
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Trade frequency analysis
        st.subheader("📊 Trade Frequency Analysis")
        st.dataframe(
            results_df[['total_trades', 'trade_frequency', 'test_period_days']],
            use_container_width=True
        )
    else:
        st.info("No statistics available yet. Train some models first!")

# Footer
st.markdown("---")
st.markdown("**🎯 NEAT Options Trading System** - Practical AI for Weekly Options Trading")
