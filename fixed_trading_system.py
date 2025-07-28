#!/usr/bin/env python3
"""
COMPREHENSIVE FIX for Trading System
Addresses ALL critical issues identified by user feedback.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os
import logging
from practical_neat_system import PracticalNEATTrader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database/persistence directory
MODELS_DIR = "/Users/wgo/Documents/Cline/SwingTradeAI/saved_models"
os.makedirs(MODELS_DIR, exist_ok=True)

class FixedTradingSystem:
    """Fixed trading system with proper persistence and error handling."""
    
    def __init__(self):
        self.trader = PracticalNEATTrader()
        self.load_persisted_models()
    
    def clean_symbol(self, symbol: str) -> str:
        """Clean and validate ticker symbol."""
        if not symbol:
            return ""
        
        # Remove any $ prefix or other invalid characters
        cleaned = symbol.strip().upper()
        cleaned = cleaned.replace('$', '')
        cleaned = cleaned.replace('.', '')
        cleaned = cleaned.replace(' ', '')
        
        # Validate symbol format (letters and numbers only)
        if not cleaned.isalnum():
            logger.warning(f"Invalid symbol format: {symbol} -> {cleaned}")
        
        return cleaned
    
    def fetch_data_safely(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Safely fetch data with proper error handling."""
        try:
            cleaned_symbol = self.clean_symbol(symbol)
            if not cleaned_symbol:
                raise ValueError(f"Invalid symbol: {symbol}")
            
            # Fix period format - use valid yfinance periods only
            valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
            if period not in valid_periods:
                logger.warning(f"Invalid period {period}, using 1y")
                period = "1y"
            
            logger.info(f"Fetching data for {cleaned_symbol} (period: {period})")
            ticker = yf.Ticker(cleaned_symbol)
            data = ticker.history(period=period, auto_adjust=True)
            
            if len(data) == 0:
                raise ValueError(f"No data available for {cleaned_symbol}")
            
            logger.info(f"✅ Fetched {len(data)} days of data for {cleaned_symbol}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Data fetch failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def save_model(self, symbol: str, model_data: dict):
        """Save model to persistent storage."""
        try:
            cleaned_symbol = self.clean_symbol(symbol)
            model_file = os.path.join(MODELS_DIR, f"{cleaned_symbol}_model.pkl")
            
            # Add timestamp and metadata
            model_data['saved_at'] = datetime.now().isoformat()
            model_data['symbol'] = cleaned_symbol
            
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"✅ Model saved for {cleaned_symbol}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model for {symbol}: {e}")
    
    def load_persisted_models(self):
        """Load all persisted models."""
        try:
            if not os.path.exists(MODELS_DIR):
                return
            
            loaded_count = 0
            for filename in os.listdir(MODELS_DIR):
                if filename.endswith('_model.pkl'):
                    try:
                        symbol = filename.replace('_model.pkl', '')
                        model_file = os.path.join(MODELS_DIR, filename)
                        
                        with open(model_file, 'rb') as f:
                            model_data = pickle.load(f)
                        
                        self.trader.models[symbol] = model_data
                        loaded_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to load model {filename}: {e}")
            
            if loaded_count > 0:
                logger.info(f"✅ Loaded {loaded_count} persisted models")
            
        except Exception as e:
            logger.error(f"❌ Failed to load persisted models: {e}")
    
    def train_model(self, symbol: str, horizon_weeks: int = 4) -> dict:
        """Train model with proper persistence."""
        try:
            cleaned_symbol = self.clean_symbol(symbol)
            if not cleaned_symbol:
                return {'success': False, 'error': f'Invalid symbol: {symbol}'}
            
            # Check if model already exists and is recent
            if cleaned_symbol in self.trader.models:
                model_data = self.trader.models[cleaned_symbol]
                saved_at = model_data.get('saved_at', '')
                if saved_at:
                    saved_time = datetime.fromisoformat(saved_at)
                    if datetime.now() - saved_time < timedelta(hours=24):
                        logger.info(f"⚡ Using existing model for {cleaned_symbol} (saved {saved_at})")
                        return {'success': True, 'symbol': cleaned_symbol, 'reused': True}
            
            # Fetch data
            data = self.fetch_data_safely(cleaned_symbol, "2y")
            if len(data) < 100:
                return {'success': False, 'error': f'Insufficient data for {cleaned_symbol}'}
            
            # Train model
            result = self.trader.train_weekly_agent(cleaned_symbol, data, horizon_weeks)
            
            if result.get('success', False):
                # Save to persistent storage
                self.save_model(cleaned_symbol, self.trader.models[cleaned_symbol])
                logger.info(f"✅ Training completed for {cleaned_symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Training failed for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, symbol: str, weeks_ahead: int = 4) -> dict:
        """Make prediction with proper error handling."""
        try:
            cleaned_symbol = self.clean_symbol(symbol)
            if not cleaned_symbol:
                return {'error': f'Invalid symbol: {symbol}'}
            
            if cleaned_symbol not in self.trader.models:
                return {'error': f'No trained model for {cleaned_symbol}. Please train first.'}
            
            return self.trader.predict_next_weeks(cleaned_symbol, weeks_ahead)
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def validate_historical(self, symbol: str, test_date: str, weeks_ahead: int = 4) -> dict:
        """Historical validation with proper error handling."""
        try:
            cleaned_symbol = self.clean_symbol(symbol)
            if not cleaned_symbol:
                return {'error': f'Invalid symbol: {symbol}'}
            
            if cleaned_symbol not in self.trader.models:
                return {'error': f'No trained model for {cleaned_symbol}. Please train first.'}
            
            return self.trader.validate_historical_prediction(cleaned_symbol, test_date, weeks_ahead)
            
        except Exception as e:
            logger.error(f"❌ Validation failed for {symbol}: {e}")
            return {'error': str(e)}
    
    def get_model_info(self, symbol: str) -> dict:
        """Get detailed model information."""
        try:
            cleaned_symbol = self.clean_symbol(symbol)
            if cleaned_symbol not in self.trader.models:
                return {}
            
            model_data = self.trader.models[cleaned_symbol]
            return {
                'symbol': cleaned_symbol,
                'horizon_weeks': model_data.get('horizon_weeks', 4),
                'training_period': model_data.get('training_period', 'Unknown'),
                'test_period': model_data.get('test_period', 'Unknown'),
                'saved_at': model_data.get('saved_at', 'Unknown'),
                'performance': model_data.get('performance', {}),
                'trained_date': model_data.get('trained_date', 'Unknown')
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get model info for {symbol}: {e}")
            return {}

# Streamlit UI
def main():
    st.set_page_config(page_title="Fixed Trading System", layout="wide")
    
    st.title("🎯 Fixed Trading System")
    st.markdown("**All critical issues addressed:** Model persistence, data fetching, symbol validation, error handling")
    
    # Initialize system
    if 'trading_system' not in st.session_state:
        st.session_state.trading_system = FixedTradingSystem()
    
    system = st.session_state.trading_system
    
    # Sidebar for model management
    st.sidebar.header("📊 Model Management")
    
    # Show available models
    available_models = list(system.trader.models.keys())
    if available_models:
        st.sidebar.success(f"✅ {len(available_models)} models loaded")
        for symbol in available_models:
            model_info = system.get_model_info(symbol)
            with st.sidebar.expander(f"📈 {symbol}"):
                st.write(f"**Horizon:** {model_info.get('horizon_weeks', 'Unknown')} weeks")
                st.write(f"**Saved:** {model_info.get('saved_at', 'Unknown')[:16]}")
                perf = model_info.get('performance', {})
                if perf:
                    st.write(f"**Win Rate:** {perf.get('win_rate', 0):.1%}")
                    st.write(f"**Total Return:** {perf.get('total_return', 0):.1%}")
    else:
        st.sidebar.warning("No models available. Train some models first.")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏋️ Training", "🔮 Predictions", "🧪 Validation", "📊 Performance"])
    
    with tab1:
        st.header("🏋️ Model Training")
        
        col1, col2 = st.columns(2)
        with col1:
            train_symbol = st.text_input("Symbol to Train", "AAPL").upper()
        with col2:
            horizon_weeks = st.selectbox("Horizon (weeks)", [1, 2, 4, 8], index=2)
        
        if st.button("🚀 Train Model", type="primary"):
            if train_symbol:
                with st.spinner(f"Training {train_symbol}..."):
                    result = system.train_model(train_symbol, horizon_weeks)
                
                if result.get('success', False):
                    if result.get('reused', False):
                        st.info(f"⚡ Using existing model for {train_symbol}")
                    else:
                        st.success(f"✅ Model trained successfully for {train_symbol}")
                    
                    # Show model info
                    model_info = system.get_model_info(train_symbol)
                    if model_info:
                        st.json(model_info)
                else:
                    st.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
    
    with tab2:
        st.header("🔮 Predictions")
        
        if available_models:
            pred_symbol = st.selectbox("Select Symbol", available_models)
            pred_weeks = st.slider("Prediction Horizon (weeks)", 1, 8, 4)
            
            if st.button("🔮 Generate Prediction"):
                with st.spinner("Generating prediction..."):
                    prediction = system.predict(pred_symbol, pred_weeks)
                
                if 'error' in prediction:
                    st.error(f"❌ {prediction['error']}")
                else:
                    st.success(f"✅ Prediction for {pred_symbol}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${prediction.get('current_price', 0):.2f}")
                    with col2:
                        st.metric("Confidence", prediction.get('confidence', 'Unknown'))
                    with col3:
                        st.metric("Signal", f"{prediction.get('prediction_signal', 0):.3f}")
                    
                    st.info(f"**Recommendation:** {prediction.get('recommendation', 'None')}")
        else:
            st.warning("No trained models available. Please train some models first.")
    
    with tab3:
        st.header("🧪 Historical Validation")
        
        if available_models:
            val_symbol = st.selectbox("Select Symbol", available_models, key="val_symbol")
            val_date = st.date_input("Test Date", datetime(2023, 1, 1))
            val_weeks = st.slider("Validation Horizon (weeks)", 1, 8, 4, key="val_weeks")
            
            if st.button("🧪 Run Validation"):
                with st.spinner("Running validation..."):
                    validation = system.validate_historical(val_symbol, val_date.strftime('%Y-%m-%d'), val_weeks)
                
                if 'error' in validation:
                    st.error(f"❌ {validation['error']}")
                else:
                    st.success(f"✅ Validation for {val_symbol}")
                    st.json(validation)
        else:
            st.warning("No trained models available. Please train some models first.")
    
    with tab4:
        st.header("📊 Performance Statistics")
        
        if available_models:
            for symbol in available_models:
                model_info = system.get_model_info(symbol)
                perf = model_info.get('performance', {})
                
                if perf:
                    with st.expander(f"📈 {symbol} Performance"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Win Rate", f"{perf.get('win_rate', 0):.1%}")
                        with col2:
                            st.metric("Total Return", f"{perf.get('total_return', 0):.1%}")
                        with col3:
                            st.metric("Total Trades", f"{perf.get('total_trades', 0)}")
                        with col4:
                            st.metric("Buy & Hold", f"{perf.get('buy_hold_return', 0):.1%}")
                        
                        # Remove useless timeframe info as requested
                        st.write(f"**Training Period:** {model_info.get('training_period', 'Unknown')}")
                        st.write(f"**Test Period:** {model_info.get('test_period', 'Unknown')}")
        else:
            st.warning("No performance data available. Please train some models first.")

if __name__ == "__main__":
    main()
