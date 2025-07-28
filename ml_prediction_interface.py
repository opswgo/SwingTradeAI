#!/usr/bin/env python3
"""
SwingTradeAI - ML Prediction Interface
User Acceptance: User sees clear confidence scores and regime explanations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import asyncio
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.core.config import Settings
from src.models.regime_model import RegimeModel
from src.features.feature_store import FeatureStore

st.set_page_config(
    page_title="SwingTradeAI - ML Predictions",
    page_icon="🤖",
    layout="wide"
)

async def get_prediction_with_explanation(symbol: str):
    """Get ML prediction with detailed explanation"""
    
    try:
        # Initialize components
        settings = Settings()
        regime_model = RegimeModel(settings)
        feature_store = FeatureStore(settings)
        
        await feature_store.initialize()
        
        # Get prediction
        st.write("🤖 **Generating ML Prediction...**")
        prediction = await regime_model.predict(symbol)
        
        # Get features for explanation
        st.write("🔍 **Analyzing Features...**")
        features_result = await feature_store.retrieve_features(symbol, limit=50)
        
        if not features_result['success']:
            st.error(f"❌ Could not retrieve features: {features_result.get('error', 'Unknown error')}")
            return None
        
        features_df = features_result['data']
        
        # Create comprehensive prediction result
        result = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'prediction': prediction,
            'features_used': len(features_df.columns) if not features_df.empty else 0,
            'data_points': len(features_df) if not features_df.empty else 0,
            'confidence_breakdown': {
                'regime_confidence': prediction.get('confidence', 0),
                'data_quality': 1.0 if not features_df.empty else 0.0,
                'feature_completeness': min(1.0, len(features_df.columns) / 20) if not features_df.empty else 0.0
            },
            'explanation': {
                'regime': prediction.get('regime', 'Unknown'),
                'reasoning': f"Based on analysis of {len(features_df)} data points and {len(features_df.columns) if not features_df.empty else 0} technical indicators",
                'key_factors': []
            }
        }
        
        # Add key factors based on available features
        if not features_df.empty and len(features_df) > 0:
            latest_data = features_df.iloc[-1]
            
            # Trend analysis
            if 'sma_20' in latest_data and 'sma_50' in latest_data:
                if latest_data['sma_20'] > latest_data['sma_50']:
                    result['explanation']['key_factors'].append("Short-term trend above medium-term (20-day > 50-day SMA)")
                else:
                    result['explanation']['key_factors'].append("Short-term trend below medium-term (20-day < 50-day SMA)")
            
            # Momentum analysis
            if 'rsi' in latest_data:
                rsi = latest_data['rsi']
                if rsi > 70:
                    result['explanation']['key_factors'].append(f"RSI indicates overbought conditions ({rsi:.1f})")
                elif rsi < 30:
                    result['explanation']['key_factors'].append(f"RSI indicates oversold conditions ({rsi:.1f})")
                else:
                    result['explanation']['key_factors'].append(f"RSI in neutral range ({rsi:.1f})")
            
            # Volatility analysis
            if 'bb_width' in latest_data:
                bb_width = latest_data['bb_width']
                if bb_width > 0.1:
                    result['explanation']['key_factors'].append("High volatility environment (wide Bollinger Bands)")
                else:
                    result['explanation']['key_factors'].append("Low volatility environment (narrow Bollinger Bands)")
        
        return result
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return None

def display_prediction_results(result):
    """Display prediction results with detailed explanations"""
    
    if not result:
        return
    
    prediction = result['prediction']
    
    # Main prediction display
    st.header(f"🎯 Prediction for {result['symbol']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        regime = prediction.get('regime', 'Unknown')
        st.metric("📈 Market Regime", regime.replace('_', ' ').title())
    
    with col2:
        confidence = prediction.get('confidence', 0)
        st.metric("🎯 Confidence", f"{confidence:.1%}")
    
    with col3:
        recommendation = "BULLISH" if "bull" in regime.lower() else "BEARISH" if "bear" in regime.lower() else "NEUTRAL"
        st.metric("💡 Signal", recommendation)
    
    # Confidence breakdown
    st.subheader("📊 Confidence Breakdown")
    
    conf_breakdown = result['confidence_breakdown']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🤖 Model Confidence", f"{conf_breakdown['regime_confidence']:.1%}")
    
    with col2:
        st.metric("📊 Data Quality", f"{conf_breakdown['data_quality']:.1%}")
    
    with col3:
        st.metric("🔧 Feature Completeness", f"{conf_breakdown['feature_completeness']:.1%}")
    
    # Overall confidence score
    overall_confidence = (
        conf_breakdown['regime_confidence'] * 0.5 +
        conf_breakdown['data_quality'] * 0.3 +
        conf_breakdown['feature_completeness'] * 0.2
    )
    
    if overall_confidence >= 0.8:
        st.success(f"🎉 High Confidence Prediction: {overall_confidence:.1%}")
    elif overall_confidence >= 0.6:
        st.info(f"✅ Moderate Confidence Prediction: {overall_confidence:.1%}")
    else:
        st.warning(f"⚠️ Low Confidence Prediction: {overall_confidence:.1%}")
    
    # Explanation
    st.subheader("💡 Prediction Explanation")
    
    explanation = result['explanation']
    st.write(f"**Reasoning**: {explanation['reasoning']}")
    
    if explanation['key_factors']:
        st.write("**Key Factors:**")
        for factor in explanation['key_factors']:
            st.write(f"• {factor}")
    
    # Data summary
    st.subheader("📊 Analysis Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("📈 Data Points Analyzed", result['data_points'])
    
    with col2:
        st.metric("🔧 Features Used", result['features_used'])
    
    # Recommendation
    st.subheader("🎯 Trading Recommendation")
    
    if overall_confidence >= 0.7:
        if "bull" in regime.lower():
            st.success("📈 **RECOMMENDATION**: Consider CALL options (1-6 week expiration)")
            st.info("💡 **Strategy**: Look for at-the-money or slightly out-of-the-money calls")
        elif "bear" in regime.lower():
            st.error("📉 **RECOMMENDATION**: Consider PUT options (1-6 week expiration)")
            st.info("💡 **Strategy**: Look for at-the-money or slightly out-of-the-money puts")
        else:
            st.warning("⚖️ **RECOMMENDATION**: NEUTRAL - Consider straddle or strangle strategies")
    else:
        st.warning("⚠️ **RECOMMENDATION**: LOW CONFIDENCE - Wait for clearer signals or reduce position size")
    
    # Risk warning
    st.warning("⚠️ **Risk Warning**: This is an AI prediction for educational purposes. Always do your own research and consider your risk tolerance before trading.")

def main():
    st.title("🤖 SwingTradeAI - ML Prediction Interface")
    st.markdown("**User Acceptance**: User sees clear confidence scores and regime explanations")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Prediction Controls")
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
    
    if st.sidebar.button("🚀 Get Prediction", type="primary"):
        st.header(f"🤖 Analyzing {symbol}")
        
        with st.spinner("Running ML prediction analysis..."):
            # Run async prediction
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(get_prediction_with_explanation(symbol))
            loop.close()
        
        if result:
            display_prediction_results(result)
            
            # Save prediction
            try:
                import json
                os.makedirs("data/predictions", exist_ok=True)
                prediction_file = f"data/predictions/{symbol}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(prediction_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                st.info(f"💾 Prediction saved to: {prediction_file}")
            except Exception as e:
                st.warning(f"⚠️ Could not save prediction: {str(e)}")
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Instructions")
    st.sidebar.markdown("""
    1. Enter a stock symbol (e.g., AAPL, MSFT)
    2. Click 'Get Prediction'
    3. Review confidence scores
    4. Understand the reasoning
    5. Consider the recommendation
    """)
    
    # Quick actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    if st.sidebar.button("📊 View Features"):
        st.info("💡 Run: `./run_dashboard.sh` to view technical features")
    
    if st.sidebar.button("📈 Run Backtest"):
        st.info("💡 Run: `./run_backtest.sh " + symbol + "` to backtest this symbol")
    
    # Model information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Information")
    st.sidebar.markdown("""
    **Model Type**: Regime Detection + Classification
    **Features**: 20+ technical indicators
    **Training**: Real market data only
    **Update Frequency**: Daily
    **Confidence Threshold**: 70% for trading signals
    """)

if __name__ == "__main__":
    main()
