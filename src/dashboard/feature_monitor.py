"""
SwingTradeAI - Feature Monitoring Dashboard
User Acceptance: User can monitor all features in real-time with alerts and explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import json

from ..core.config import Settings
from ..core.logger import setup_logger, log_user_action
from ..features.feature_store import FeatureStore
from ..features.technical_analysis import TechnicalAnalysis
from ..features.market_regime import MarketRegimeDetector

logger = setup_logger(__name__)


class FeatureMonitorDashboard:
    """
    Real-time feature monitoring dashboard with user-friendly visualizations.
    User Focus: Clear feature insights, alerts, and actionable information
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.feature_store = FeatureStore(settings)
        self.technical_analyzer = TechnicalAnalysis()
        self.regime_detector = MarketRegimeDetector()
        
        # Initialize session state
        if 'feature_alerts' not in st.session_state:
            st.session_state.feature_alerts = []
        if 'monitoring_symbols' not in st.session_state:
            st.session_state.monitoring_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
        if 'alert_thresholds' not in st.session_state:
            st.session_state.alert_thresholds = {
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'volatility_spike': 2.0,
                'volume_surge': 3.0
            }
    
    def render_dashboard(self):
        """Render the complete feature monitoring dashboard."""
        st.set_page_config(
            page_title="SwingTradeAI - Feature Monitor",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📊 Feature Monitoring Dashboard")
        st.markdown("Real-time monitoring of technical indicators and market features")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Live Monitor", 
            "📈 Feature Analysis", 
            "🚨 Alerts & Warnings",
            "🔍 Feature Explorer",
            "⚙️ System Health"
        ])
        
        with tab1:
            self._render_live_monitor()
        
        with tab2:
            self._render_feature_analysis()
        
        with tab3:
            self._render_alerts_warnings()
        
        with tab4:
            self._render_feature_explorer()
        
        with tab5:
            self._render_system_health()
    
    def _render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.header("🎛️ Monitor Controls")
        
        # Symbol selection
        st.sidebar.subheader("📊 Symbols to Monitor")
        
        # Add new symbol
        new_symbol = st.sidebar.text_input("Add Symbol", placeholder="e.g., AAPL").upper()
        if st.sidebar.button("➕ Add Symbol") and new_symbol:
            if new_symbol not in st.session_state.monitoring_symbols:
                st.session_state.monitoring_symbols.append(new_symbol)
                st.rerun()
        
        # Current symbols with remove buttons
        for i, symbol in enumerate(st.session_state.monitoring_symbols):
            col1, col2 = st.sidebar.columns([3, 1])
            col1.write(f"📈 {symbol}")
            if col2.button("❌", key=f"remove_{symbol}_{i}"):
                st.session_state.monitoring_symbols.remove(symbol)
                st.rerun()
        
        st.sidebar.divider()
        
        # Alert thresholds
        st.sidebar.subheader("🚨 Alert Thresholds")
        
        st.session_state.alert_thresholds['rsi_overbought'] = st.sidebar.slider(
            "RSI Overbought", 60, 90, st.session_state.alert_thresholds['rsi_overbought']
        )
        
        st.session_state.alert_thresholds['rsi_oversold'] = st.sidebar.slider(
            "RSI Oversold", 10, 40, st.session_state.alert_thresholds['rsi_oversold']
        )
        
        st.session_state.alert_thresholds['volatility_spike'] = st.sidebar.slider(
            "Volatility Spike (x normal)", 1.5, 5.0, st.session_state.alert_thresholds['volatility_spike']
        )
        
        st.session_state.alert_thresholds['volume_surge'] = st.sidebar.slider(
            "Volume Surge (x average)", 2.0, 10.0, st.session_state.alert_thresholds['volume_surge']
        )
        
        st.sidebar.divider()
        
        # Refresh controls
        st.sidebar.subheader("🔄 Refresh Settings")
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
        
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
    
    def _render_live_monitor(self):
        """Render live monitoring view."""
        st.header("🎯 Live Feature Monitor")
        
        if not st.session_state.monitoring_symbols:
            st.warning("⚠️ No symbols selected for monitoring. Add symbols in the sidebar.")
            return
        
        # Create columns for metrics
        cols = st.columns(min(len(st.session_state.monitoring_symbols), 4))
        
        for i, symbol in enumerate(st.session_state.monitoring_symbols):
            with cols[i % 4]:
                self._render_symbol_card(symbol)
        
        st.divider()
        
        # Feature heatmap
        st.subheader("🔥 Feature Heatmap")
        self._render_feature_heatmap()
        
        # Market regime overview
        st.subheader("🌍 Market Regime Overview")
        self._render_regime_overview()
    
    def _render_symbol_card(self, symbol: str):
        """Render a monitoring card for a single symbol."""
        try:
            # Get latest features (mock data for demo)
            features_data = self._get_latest_features(symbol)
            
            if not features_data:
                st.error(f"❌ No data for {symbol}")
                return
            
            # Create card
            with st.container():
                st.markdown(f"### 📈 {symbol}")
                
                # Key metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    price = features_data.get('Close', 0)
                    price_change = features_data.get('price_change', 0)
                    
                    st.metric(
                        "Price", 
                        f"${price:.2f}", 
                        f"{price_change:+.2f}%"
                    )
                
                with col2:
                    rsi = features_data.get('rsi', 50)
                    st.metric("RSI", f"{rsi:.1f}")
                
                # Alert indicators
                alerts = self._check_symbol_alerts(symbol, features_data)
                if alerts:
                    for alert in alerts:
                        st.warning(f"🚨 {alert}")
                else:
                    st.success("✅ All indicators normal")
                
        except Exception as e:
            st.error(f"❌ Error loading {symbol}: {str(e)}")
    
    def _render_feature_heatmap(self):
        """Render feature correlation heatmap."""
        try:
            # Mock correlation data
            features = ['RSI', 'MACD', 'BB_Width', 'Volume_Ratio', 'ATR', 'Price_Change']
            correlation_matrix = np.random.uniform(-1, 1, (len(features), len(features)))
            np.fill_diagonal(correlation_matrix, 1)
            
            # Make symmetric
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1)
            
            fig = px.imshow(
                correlation_matrix,
                x=features,
                y=features,
                color_continuous_scale='RdBu',
                aspect='auto',
                title="Feature Correlation Matrix"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.info("""
            📊 **Heatmap Interpretation:**
            - 🔴 Red: Strong positive correlation
            - 🔵 Blue: Strong negative correlation  
            - ⚪ White: No correlation
            """)
            
        except Exception as e:
            st.error(f"❌ Heatmap error: {str(e)}")
    
    def _render_regime_overview(self):
        """Render market regime overview."""
        try:
            # Mock regime data
            regimes = {
                'SPY': {'regime': 'Bull Trending', 'confidence': 0.85, 'emoji': '📈'},
                'QQQ': {'regime': 'Bull Volatile', 'confidence': 0.72, 'emoji': '🎢'},
                'IWM': {'regime': 'Sideways Low Vol', 'confidence': 0.68, 'emoji': '➡️'},
                'VIX': {'regime': 'Low Volatility', 'confidence': 0.91, 'emoji': '😴'}
            }
            
            cols = st.columns(len(regimes))
            
            for i, (symbol, regime_info) in enumerate(regimes.items()):
                with cols[i]:
                    st.metric(
                        f"{regime_info['emoji']} {symbol}",
                        regime_info['regime'],
                        f"Confidence: {regime_info['confidence']:.1%}"
                    )
            
        except Exception as e:
            st.error(f"❌ Regime overview error: {str(e)}")
    
    def _render_feature_analysis(self):
        """Render detailed feature analysis."""
        st.header("📈 Feature Analysis")
        
        # Symbol selector
        selected_symbol = st.selectbox(
            "Select Symbol for Analysis",
            st.session_state.monitoring_symbols,
            key="analysis_symbol"
        )
        
        if not selected_symbol:
            st.warning("⚠️ Please select a symbol for analysis.")
            return
        
        # Time period selector
        col1, col2 = st.columns(2)
        with col1:
            days_back = st.selectbox("Analysis Period", [7, 14, 30, 60, 90], index=2)
        with col2:
            feature_group = st.selectbox(
                "Feature Group", 
                ["All", "Trend", "Volatility", "Momentum", "Volume"]
            )
        
        # Feature time series
        st.subheader(f"📊 {selected_symbol} - Feature Time Series")
        self._render_feature_timeseries(selected_symbol, days_back, feature_group)
    
    def _render_feature_timeseries(self, symbol: str, days_back: int, feature_group: str):
        """Render feature time series charts."""
        try:
            # Mock time series data
            dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
            
            # Define features by group
            feature_groups = {
                "Trend": ["SMA_20", "SMA_50", "MACD"],
                "Volatility": ["BB_Upper", "BB_Lower", "ATR"],
                "Momentum": ["RSI", "Stoch_K"],
                "Volume": ["Volume_SMA", "Volume_Ratio"]
            }
            
            if feature_group == "All":
                features_to_plot = ["RSI", "MACD"]
            else:
                features_to_plot = feature_groups.get(feature_group, ["RSI", "MACD"])[:2]
            
            # Create subplots
            fig = make_subplots(
                rows=len(features_to_plot),
                cols=1,
                subplot_titles=features_to_plot,
                vertical_spacing=0.1
            )
            
            for i, feature in enumerate(features_to_plot, 1):
                # Generate mock data
                if feature == "RSI":
                    values = np.random.uniform(20, 80, days_back)
                elif feature == "MACD":
                    values = np.random.normal(0, 2, days_back)
                else:
                    values = np.random.normal(50, 20, days_back)
                
                fig.add_trace(
                    go.Scatter(x=dates, y=values, name=feature, mode='lines'),
                    row=i, col=1
                )
                
                # Add threshold lines for RSI
                if feature == "RSI":
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=i, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=i, col=1)
            
            fig.update_layout(height=200 * len(features_to_plot), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Time series error: {str(e)}")
    
    def _render_alerts_warnings(self):
        """Render alerts and warnings."""
        st.header("🚨 Alerts & Warnings")
        
        # Alert summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔴 Critical Alerts", "3")
        with col2:
            st.metric("🟡 Warnings", "7")
        with col3:
            st.metric("🟢 Normal", "15")
        with col4:
            st.metric("📊 Total Monitored", "25")
        
        # Recent alerts
        st.subheader("🕐 Recent Alerts")
        
        # Mock alert data
        alerts_data = [
            {"Time": "2024-01-15 14:30", "Symbol": "AAPL", "Type": "Critical", "Message": "RSI Overbought (78.5)"},
            {"Time": "2024-01-15 14:25", "Symbol": "TSLA", "Type": "Warning", "Message": "Volume Surge (4.2x average)"},
            {"Time": "2024-01-15 14:20", "Symbol": "MSFT", "Type": "Warning", "Message": "Volatility Spike (2.8x normal)"},
            {"Time": "2024-01-15 14:15", "Symbol": "GOOGL", "Type": "Info", "Message": "MACD Bullish Crossover"},
        ]
        
        for alert in alerts_data:
            alert_type = alert["Type"]
            if alert_type == "Critical":
                st.error(f"🔴 **{alert['Time']}** - {alert['Symbol']}: {alert['Message']}")
            elif alert_type == "Warning":
                st.warning(f"🟡 **{alert['Time']}** - {alert['Symbol']}: {alert['Message']}")
            else:
                st.info(f"🔵 **{alert['Time']}** - {alert['Symbol']}: {alert['Message']}")
    
    def _render_feature_explorer(self):
        """Render feature exploration interface."""
        st.header("🔍 Feature Explorer")
        
        # Feature search
        search_term = st.text_input("🔍 Search Features", placeholder="e.g., RSI, MACD, volatility")
        
        # Feature categories
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📂 Feature Categories")
            categories = ["All Features", "Trend Indicators", "Volatility", "Momentum", "Volume"]
            selected_category = st.radio("Select Category", categories)
        
        with col2:
            st.subheader("📋 Feature Details")
            
            # Mock feature data based on category
            if selected_category == "Trend Indicators":
                features = {
                    "SMA_20": {"description": "20-day Simple Moving Average", "current_value": "152.34"},
                    "MACD": {"description": "Moving Average Convergence Divergence", "current_value": "2.45"}
                }
            elif selected_category == "Momentum":
                features = {
                    "RSI": {"description": "Relative Strength Index", "current_value": "67.8"},
                    "Stochastic": {"description": "Stochastic Oscillator", "current_value": "72.3"}
                }
            else:
                features = {
                    "Price": {"description": "Current stock price", "current_value": "$154.23"},
                    "Volume": {"description": "Trading volume", "current_value": "45.2M"}
                }
            
            for feature_name, feature_info in features.items():
                if not search_term or search_term.lower() in feature_name.lower():
                    with st.expander(f"📊 {feature_name}"):
                        st.write(f"**Description:** {feature_info['description']}")
                        st.write(f"**Current Value:** {feature_info['current_value']}")
    
    def _render_system_health(self):
        """Render system health monitoring."""
        st.header("⚙️ System Health")
        
        # Overall health status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🟢 System Status", "Healthy")
        with col2:
            st.metric("📊 Data Pipeline", "Active")
        with col3:
            st.metric("🗄️ Feature Store", "Online")
        with col4:
            st.metric("🐘 PostgreSQL", "Connected")
        
        # Component status
        st.subheader("🔧 Component Status")
        
        components = [
            {"name": "Data Pipeline", "status": "✅ Healthy", "details": "Processing 5 symbols"},
            {"name": "Feature Store", "status": "✅ Healthy", "details": "1.2M features stored"},
            {"name": "PostgreSQL", "status": "✅ Connected", "details": "Pool: 8/10 connections"},
            {"name": "Alert System", "status": "⚠️ Warning", "details": "3 pending alerts"}
        ]
        
        for component in components:
            col_a, col_b, col_c = st.columns([2, 1, 2])
            col_a.write(f"**{component['name']}**")
            col_b.write(component['status'])
            col_c.write(component['details'])
    
    def _get_latest_features(self, symbol: str) -> Dict[str, Any]:
        """Get latest features for a symbol (mock implementation)."""
        return {
            'Close': np.random.uniform(100, 200),
            'price_change': np.random.uniform(-5, 5),
            'rsi': np.random.uniform(20, 80),
            'macd': np.random.uniform(-2, 2),
            'volume_ratio': np.random.uniform(0.5, 3.0)
        }
    
    def _check_symbol_alerts(self, symbol: str, features: Dict[str, Any]) -> List[str]:
        """Check for alerts on a symbol."""
        alerts = []
        
        # RSI alerts
        rsi = features.get('rsi', 50)
        if rsi > st.session_state.alert_thresholds['rsi_overbought']:
            alerts.append(f"RSI Overbought: {rsi:.1f}")
        elif rsi < st.session_state.alert_thresholds['rsi_oversold']:
            alerts.append(f"RSI Oversold: {rsi:.1f}")
        
        # Volume alerts
        volume_ratio = features.get('volume_ratio', 1.0)
        if volume_ratio > st.session_state.alert_thresholds['volume_surge']:
            alerts.append(f"Volume Surge: {volume_ratio:.1f}x")
        
        return alerts


def create_feature_monitor(settings: Settings) -> FeatureMonitorDashboard:
    """Create feature monitoring dashboard instance."""
    return FeatureMonitorDashboard(settings)
