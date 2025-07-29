#!/usr/bin/env python3
"""
SwingTradeAI - Trade Management Dashboard
User Acceptance: User can quickly assess current risk levels
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.trading.trade_logger import TradeLogger, TradeEntry, TradeType, TradeStatus, TradeAction
from src.trading.risk_manager import RiskManager, RiskParameters
from src.trading.pnl_tracker import PnLTracker
from src.core.config import Settings

st.set_page_config(
    page_title="SwingTradeAI - Trade Management",
    page_icon="📊",
    layout="wide"
)

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_sample_current_prices():
    """Get sample current prices for demo"""
    return {
        'AAPL': 185.50,
        'MSFT': 378.25,
        'GOOGL': 142.80,
        'TSLA': 248.75,
        'NVDA': 875.30,
        'META': 325.60,
        'AMZN': 145.90
    }

def initialize_components():
    """Initialize trading components"""
    if 'trade_logger' not in st.session_state:
        st.session_state.trade_logger = TradeLogger()
    
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = RiskManager(st.session_state.trade_logger)
    
    if 'pnl_tracker' not in st.session_state:
        st.session_state.pnl_tracker = PnLTracker(
            st.session_state.trade_logger, 
            st.session_state.risk_manager
        )

def display_portfolio_overview():
    """Display portfolio overview with key metrics"""
    st.header("📊 Portfolio Overview")
    
    # Get current prices and update P&L
    current_prices = get_sample_current_prices()
    
    # Run async P&L update
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    pnl_data = loop.run_until_complete(
        st.session_state.pnl_tracker.update_real_time_pnl(current_prices)
    )
    loop.close()
    
    # Get performance metrics
    metrics = st.session_state.pnl_tracker.calculate_performance_metrics()
    portfolio_summary = st.session_state.risk_manager.get_portfolio_summary()
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_pnl = pnl_data.get('total_pnl', 0)
        st.metric(
            "💰 Total P&L", 
            f"${total_pnl:,.2f}",
            delta=f"{metrics.total_return_pct:.1f}%"
        )
    
    with col2:
        st.metric(
            "📈 Win Rate", 
            f"{metrics.win_rate:.1f}%",
            delta=f"{metrics.winning_trades}/{metrics.total_trades} trades"
        )
    
    with col3:
        current_capital = portfolio_summary.get('current_capital', 0)
        st.metric(
            "💵 Current Capital", 
            f"${current_capital:,.2f}",
            delta=f"${pnl_data.get('total_unrealized_pnl', 0):,.2f} unrealized"
        )
    
    with col4:
        exposure_pct = portfolio_summary.get('exposure_pct', 0)
        st.metric(
            "⚖️ Portfolio Exposure", 
            f"{exposure_pct:.1f}%",
            delta=f"{portfolio_summary.get('open_positions', 0)} positions"
        )
    
    with col5:
        if metrics.sharpe_ratio > 2.0:
            sharpe_color = "normal"
        elif metrics.sharpe_ratio > 1.0:
            sharpe_color = "normal"
        else:
            sharpe_color = "inverse"
        
        st.metric(
            "📊 Sharpe Ratio", 
            f"{metrics.sharpe_ratio:.2f}",
            delta=f"{metrics.max_drawdown_pct:.1f}% max DD",
            delta_color=sharpe_color
        )

def display_open_positions():
    """Display current open positions with real-time P&L"""
    st.header("📋 Open Positions")
    
    # Get open positions
    open_trades = st.session_state.trade_logger.get_trade_history(status=TradeStatus.OPEN)
    
    if open_trades.empty:
        st.info("💡 No open positions. Ready to trade!")
        return
    
    # Get current prices
    current_prices = get_sample_current_prices()
    
    # Calculate real-time metrics for each position
    position_data = []
    for _, trade in open_trades.iterrows():
        symbol = trade['symbol']
        current_price = current_prices.get(symbol, trade['entry_price'])
        
        # Calculate P&L
        if trade['action'] in ['buy_to_open', 'buy_to_close']:
            price_diff = current_price - trade['entry_price']
        else:
            price_diff = trade['entry_price'] - current_price
        
        unrealized_pnl = price_diff * trade['quantity']
        position_value = trade['entry_price'] * trade['quantity']
        return_pct = (unrealized_pnl / position_value) * 100 if position_value > 0 else 0
        
        # Days held
        entry_time = pd.to_datetime(trade['entry_timestamp'])
        days_held = (datetime.now() - entry_time).days
        
        position_data.append({
            'Symbol': symbol,
            'Type': trade['trade_type'].upper(),
            'Quantity': trade['quantity'],
            'Entry Price': f"${trade['entry_price']:.2f}",
            'Current Price': f"${current_price:.2f}",
            'Unrealized P&L': f"${unrealized_pnl:.2f}",
            'Return %': f"{return_pct:.1f}%",
            'Days Held': days_held,
            'Stop Loss': f"${trade['stop_loss']:.2f}" if trade['stop_loss'] else "None",
            'Take Profit': f"${trade['take_profit']:.2f}" if trade['take_profit'] else "None",
            'AI Confidence': f"{trade['ai_confidence']:.1%}",
            'Trade ID': trade['trade_id']
        })
    
    # Display positions table
    if position_data:
        df = pd.DataFrame(position_data)
        
        # Color code the P&L column
        def color_pnl(val):
            if 'P&L' in val:
                if val.startswith('$-'):
                    return 'color: red'
                elif val.startswith('$') and not val.startswith('$-'):
                    return 'color: green'
            return ''
        
        styled_df = df.style.applymap(color_pnl, subset=['Unrealized P&L'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Position actions
        st.subheader("⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Update All Prices"):
                st.rerun()
        
        with col2:
            selected_trade = st.selectbox("Select Trade to Close", 
                                        options=[f"{row['Symbol']} ({row['Trade ID'][:8]})" for row in position_data])
            if st.button("❌ Close Selected Trade"):
                st.info("💡 Trade closing functionality would be implemented here")
        
        with col3:
            if st.button("📊 Export Positions"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def display_risk_dashboard():
    """Display comprehensive risk dashboard"""
    st.header("🛡️ Risk Dashboard")
    
    # Get risk parameters and portfolio summary
    risk_params = st.session_state.risk_manager.risk_params
    portfolio_summary = st.session_state.risk_manager.get_portfolio_summary()
    
    # Risk utilization gauges
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Position count utilization
        current_positions = portfolio_summary.get('open_positions', 0)
        max_positions = risk_params.max_open_positions
        position_util = (current_positions / max_positions) * 100
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = position_util,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Position Utilization"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Exposure utilization
        exposure_pct = portfolio_summary.get('exposure_pct', 0)
        max_exposure = risk_params.max_total_exposure_pct
        exposure_util = (exposure_pct / max_exposure) * 100
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = exposure_util,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Exposure Utilization"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Daily trade utilization
        daily_trades = portfolio_summary.get('daily_trades', 0)
        max_daily = risk_params.max_daily_trades
        daily_util = (daily_trades / max_daily) * 100
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = daily_util,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Daily Trade Utilization"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkorange"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk parameters table
    st.subheader("🎛️ Current Risk Parameters")
    
    risk_data = {
        'Parameter': [
            'Max Position Size %',
            'Max Portfolio Risk %',
            'Max Total Exposure %',
            'Default Stop Loss %',
            'Default Take Profit %',
            'Max Open Positions',
            'Max Daily Trades',
            'Max Daily Loss %',
            'Max Weekly Loss %'
        ],
        'Current Value': [
            f"{risk_params.max_position_size_pct}%",
            f"{risk_params.max_portfolio_risk_pct}%",
            f"{risk_params.max_total_exposure_pct}%",
            f"{risk_params.default_stop_loss_pct}%",
            f"{risk_params.default_take_profit_pct}%",
            risk_params.max_open_positions,
            risk_params.max_daily_trades,
            f"{risk_params.max_daily_loss_pct}%",
            f"{risk_params.max_weekly_loss_pct}%"
        ],
        'Status': [
            '✅ Normal' if position_util < 80 else '⚠️ High',
            '✅ Normal',
            '✅ Normal' if exposure_util < 80 else '⚠️ High',
            '✅ Normal',
            '✅ Normal',
            '✅ Normal' if current_positions < max_positions else '⚠️ At Limit',
            '✅ Normal' if daily_trades < max_daily else '⚠️ At Limit',
            '✅ Normal',
            '✅ Normal'
        ]
    }
    
    st.dataframe(pd.DataFrame(risk_data), use_container_width=True)

def display_performance_analytics():
    """Display performance analytics and charts"""
    st.header("📈 Performance Analytics")
    
    # Get performance data
    metrics = st.session_state.pnl_tracker.calculate_performance_metrics()
    report = st.session_state.pnl_tracker.generate_performance_report()
    
    # Performance summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Return", f"{metrics.total_return_pct:.1f}%")
        st.metric("🎯 Profit Factor", f"{metrics.profit_factor:.2f}")
    
    with col2:
        st.metric("📈 Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
        st.metric("📉 Max Drawdown", f"{metrics.max_drawdown_pct:.1f}%")
    
    with col3:
        st.metric("🎲 Win Rate", f"{metrics.win_rate:.1f}%")
        st.metric("⏱️ Avg Hold Time", f"{metrics.avg_holding_period_hours:.1f}h")
    
    with col4:
        st.metric("💰 Daily P&L", f"${metrics.daily_pnl:.2f}")
        st.metric("📅 Weekly P&L", f"${metrics.weekly_pnl:.2f}")
    
    # Performance grades
    if report:
        st.subheader("🎓 Performance Grades")
        grade_col1, grade_col2, grade_col3, grade_col4 = st.columns(4)
        
        with grade_col1:
            return_grade = report.get('performance_summary', {}).get('return_grade', 'N/A')
            st.metric("Return Grade", return_grade)
        
        with grade_col2:
            win_rate_grade = report.get('trading_metrics', {}).get('win_rate_grade', 'N/A')
            st.metric("Win Rate Grade", win_rate_grade)
        
        with grade_col3:
            sharpe_grade = report.get('risk_metrics', {}).get('sharpe_grade', 'N/A')
            st.metric("Sharpe Grade", sharpe_grade)
        
        with grade_col4:
            drawdown_grade = report.get('risk_metrics', {}).get('drawdown_grade', 'N/A')
            st.metric("Drawdown Grade", drawdown_grade)
    
    # Recommendations
    if report and 'recommendations' in report:
        st.subheader("💡 AI Recommendations")
        for rec in report['recommendations']:
            st.info(rec)

def display_trade_entry_simulator():
    """Display trade entry simulator for testing risk management"""
    st.header("🧪 Trade Entry Simulator")
    st.markdown("**Test risk management with simulated trades**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Symbol", value="AAPL").upper()
        trade_type = st.selectbox("Trade Type", ["CALL", "PUT", "STOCK"])
        entry_price = st.number_input("Entry Price", value=185.50, min_value=0.01)
        quantity = st.number_input("Quantity", value=100, min_value=1)
    
    with col2:
        ai_confidence = st.slider("AI Confidence", 0.0, 1.0, 0.75, 0.05)
        days_to_exp = st.number_input("Days to Expiration", value=30, min_value=1) if trade_type != "STOCK" else None
        iv_percentile = st.slider("IV Percentile", 0.0, 100.0, 50.0) if trade_type != "STOCK" else None
    
    if st.button("🔍 Assess Trade Risk", type="primary"):
        # Simulate market conditions
        market_conditions = {
            'days_to_expiration': days_to_exp,
            'iv_percentile': iv_percentile,
            'volume': 50000  # Sample volume
        }
        
        # Assess risk
        risk_assessment = st.session_state.risk_manager.assess_trade_risk(
            symbol=symbol,
            trade_type=TradeType(trade_type.lower()),
            entry_price=entry_price,
            quantity=quantity,
            ai_confidence=ai_confidence,
            market_conditions=market_conditions
        )
        
        # Display results
        st.subheader("📊 Risk Assessment Results")
        
        # Approval status
        if risk_assessment.approved:
            st.success("✅ TRADE APPROVED")
        else:
            st.error("❌ TRADE REJECTED")
        
        # Risk metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{risk_assessment.risk_score:.1f}/100")
            st.metric("Recommended Size", f"{risk_assessment.position_size} shares")
        
        with col2:
            st.metric("Stop Loss", f"${risk_assessment.stop_loss_price:.2f}" if risk_assessment.stop_loss_price else "None")
            st.metric("Take Profit", f"${risk_assessment.take_profit_price:.2f}" if risk_assessment.take_profit_price else "None")
        
        with col3:
            st.metric("Max Loss", f"${risk_assessment.max_loss_amount:.2f}")
            st.metric("Risk/Reward", f"{risk_assessment.risk_reward_ratio:.2f}")
        
        # Warnings
        if risk_assessment.warnings:
            st.subheader("⚠️ Risk Warnings")
            for warning in risk_assessment.warnings:
                st.warning(warning)

def main():
    st.title("📊 SwingTradeAI - Trade Management Dashboard")
    st.markdown("**User Acceptance**: User can quickly assess current risk levels")
    
    # Initialize components
    initialize_components()
    
    # Sidebar navigation
    st.sidebar.header("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Select Dashboard",
        ["Portfolio Overview", "Open Positions", "Risk Dashboard", "Performance Analytics", "Trade Simulator"]
    )
    
    # Display selected page
    if page == "Portfolio Overview":
        display_portfolio_overview()
    elif page == "Open Positions":
        display_open_positions()
    elif page == "Risk Dashboard":
        display_risk_dashboard()
    elif page == "Performance Analytics":
        display_performance_analytics()
    elif page == "Trade Simulator":
        display_trade_entry_simulator()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    portfolio_summary = st.session_state.risk_manager.get_portfolio_summary()
    st.sidebar.metric("💰 Capital", f"${portfolio_summary.get('current_capital', 0):,.0f}")
    st.sidebar.metric("📈 Positions", portfolio_summary.get('open_positions', 0))
    st.sidebar.metric("⚖️ Exposure", f"{portfolio_summary.get('exposure_pct', 0):.1f}%")
    
    # Auto-refresh
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)")
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
