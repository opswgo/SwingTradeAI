"""
Performance Attribution Analyzer for NEAT Options System
Detailed analysis of strategy performance attribution and factor decomposition.
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neat_options_advisor import neat_advisor
from neat_framework import neat_evolution

def create_performance_attribution_interface():
    """Create comprehensive performance attribution analysis interface."""
    
    st.header("📈 Performance Attribution Analyzer")
    st.markdown("Deep dive into strategy performance drivers and factor attribution")
    
    # Sidebar for analysis parameters
    st.sidebar.header("🔧 Analysis Parameters")
    
    # Time period selection
    st.sidebar.subheader("📅 Analysis Period")
    
    analysis_period = st.sidebar.selectbox(
        "Period",
        options=["Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year", "Custom"],
        index=2
    )
    
    if analysis_period == "Custom":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.sidebar.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=180)
            )
        with col2:
            end_date = st.sidebar.date_input(
                "End Date",
                value=datetime.now()
            )
    else:
        days_map = {
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "Last 6 Months": 180,
            "Last Year": 365
        }
        days = days_map[analysis_period]
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
    
    # Attribution method
    st.sidebar.subheader("📊 Attribution Method")
    
    attribution_method = st.sidebar.selectbox(
        "Method",
        options=["Factor-Based", "Sector-Based", "Time-Based", "Risk-Based"],
        index=0
    )
    
    # Benchmark selection
    benchmark = st.sidebar.selectbox(
        "Benchmark",
        options=["SPY", "QQQ", "IWM", "VTI", "Buy & Hold"],
        index=0
    )
    
    # Analysis granularity
    granularity = st.sidebar.selectbox(
        "Granularity",
        options=["Daily", "Weekly", "Monthly"],
        index=1
    )
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Factor Attribution", 
        "🏭 Sector Attribution", 
        "📅 Time Attribution", 
        "⚖️ Risk Attribution"
    ])
    
    # Generate sample data for demonstration
    sample_data = generate_sample_attribution_data(start_date, end_date)
    
    with tab1:
        display_factor_attribution(sample_data, benchmark)
    
    with tab2:
        display_sector_attribution(sample_data)
    
    with tab3:
        display_time_attribution(sample_data, granularity)
    
    with tab4:
        display_risk_attribution(sample_data, benchmark)

def generate_sample_attribution_data(start_date, end_date):
    """Generate sample attribution data for analysis."""
    
    # Create date range
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Factor returns (simplified simulation)
    np.random.seed(42)  # For reproducible results
    
    factors = {
        'Market': np.random.normal(0.0005, 0.015, len(dates)),
        'Value': np.random.normal(0.0002, 0.008, len(dates)),
        'Growth': np.random.normal(0.0003, 0.012, len(dates)),
        'Momentum': np.random.normal(0.0001, 0.010, len(dates)),
        'Quality': np.random.normal(0.0002, 0.006, len(dates)),
        'Volatility': np.random.normal(-0.0001, 0.020, len(dates)),
        'Size': np.random.normal(0.0001, 0.009, len(dates))
    }
    
    # Strategy exposures to factors
    exposures = {
        'Market': 0.8,
        'Value': 0.3,
        'Growth': -0.1,
        'Momentum': 0.4,
        'Quality': 0.6,
        'Volatility': -0.5,
        'Size': 0.2
    }
    
    # Calculate strategy returns based on factor exposures
    strategy_returns = np.zeros(len(dates))
    factor_contributions = {}
    
    for factor, returns in factors.items():
        contribution = returns * exposures[factor]
        factor_contributions[factor] = contribution
        strategy_returns += contribution
    
    # Add alpha (strategy-specific return)
    alpha = np.random.normal(0.0003, 0.005, len(dates))
    strategy_returns += alpha
    factor_contributions['Alpha'] = alpha
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'strategy_return': strategy_returns,
        'cumulative_return': (1 + pd.Series(strategy_returns)).cumprod() - 1
    })
    
    # Add factor contributions
    for factor, contribution in factor_contributions.items():
        data[f'{factor}_contribution'] = contribution
        data[f'{factor}_cumulative'] = (1 + pd.Series(contribution)).cumprod() - 1
    
    # Add sector data
    sectors = ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial', 'Energy']
    sector_weights = np.random.dirichlet(np.ones(len(sectors)), len(dates))
    sector_returns = np.random.normal(0.0005, 0.015, (len(dates), len(sectors)))
    
    for i, sector in enumerate(sectors):
        data[f'{sector}_weight'] = sector_weights[:, i]
        data[f'{sector}_return'] = sector_returns[:, i]
        data[f'{sector}_contribution'] = sector_weights[:, i] * sector_returns[:, i]
    
    # Add risk metrics
    data['volatility'] = pd.Series(strategy_returns).rolling(30).std() * np.sqrt(252)
    data['sharpe_ratio'] = (pd.Series(strategy_returns).rolling(30).mean() * 252) / data['volatility']
    data['max_drawdown'] = calculate_rolling_max_drawdown(data['cumulative_return'], 30)
    
    return data

def calculate_rolling_max_drawdown(returns, window):
    """Calculate rolling maximum drawdown."""
    rolling_max = returns.rolling(window).max()
    drawdown = (returns - rolling_max) / rolling_max
    return drawdown.rolling(window).min().abs()

def display_factor_attribution(data, benchmark):
    """Display factor-based attribution analysis."""
    
    st.subheader("🔍 Factor Attribution Analysis")
    
    # Factor contribution summary
    factors = ['Market', 'Value', 'Growth', 'Momentum', 'Quality', 'Volatility', 'Size', 'Alpha']
    
    # Calculate total contributions
    factor_contributions = {}
    for factor in factors:
        total_contribution = data[f'{factor}_cumulative'].iloc[-1] if len(data) > 0 else 0
        factor_contributions[factor] = total_contribution
    
    # Create contribution chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Waterfall chart showing factor contributions
        fig = go.Figure()
        
        x_labels = list(factor_contributions.keys())
        y_values = list(factor_contributions.values())
        
        # Create waterfall chart
        cumulative = 0
        for i, (factor, contribution) in enumerate(factor_contributions.items()):
            if factor == 'Alpha':
                color = 'gold'
            elif contribution > 0:
                color = 'green'
            else:
                color = 'red'
            
            fig.add_trace(go.Bar(
                x=[factor],
                y=[contribution],
                name=factor,
                marker_color=color,
                text=f"{contribution:.1%}",
                textposition="outside"
            ))
        
        fig.update_layout(
            title="Factor Contribution to Total Return",
            xaxis_title="Factors",
            yaxis_title="Contribution (%)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Factor Summary")
        
        # Top contributing factors
        sorted_factors = sorted(factor_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for factor, contribution in sorted_factors[:5]:
            delta_color = "normal" if contribution >= 0 else "inverse"
            st.metric(
                factor,
                f"{contribution:.1%}",
                delta=f"{contribution:.1%}",
                delta_color=delta_color
            )
    
    # Factor exposure over time
    st.subheader("📈 Factor Exposure Over Time")
    
    # Create time series chart of factor contributions
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Cumulative Factor Contributions", "Daily Factor Contributions"),
        vertical_spacing=0.1
    )
    
    # Cumulative contributions
    for factor in factors[:4]:  # Show top 4 factors
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data[f'{factor}_cumulative'] * 100,
                mode='lines',
                name=f'{factor} (Cumulative)',
                line=dict(width=2)
            ),
            row=1, col=1
        )
    
    # Daily contributions
    for factor in factors[:4]:
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data[f'{factor}_contribution'] * 100,
                mode='lines',
                name=f'{factor} (Daily)',
                line=dict(width=1),
                opacity=0.7,
                showlegend=False
            ),
            row=2, col=1
        )
    
    fig.update_layout(height=600, title_text="Factor Attribution Over Time")
    fig.update_yaxes(title_text="Contribution (%)", row=1, col=1)
    fig.update_yaxes(title_text="Daily Contribution (%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Factor correlation matrix
    st.subheader("🔗 Factor Correlation Matrix")
    
    factor_data = data[[f'{factor}_contribution' for factor in factors]].copy()
    factor_data.columns = factors
    
    correlation_matrix = factor_data.corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Factor Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_sector_attribution(data):
    """Display sector-based attribution analysis."""
    
    st.subheader("🏭 Sector Attribution Analysis")
    
    sectors = ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial', 'Energy']
    
    # Current sector allocation
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Current Sector Allocation")
        
        if len(data) > 0:
            current_weights = {sector: data[f'{sector}_weight'].iloc[-1] for sector in sectors}
            
            fig = px.pie(
                values=list(current_weights.values()),
                names=list(current_weights.keys()),
                title="Portfolio Sector Weights"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Sector Contribution")
        
        if len(data) > 0:
            sector_contributions = {}
            for sector in sectors:
                contribution = data[f'{sector}_contribution'].sum()
                sector_contributions[sector] = contribution
            
            sorted_sectors = sorted(sector_contributions.items(), key=lambda x: x[1], reverse=True)
            
            for sector, contribution in sorted_sectors:
                delta_color = "normal" if contribution >= 0 else "inverse"
                st.metric(
                    sector,
                    f"{contribution:.2%}",
                    delta=f"{contribution:.2%}",
                    delta_color=delta_color
                )
    
    # Sector performance over time
    st.subheader("📈 Sector Performance Over Time")
    
    fig = go.Figure()
    
    for sector in sectors:
        if len(data) > 0:
            cumulative_contribution = data[f'{sector}_contribution'].cumsum()
            
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=cumulative_contribution * 100,
                mode='lines',
                name=sector,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title="Cumulative Sector Contributions",
        xaxis_title="Date",
        yaxis_title="Cumulative Contribution (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector allocation over time
    st.subheader("⚖️ Sector Allocation Over Time")
    
    if len(data) > 0:
        sector_weights_df = data[['date'] + [f'{sector}_weight' for sector in sectors]].copy()
        sector_weights_df.columns = ['date'] + sectors
        
        fig = go.Figure()
        
        for sector in sectors:
            fig.add_trace(go.Scatter(
                x=sector_weights_df['date'],
                y=sector_weights_df[sector] * 100,
                mode='lines',
                stackgroup='one',
                name=sector
            ))
        
        fig.update_layout(
            title="Sector Allocation Over Time",
            xaxis_title="Date",
            yaxis_title="Allocation (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_time_attribution(data, granularity):
    """Display time-based attribution analysis."""
    
    st.subheader("📅 Time-Based Attribution Analysis")
    
    if len(data) == 0:
        st.warning("No data available for time attribution analysis")
        return
    
    # Aggregate data based on granularity
    if granularity == "Daily":
        time_data = data.copy()
        time_data['period'] = time_data['date'].dt.strftime('%Y-%m-%d')
    elif granularity == "Weekly":
        time_data = data.copy()
        time_data['period'] = time_data['date'].dt.to_period('W').astype(str)
        time_data = time_data.groupby('period').agg({
            'strategy_return': 'sum',
            'volatility': 'mean',
            'sharpe_ratio': 'mean'
        }).reset_index()
    else:  # Monthly
        time_data = data.copy()
        time_data['period'] = time_data['date'].dt.to_period('M').astype(str)
        time_data = time_data.groupby('period').agg({
            'strategy_return': 'sum',
            'volatility': 'mean',
            'sharpe_ratio': 'mean'
        }).reset_index()
    
    # Performance by time period
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Return distribution by period
        fig = px.bar(
            time_data,
            x='period',
            y='strategy_return',
            title=f"{granularity} Returns",
            color='strategy_return',
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Period Statistics")
        
        if len(time_data) > 0:
            avg_return = time_data['strategy_return'].mean()
            std_return = time_data['strategy_return'].std()
            best_period = time_data.loc[time_data['strategy_return'].idxmax(), 'period']
            worst_period = time_data.loc[time_data['strategy_return'].idxmin(), 'period']
            
            st.metric("Average Return", f"{avg_return:.2%}")
            st.metric("Return Volatility", f"{std_return:.2%}")
            st.metric("Best Period", best_period)
            st.metric("Worst Period", worst_period)
    
    # Rolling performance metrics
    st.subheader("📈 Rolling Performance Metrics")
    
    if len(data) > 30:  # Need sufficient data for rolling metrics
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Rolling Returns", "Rolling Volatility", "Rolling Sharpe Ratio"),
            vertical_spacing=0.08
        )
        
        # Rolling returns (30-day)
        rolling_returns = data['strategy_return'].rolling(30).sum()
        fig.add_trace(
            go.Scatter(x=data['date'], y=rolling_returns * 100, name="30-Day Returns"),
            row=1, col=1
        )
        
        # Rolling volatility
        fig.add_trace(
            go.Scatter(x=data['date'], y=data['volatility'] * 100, name="Volatility"),
            row=2, col=1
        )
        
        # Rolling Sharpe ratio
        fig.add_trace(
            go.Scatter(x=data['date'], y=data['sharpe_ratio'], name="Sharpe Ratio"),
            row=3, col=1
        )
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

def display_risk_attribution(data, benchmark):
    """Display risk-based attribution analysis."""
    
    st.subheader("⚖️ Risk Attribution Analysis")
    
    if len(data) == 0:
        st.warning("No data available for risk attribution analysis")
        return
    
    # Risk metrics summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = data['cumulative_return'].iloc[-1] if len(data) > 0 else 0
        st.metric("Total Return", f"{total_return:.1%}")
    
    with col2:
        volatility = data['volatility'].iloc[-1] if len(data) > 0 else 0
        st.metric("Volatility", f"{volatility:.1%}")
    
    with col3:
        sharpe_ratio = data['sharpe_ratio'].iloc[-1] if len(data) > 0 else 0
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with col4:
        max_drawdown = data['max_drawdown'].max() if len(data) > 0 else 0
        st.metric("Max Drawdown", f"{max_drawdown:.1%}")
    
    # Risk decomposition
    st.subheader("🔍 Risk Decomposition")
    
    # Calculate risk contributions (simplified)
    risk_factors = {
        'Market Risk': 0.60,
        'Specific Risk': 0.25,
        'Model Risk': 0.10,
        'Execution Risk': 0.05
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Risk contribution pie chart
        fig = px.pie(
            values=list(risk_factors.values()),
            names=list(risk_factors.keys()),
            title="Risk Contribution Breakdown"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk metrics table
        st.subheader("📊 Risk Metrics")
        
        risk_metrics = pd.DataFrame([
            ["Value at Risk (95%)", f"{np.percentile(data['strategy_return'], 5) * 100:.2f}%"],
            ["Expected Shortfall", f"{data['strategy_return'][data['strategy_return'] <= np.percentile(data['strategy_return'], 5)].mean() * 100:.2f}%"],
            ["Skewness", f"{data['strategy_return'].skew():.2f}"],
            ["Kurtosis", f"{data['strategy_return'].kurtosis():.2f}"],
            ["Downside Deviation", f"{data['strategy_return'][data['strategy_return'] < 0].std() * np.sqrt(252) * 100:.1f}%"]
        ], columns=["Metric", "Value"])
        
        st.dataframe(risk_metrics, use_container_width=True, hide_index=True)
    
    # Risk over time
    st.subheader("📈 Risk Metrics Over Time")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Volatility and Drawdown", "Risk-Adjusted Returns"),
        vertical_spacing=0.1
    )
    
    # Volatility and drawdown
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['volatility'] * 100, name="Volatility", line=dict(color='red')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['max_drawdown'] * 100, name="Max Drawdown", line=dict(color='orange')),
        row=1, col=1
    )
    
    # Sharpe ratio
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['sharpe_ratio'], name="Sharpe Ratio", line=dict(color='blue')),
        row=2, col=1
    )
    
    fig.update_layout(height=500)
    fig.update_yaxes(title_text="Percentage (%)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Performance Attribution Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    create_performance_attribution_interface()
