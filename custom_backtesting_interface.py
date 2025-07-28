"""
Custom Backtesting Interface for NEAT Options System
Advanced backtesting with custom parameters and detailed analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neat_options_advisor import neat_advisor
from neat_framework import neat_evolution

def create_custom_backtesting_interface():
    """Create comprehensive custom backtesting interface."""
    
    st.header("🔬 Custom Backtesting Laboratory")
    st.markdown("Advanced backtesting with custom parameters for NEAT options strategies")
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Backtesting Parameters")
    
    # Time period selection
    st.sidebar.subheader("📅 Time Period")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=730),  # 2 years ago
            max_value=datetime.now() - timedelta(days=30)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now() - timedelta(days=30),  # 1 month ago
            max_value=datetime.now()
        )
    
    # Symbol selection
    st.sidebar.subheader("📊 Symbols")
    
    symbol_selection = st.sidebar.radio(
        "Symbol Selection",
        options=["Single Symbol", "Multiple Symbols", "Portfolio"],
        index=1
    )
    
    if symbol_selection == "Single Symbol":
        symbols = [st.sidebar.text_input("Symbol", value="AAPL").upper()]
    elif symbol_selection == "Multiple Symbols":
        symbols_input = st.sidebar.text_area(
            "Symbols (comma-separated)",
            value="AAPL,MSFT,GOOGL,TSLA,NVDA"
        )
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    else:  # Portfolio
        symbols = neat_advisor.sp500_symbols[:20]  # Top 20 for performance
    
    # Strategy parameters
    st.sidebar.subheader("🎯 Strategy Parameters")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="Minimum AI confidence to enter trade"
    )
    
    min_premium_yield = st.sidebar.slider(
        "Min Premium Yield (%/week)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1,
        help="Minimum weekly premium yield target"
    ) / 100
    
    max_assignment_rate = st.sidebar.slider(
        "Max Assignment Rate (%)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Maximum acceptable assignment probability"
    ) / 100
    
    position_size_method = st.sidebar.selectbox(
        "Position Sizing",
        options=["Fixed Dollar", "Fixed Percentage", "Kelly Criterion", "Risk Parity"],
        index=1
    )
    
    if position_size_method == "Fixed Dollar":
        position_size = st.sidebar.number_input(
            "Position Size ($)",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000
        )
    else:
        position_size = st.sidebar.slider(
            "Position Size (%)",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        ) / 100
    
    # Risk management
    st.sidebar.subheader("⚖️ Risk Management")
    
    stop_loss_enabled = st.sidebar.checkbox("Enable Stop Loss", value=True)
    if stop_loss_enabled:
        stop_loss_pct = st.sidebar.slider(
            "Stop Loss (%)",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            help="Stop loss as percentage of premium received"
        ) / 100
    
    early_close_enabled = st.sidebar.checkbox("Enable Early Close", value=True)
    if early_close_enabled:
        early_close_profit = st.sidebar.slider(
            "Early Close Profit (%)",
            min_value=25,
            max_value=90,
            value=50,
            step=5,
            help="Close position when this % of max profit is achieved"
        ) / 100
    
    # Advanced options
    st.sidebar.subheader("🔧 Advanced Options")
    
    transaction_costs = st.sidebar.number_input(
        "Transaction Costs ($)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Cost per option contract"
    )
    
    slippage_pct = st.sidebar.slider(
        "Slippage (%)",
        min_value=0.0,
        max_value=2.0,
        value=0.1,
        step=0.1,
        help="Slippage as percentage of bid-ask spread"
    ) / 100
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🚀 Run Custom Backtest")
        
        if st.button("🔬 Execute Backtest", type="primary"):
            with st.spinner("Running comprehensive backtest..."):
                try:
                    # Prepare backtest parameters
                    backtest_params = {
                        'start_date': start_date,
                        'end_date': end_date,
                        'symbols': symbols,
                        'confidence_threshold': confidence_threshold,
                        'min_premium_yield': min_premium_yield,
                        'max_assignment_rate': max_assignment_rate,
                        'position_size_method': position_size_method,
                        'position_size': position_size,
                        'stop_loss_enabled': stop_loss_enabled,
                        'stop_loss_pct': stop_loss_pct if stop_loss_enabled else None,
                        'early_close_enabled': early_close_enabled,
                        'early_close_profit': early_close_profit if early_close_enabled else None,
                        'transaction_costs': transaction_costs,
                        'slippage_pct': slippage_pct
                    }
                    
                    # Run backtest
                    results = run_custom_backtest(backtest_params)
                    
                    # Store results in session state
                    st.session_state['backtest_results'] = results
                    st.session_state['backtest_params'] = backtest_params
                    
                    st.success(f"✅ Backtest completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Backtest failed: {e}")
    
    with col2:
        st.subheader("📋 Quick Stats")
        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']
            
            st.metric("Total Trades", results.get('total_trades', 0))
            st.metric("Win Rate", f"{results.get('win_rate', 0):.1%}")
            st.metric("Total Return", f"{results.get('total_return', 0):.1%}")
            st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
            st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.1%}")
            st.metric("Assignment Rate", f"{results.get('assignment_rate', 0):.1%}")
    
    # Display results if available
    if 'backtest_results' in st.session_state:
        display_backtest_results(st.session_state['backtest_results'], st.session_state['backtest_params'])

def run_custom_backtest(params):
    """Run custom backtest with specified parameters."""
    
    # Initialize tracking variables
    trades = []
    portfolio_values = []
    daily_returns = []
    
    start_date = params['start_date']
    end_date = params['end_date']
    symbols = params['symbols']
    
    # Initial capital
    initial_capital = 100000  # $100k for backtesting
    current_capital = initial_capital
    
    # Generate trading dates (weekly)
    trading_dates = pd.date_range(start_date, end_date, freq='W-MON')
    
    for trade_date in trading_dates:
        try:
            # Analyze each symbol
            for symbol in symbols:
                try:
                    # Get NEAT recommendation
                    recommendation = simulate_neat_recommendation(symbol, trade_date, params)
                    
                    if recommendation and should_enter_trade(recommendation, params):
                        # Calculate position size
                        position_value = calculate_position_size(
                            current_capital, 
                            recommendation, 
                            params
                        )
                        
                        if position_value > 0:
                            # Simulate trade execution
                            trade_result = simulate_trade_execution(
                                symbol, 
                                trade_date, 
                                recommendation, 
                                position_value, 
                                params
                            )
                            
                            if trade_result:
                                trades.append(trade_result)
                                current_capital += trade_result['pnl']
                
                except Exception as e:
                    continue
            
            # Record portfolio value
            portfolio_values.append({
                'date': trade_date,
                'value': current_capital,
                'return': (current_capital - initial_capital) / initial_capital
            })
            
        except Exception as e:
            continue
    
    # Calculate performance metrics
    if portfolio_values:
        portfolio_df = pd.DataFrame(portfolio_values)
        returns = portfolio_df['return'].pct_change().dropna()
        
        total_return = (current_capital - initial_capital) / initial_capital
        sharpe_ratio = calculate_sharpe_ratio(returns)
        max_drawdown = calculate_max_drawdown(portfolio_df['return'])
        
        # Trade statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        assigned_trades = len([t for t in trades if t.get('assigned', False)])
        assignment_rate = assigned_trades / total_trades if total_trades > 0 else 0
        
        avg_trade_return = np.mean([t['pnl'] for t in trades]) if trades else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'assignment_rate': assignment_rate,
            'avg_trade_return': avg_trade_return,
            'final_capital': current_capital,
            'trades': trades,
            'portfolio_values': portfolio_df,
            'daily_returns': returns
        }
    else:
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_trades': 0,
            'win_rate': 0,
            'assignment_rate': 0,
            'avg_trade_return': 0,
            'final_capital': initial_capital,
            'trades': [],
            'portfolio_values': pd.DataFrame(),
            'daily_returns': pd.Series()
        }

def simulate_neat_recommendation(symbol, date, params):
    """Simulate NEAT recommendation for backtesting."""
    
    try:
        # Create simplified recommendation based on NEAT logic
        features = neat_evolution.create_feature_vector(symbol, date)
        
        if features is None:
            return None
        
        # Use existing NEAT model if available
        if neat_advisor.network:
            output = neat_advisor.network.activate(features)
            
            bullish_confidence = max(0, min(1, (output[0] + 1) / 2))
            strike_distance = 0.03 + (max(0, min(1, (output[1] + 1) / 2))) * 0.07
            weeks_to_exp = 1 + int(max(0, min(1, (output[2] + 1) / 2)) * 7)
            position_size = max(0.1, min(1, (output[3] + 1) / 2))
            
            # Get current price (simplified)
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=date, end=date + timedelta(days=5))
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[0]
            
            # Calculate strike price
            confidence_adjustment = bullish_confidence * 0.05
            strike_price = current_price * (0.95 - confidence_adjustment)
            
            # Estimate premium
            days_to_exp = weeks_to_exp * 7
            time_value = max(0.01, 0.05 * np.sqrt(days_to_exp / 365))
            estimated_premium = time_value * current_price
            
            # Calculate metrics
            premium_yield = estimated_premium / strike_price
            weekly_yield = premium_yield * (52 / weeks_to_exp)
            assignment_prob = max(0, min(0.5, 0.5 - ((current_price - strike_price) / current_price) * 10))
            
            return {
                'symbol': symbol,
                'date': date,
                'current_price': current_price,
                'strike_price': strike_price,
                'estimated_premium': estimated_premium,
                'weeks_to_exp': weeks_to_exp,
                'confidence': bullish_confidence,
                'weekly_yield': weekly_yield,
                'assignment_probability': assignment_prob
            }
        
        return None
        
    except Exception as e:
        return None

def should_enter_trade(recommendation, params):
    """Determine if trade meets entry criteria."""
    
    return (
        recommendation['confidence'] >= params['confidence_threshold'] and
        recommendation['weekly_yield'] >= params['min_premium_yield'] and
        recommendation['assignment_probability'] <= params['max_assignment_rate']
    )

def calculate_position_size(capital, recommendation, params):
    """Calculate position size based on method."""
    
    method = params['position_size_method']
    size_param = params['position_size']
    
    if method == "Fixed Dollar":
        return min(size_param, capital * 0.2)  # Max 20% of capital
    elif method == "Fixed Percentage":
        return capital * size_param
    elif method == "Kelly Criterion":
        # Simplified Kelly calculation
        win_prob = 1 - recommendation['assignment_probability']
        avg_win = recommendation['estimated_premium']
        avg_loss = recommendation['strike_price'] * 0.1  # Assume 10% loss on assignment
        
        if avg_loss > 0:
            kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_loss
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
            return capital * kelly_fraction
        else:
            return capital * 0.05
    else:  # Risk Parity
        volatility = 0.2  # Assume 20% volatility
        target_risk = 0.1  # 10% risk
        return capital * (target_risk / volatility)

def simulate_trade_execution(symbol, entry_date, recommendation, position_value, params):
    """Simulate trade execution and outcome."""
    
    try:
        # Calculate number of contracts
        strike_price = recommendation['strike_price']
        premium = recommendation['estimated_premium']
        contracts = int(position_value / (strike_price * 100))
        
        if contracts == 0:
            return None
        
        # Apply transaction costs and slippage
        total_premium = premium * contracts * 100
        transaction_cost = params['transaction_costs'] * contracts
        slippage_cost = total_premium * params['slippage_pct']
        
        net_premium_received = total_premium - transaction_cost - slippage_cost
        
        # Simulate outcome (simplified)
        expiration_date = entry_date + timedelta(weeks=recommendation['weeks_to_exp'])
        
        # Determine if assigned (random based on probability)
        assigned = np.random.random() < recommendation['assignment_probability']
        
        if assigned:
            # Assignment occurred - calculate loss
            assignment_loss = strike_price * contracts * 100 * 0.05  # Assume 5% loss
            pnl = net_premium_received - assignment_loss - transaction_cost
        else:
            # Premium collected successfully
            pnl = net_premium_received
        
        return {
            'symbol': symbol,
            'entry_date': entry_date,
            'expiration_date': expiration_date,
            'strike_price': strike_price,
            'premium': premium,
            'contracts': contracts,
            'premium_received': net_premium_received,
            'assigned': assigned,
            'pnl': pnl,
            'return_pct': pnl / position_value if position_value > 0 else 0
        }
        
    except Exception as e:
        return None

def calculate_sharpe_ratio(returns):
    """Calculate Sharpe ratio."""
    if len(returns) < 2:
        return 0
    
    excess_returns = returns - 0.02/252  # Assume 2% risk-free rate
    return np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown."""
    if len(returns) < 2:
        return 0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return abs(drawdown.min())

def display_backtest_results(results, params):
    """Display comprehensive backtest results."""
    
    st.header("📊 Backtest Results Analysis")
    
    # Performance summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{results['total_return']:.1%}",
            delta=f"vs 0.0%"
        )
        st.metric("Total Trades", results['total_trades'])
    
    with col2:
        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        st.metric("Win Rate", f"{results['win_rate']:.1%}")
    
    with col3:
        st.metric("Max Drawdown", f"{results['max_drawdown']:.1%}")
        st.metric("Assignment Rate", f"{results['assignment_rate']:.1%}")
    
    with col4:
        st.metric("Final Capital", f"${results['final_capital']:,.0f}")
        st.metric("Avg Trade P&L", f"${results['avg_trade_return']:.0f}")
    
    # Portfolio performance chart
    if not results['portfolio_values'].empty:
        st.subheader("📈 Portfolio Performance")
        
        portfolio_df = results['portfolio_values']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_df['date'],
            y=portfolio_df['return'] * 100,
            mode='lines',
            name='Portfolio Return',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade analysis
    if results['trades']:
        st.subheader("📋 Trade Analysis")
        
        trades_df = pd.DataFrame(results['trades'])
        
        # Trade distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # P&L distribution
            fig_pnl = px.histogram(
                trades_df,
                x='pnl',
                title="P&L Distribution",
                nbins=20
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        with col2:
            # Symbol performance
            symbol_performance = trades_df.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
            fig_symbols = px.bar(
                x=symbol_performance.index,
                y=symbol_performance.values,
                title="P&L by Symbol"
            )
            st.plotly_chart(fig_symbols, use_container_width=True)
        
        # Detailed trades table
        st.subheader("📊 Detailed Trades")
        
        display_trades_df = trades_df.copy()
        display_trades_df['entry_date'] = display_trades_df['entry_date'].dt.strftime('%Y-%m-%d')
        display_trades_df['expiration_date'] = display_trades_df['expiration_date'].dt.strftime('%Y-%m-%d')
        display_trades_df['strike_price'] = display_trades_df['strike_price'].apply(lambda x: f"${x:.2f}")
        display_trades_df['premium'] = display_trades_df['premium'].apply(lambda x: f"${x:.2f}")
        display_trades_df['pnl'] = display_trades_df['pnl'].apply(lambda x: f"${x:.0f}")
        display_trades_df['return_pct'] = display_trades_df['return_pct'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(display_trades_df, use_container_width=True, hide_index=True)
    
    # Parameter summary
    st.subheader("⚙️ Backtest Parameters")
    
    params_df = pd.DataFrame([
        ["Start Date", params['start_date'].strftime('%Y-%m-%d')],
        ["End Date", params['end_date'].strftime('%Y-%m-%d')],
        ["Symbols", ", ".join(params['symbols'][:5]) + ("..." if len(params['symbols']) > 5 else "")],
        ["Confidence Threshold", f"{params['confidence_threshold']:.1%}"],
        ["Min Premium Yield", f"{params['min_premium_yield']:.1%}"],
        ["Max Assignment Rate", f"{params['max_assignment_rate']:.1%}"],
        ["Position Sizing", params['position_size_method']],
        ["Transaction Costs", f"${params['transaction_costs']:.2f}"],
        ["Slippage", f"{params['slippage_pct']:.1%}"]
    ], columns=["Parameter", "Value"])
    
    st.dataframe(params_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Custom Backtesting Interface",
        page_icon="🔬",
        layout="wide"
    )
    
    create_custom_backtesting_interface()
