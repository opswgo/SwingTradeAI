"""
NEAT Options Premium Harvesting Dashboard
User-friendly interface for NEAT-based cash-secured put recommendations.
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

# Page configuration
st.set_page_config(
    page_title="NEAT Options Premium Harvesting",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard interface."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 NEAT Options Premium Harvesting</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Cash-Secured Put Strategy using Evolutionary Neural Networks**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Capital input
    capital = st.sidebar.number_input(
        "Available Capital ($)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Total capital available for options trading"
    )
    
    # Update advisor capital
    neat_advisor.capital = capital
    
    # Risk parameters
    st.sidebar.subheader("Risk Parameters")
    max_assignment_rate = st.sidebar.slider(
        "Max Assignment Rate (%)",
        min_value=1,
        max_value=10,
        value=5,
        help="Maximum acceptable assignment probability"
    ) / 100
    
    min_weekly_yield = st.sidebar.slider(
        "Min Weekly Yield (%)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1,
        help="Minimum weekly premium yield target"
    ) / 100
    
    # Stock screening parameters
    st.sidebar.subheader("Stock Screening")
    min_market_cap = st.sidebar.selectbox(
        "Minimum Market Cap",
        options=[1e9, 5e9, 10e9, 50e9],
        index=1,
        format_func=lambda x: f"${x/1e9:.0f}B"
    )
    
    max_recommendations = st.sidebar.slider(
        "Max Recommendations",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum number of recommendations to display"
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Current Recommendations")
        
        # Generate recommendations button
        if st.button("🚀 Generate NEAT Recommendations", type="primary"):
            with st.spinner("Analyzing markets with NEAT evolutionary AI..."):
                try:
                    # Get portfolio recommendations
                    portfolio = neat_advisor.get_portfolio_recommendations()
                    recommendations = portfolio['recommendations']
                    metrics = portfolio['portfolio_metrics']
                    
                    # Store in session state
                    st.session_state['recommendations'] = recommendations
                    st.session_state['portfolio_metrics'] = metrics
                    st.session_state['risk_warnings'] = portfolio['risk_warnings']
                    st.session_state['next_actions'] = portfolio['next_actions']
                    
                    st.success(f"✅ Generated {len(recommendations)} recommendations!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {e}")
        
        # Display recommendations if available
        if 'recommendations' in st.session_state and st.session_state['recommendations']:
            recommendations = st.session_state['recommendations']
            
            # Filter recommendations based on user criteria
            filtered_recs = [
                rec for rec in recommendations
                if rec['assignment_probability'] <= max_assignment_rate
                and rec['weekly_yield'] >= min_weekly_yield
            ][:max_recommendations]
            
            if filtered_recs:
                # Create recommendations dataframe
                df_recs = pd.DataFrame(filtered_recs)
                
                # Format for display
                display_df = df_recs[[
                    'symbol', 'current_price', 'strike_price', 'expiration_weeks',
                    'weekly_yield', 'assignment_probability', 'confidence',
                    'total_premium', 'cash_required'
                ]].copy()
                
                display_df.columns = [
                    'Symbol', 'Current Price', 'Strike Price', 'Weeks to Exp',
                    'Weekly Yield', 'Assignment Risk', 'AI Confidence',
                    'Total Premium', 'Cash Required'
                ]
                
                # Format percentages and currency
                display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}")
                display_df['Strike Price'] = display_df['Strike Price'].apply(lambda x: f"${x:.2f}")
                display_df['Weekly Yield'] = display_df['Weekly Yield'].apply(lambda x: f"{x:.2%}")
                display_df['Assignment Risk'] = display_df['Assignment Risk'].apply(lambda x: f"{x:.1%}")
                display_df['AI Confidence'] = display_df['AI Confidence'].apply(lambda x: f"{x:.1%}")
                display_df['Total Premium'] = display_df['Total Premium'].apply(lambda x: f"${x:.2f}")
                display_df['Cash Required'] = display_df['Cash Required'].apply(lambda x: f"${x:,.0f}")
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Detailed recommendation cards
                st.subheader("📋 Detailed Analysis")
                
                for i, rec in enumerate(filtered_recs[:3]):  # Show top 3 in detail
                    with st.expander(f"🎯 {rec['symbol']} - {rec['weekly_yield']:.2%} Weekly Yield"):
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Strike Price", f"${rec['strike_price']:.2f}")
                            st.metric("Premium per Contract", f"${rec['estimated_premium']:.2f}")
                            st.metric("Contracts", f"{rec['contracts']}")
                        
                        with col_b:
                            st.metric("Weekly Yield", f"{rec['weekly_yield']:.2%}")
                            st.metric("Annualized Yield", f"{rec['annualized_yield']:.1%}")
                            st.metric("Effective Yield", f"{rec['effective_yield']:.2%}")
                        
                        with col_c:
                            st.metric("Assignment Risk", f"{rec['assignment_probability']:.1%}")
                            st.metric("AI Confidence", f"{rec['confidence']:.1%}")
                            st.metric("Cash Required", f"${rec['cash_required']:,.0f}")
                        
                        st.markdown(f"**AI Reasoning:** {rec['reasoning']}")
                        
                        # Risk assessment
                        if rec['assignment_probability'] > 0.05:
                            st.markdown('<div class="warning-card">⚠️ <strong>High Assignment Risk:</strong> Consider reducing position size or choosing different strike</div>', unsafe_allow_html=True)
                        elif rec['weekly_yield'] > 0.03:
                            st.markdown('<div class="success-card">🎯 <strong>High Yield Opportunity:</strong> Excellent risk-adjusted returns</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ No recommendations meet your current risk criteria. Consider adjusting parameters.")
        
        else:
            st.info("👆 Click 'Generate NEAT Recommendations' to get AI-powered options strategies")
    
    with col2:
        st.header("📊 Portfolio Metrics")
        
        if 'portfolio_metrics' in st.session_state:
            metrics = st.session_state['portfolio_metrics']
            
            # Key metrics
            st.metric(
                "Expected Weekly Return",
                f"{metrics['expected_weekly_return']:.2%}",
                help="Expected weekly return based on current recommendations"
            )
            
            st.metric(
                "Capital Utilization",
                f"{metrics['capital_utilization']:.1%}",
                help="Percentage of available capital that would be used"
            )
            
            st.metric(
                "Average Assignment Risk",
                f"{metrics['avg_assignment_probability']:.1%}",
                help="Average probability of assignment across all positions"
            )
            
            st.metric(
                "Risk Score",
                f"{metrics['risk_score']:.1f}/10",
                help="Overall portfolio risk assessment (lower is better)"
            )
            
            # Risk gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = metrics['risk_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Score"},
                gauge = {
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 3], 'color': "lightgreen"},
                        {'range': [3, 6], 'color': "yellow"},
                        {'range': [6, 10], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 5
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Risk warnings
        if 'risk_warnings' in st.session_state and st.session_state['risk_warnings']:
            st.subheader("⚠️ Risk Warnings")
            for warning in st.session_state['risk_warnings']:
                st.markdown(f'<div class="warning-card">⚠️ {warning}</div>', unsafe_allow_html=True)
        
        # Next actions
        if 'next_actions' in st.session_state and st.session_state['next_actions']:
            st.subheader("📝 Next Actions")
            for action in st.session_state['next_actions']:
                st.markdown(f"• {action}")
    
    # Additional sections
    st.header("🧠 NEAT AI Model Status")
    
    col_model1, col_model2, col_model3 = st.columns(3)
    
    with col_model1:
        if neat_advisor.best_genome:
            st.success(f"✅ Model Loaded (Fitness: {neat_advisor.best_genome.fitness:.2f})")
        else:
            st.error("❌ No Model Available")
    
    with col_model2:
        st.info(f"📊 Population Size: {neat_advisor.neat_engine.config.pop_size}")
    
    with col_model3:
        st.info(f"🎯 Feature Inputs: 50 Technical Indicators")
    
    # Model performance visualization
    if hasattr(neat_advisor.neat_engine, 'fitness_history') and neat_advisor.neat_engine.fitness_history:
        st.subheader("📈 Model Evolution History")
        
        fitness_df = pd.DataFrame(neat_advisor.neat_engine.fitness_history)
        
        fig_evolution = px.line(
            fitness_df,
            x='generation',
            y=['best_fitness', 'avg_fitness'],
            title="NEAT Evolution Progress",
            labels={'value': 'Fitness Score', 'generation': 'Generation'}
        )
        st.plotly_chart(fig_evolution, use_container_width=True)
    
    # Educational content
    with st.expander("📚 How NEAT Options Strategy Works"):
        st.markdown("""
        ### 🧬 NEAT (NeuroEvolution of Augmenting Topologies)
        
        Our AI system uses evolutionary algorithms to optimize options trading strategies:
        
        **🎯 Strategy Focus:**
        - Cash-secured puts on high-quality stocks
        - Target 1.5%+ weekly premium yields
        - Minimize assignment risk (<5%)
        - Dynamic strike price selection based on confidence
        
        **🧠 AI Features:**
        - 50 technical indicators as inputs
        - Evolving neural network topology
        - Multi-objective fitness function
        - Continuous learning and adaptation
        
        **⚖️ Risk Management:**
        - Position sizing based on available capital
        - Sector diversification monitoring
        - Assignment probability estimation
        - Dynamic stop-loss recommendations
        
        **📊 Success Metrics (Priority Order):**
        1. Low assignment rate (<5%)
        2. Weekly premium yield ≥1.5%
        3. Risk-adjusted returns (Sharpe ratio)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This is an AI-powered tool for educational purposes. Always verify options data and consult with financial professionals before trading.")

if __name__ == "__main__":
    main()
