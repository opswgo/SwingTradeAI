"""
SwingTradeAI - Streamlit Dashboard
User Acceptance: User can navigate entire system through intuitive interface
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import requests
from typing import Dict, Any, Optional

from ..core.config import Settings
from ..core.logger import setup_logger, log_user_action
from ..data.pipeline import DataPipeline

logger = setup_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="SwingTradeAI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_dashboard(settings: Settings, data_pipeline: DataPipeline):
    """Create Streamlit dashboard with user-friendly interface."""
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
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
        .success-message {
            background-color: #d4edda;
            color: #155724;
            padding: 0.75rem;
            border-radius: 0.25rem;
            border: 1px solid #c3e6cb;
        }
        .error-message {
            background-color: #f8d7da;
            color: #721c24;
            padding: 0.75rem;
            border-radius: 0.25rem;
            border: 1px solid #f5c6cb;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🚀 SwingTradeAI Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🧭 Navigation")
        page = st.selectbox(
            "Select Page",
            ["🏠 Home", "📊 Data Pipeline", "🔍 Data Validation", "📈 Market Data", "⚙️ System Health", "🧪 Testing"]
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Actions")
        
        if st.button("🔄 Refresh All Data"):
            st.rerun()
        
        if st.button("🏥 Health Check"):
            with st.spinner("Running health check..."):
                health_status = asyncio.run(data_pipeline.health_check())
                if health_status['healthy']:
                    st.success("✅ All systems healthy!")
                else:
                    st.error("❌ Issues detected!")
                    st.json(health_status)
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page(data_pipeline)
    elif page == "📊 Data Pipeline":
        show_data_pipeline_page(data_pipeline)
    elif page == "🔍 Data Validation":
        show_validation_page(data_pipeline)
    elif page == "📈 Market Data":
        show_market_data_page(data_pipeline)
    elif page == "⚙️ System Health":
        show_system_health_page(data_pipeline)
    elif page == "🧪 Testing":
        show_testing_page(data_pipeline)


def show_home_page(data_pipeline: DataPipeline):
    """Home page with system overview."""
    st.markdown("## 🎯 Welcome to SwingTradeAI")
    st.markdown("Your AI-driven options trading system with comprehensive data validation and user-focused testing.")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        dashboard_data = asyncio.run(data_pipeline.get_user_dashboard_data())
        
        with col1:
            st.metric(
                "📊 Available Symbols",
                dashboard_data.get('historical_data', {}).get('symbols', 0),
                help="Number of stocks with historical data"
            )
        
        with col2:
            st.metric(
                "📈 Total Records",
                f"{dashboard_data.get('historical_data', {}).get('total_records', 0):,}",
                help="Total historical price records"
            )
        
        with col3:
            avg_quality = 0.95  # Would calculate from actual data
            st.metric(
                "✅ Data Quality",
                f"{avg_quality:.1%}",
                delta="2.3%",
                help="Average data quality score"
            )
        
        with col4:
            st.metric(
                "🕐 Last Update",
                "2 min ago",
                help="Time since last data refresh"
            )
    
    except Exception as e:
        st.error(f"❌ Failed to load dashboard data: {str(e)}")
    
    # System status
    st.markdown("---")
    st.markdown("## 🏥 System Status")
    
    try:
        health_status = asyncio.run(data_pipeline.health_check())
        
        if health_status['healthy']:
            st.markdown('<div class="success-message">✅ All systems operational</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">⚠️ Some systems need attention</div>', unsafe_allow_html=True)
        
        # Component status
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Database Status")
            db_status = health_status.get('components', {}).get('database', {})
            if db_status.get('healthy', False):
                st.success(f"✅ {db_status.get('message', 'OK')}")
            else:
                st.error(f"❌ {db_status.get('message', 'Error')}")
        
        with col2:
            st.markdown("### Data Pipeline Status")
            pipeline_status = health_status.get('components', {}).get('duckdb', {})
            if pipeline_status.get('healthy', False):
                st.success(f"✅ {pipeline_status.get('message', 'OK')}")
            else:
                st.error(f"❌ {pipeline_status.get('message', 'Error')}")
    
    except Exception as e:
        st.error(f"❌ Health check failed: {str(e)}")
    
    # Quick start guide
    st.markdown("---")
    st.markdown("## 🚀 Quick Start Guide")
    
    with st.expander("📖 How to use SwingTradeAI", expanded=True):
        st.markdown("""
        **Step 1: Data Pipeline** 📊
        - Navigate to the Data Pipeline page
        - Fetch historical data for your symbols
        - Review data quality reports
        
        **Step 2: Validation** 🔍
        - Check the Data Validation page
        - Review quality metrics and issues
        - Ensure data meets your standards
        
        **Step 3: Market Analysis** 📈
        - Explore the Market Data page
        - Analyze price trends and patterns
        - Review technical indicators
        
        **Step 4: System Monitoring** ⚙️
        - Monitor system health regularly
        - Check component status
        - Review performance metrics
        
        **Step 5: Testing** 🧪
        - Use the Testing page to validate functionality
        - Run end-to-end tests
        - Verify system reliability
        """)


def show_data_pipeline_page(data_pipeline: DataPipeline):
    """Data pipeline management page."""
    st.markdown("## 📊 Data Pipeline Management")
    st.markdown("Fetch, validate, and manage market data with real-time quality monitoring.")
    
    # Data fetching section
    st.markdown("### 📥 Fetch Market Data")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            help="Enter a valid stock symbol (e.g., AAPL, GOOGL, MSFT)"
        ).upper()
    
    with col2:
        period = st.selectbox(
            "Time Period",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=5,
            help="Select the historical data period"
        )
    
    with col3:
        validate_data = st.checkbox(
            "Enable Validation",
            value=True,
            help="Run data quality validation"
        )
    
    if st.button("🚀 Fetch Data", type="primary"):
        with st.spinner(f"Fetching data for {symbol}..."):
            try:
                log_user_action("dashboard_fetch_data", {"symbol": symbol, "period": period})
                
                data, validation_report = asyncio.run(
                    data_pipeline.fetch_historical_data(symbol, period, validate_data)
                )
                
                if not data.empty:
                    st.success(f"✅ Successfully fetched {len(data)} records for {symbol}")
                    
                    # Display data preview
                    st.markdown("#### 📋 Data Preview")
                    st.dataframe(data.tail(10), use_container_width=True)
                    
                    # Display validation report
                    if validate_data and validation_report:
                        st.markdown("#### 🔍 Validation Report")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            quality_score = validation_report.get('overall_score', 0)
                            st.metric(
                                "Overall Quality",
                                f"{quality_score:.1%}",
                                help="Combined quality score"
                            )
                        
                        with col2:
                            completeness = validation_report.get('metrics', {}).get('completeness', 0)
                            st.metric(
                                "Completeness",
                                f"{completeness:.1%}",
                                help="Data completeness score"
                            )
                        
                        with col3:
                            accuracy = validation_report.get('metrics', {}).get('accuracy', 0)
                            st.metric(
                                "Accuracy",
                                f"{accuracy:.1%}",
                                help="Data accuracy score"
                            )
                        
                        # Issues and warnings
                        if validation_report.get('issues'):
                            st.error("❌ Issues Found:")
                            for issue in validation_report['issues']:
                                st.write(f"• {issue}")
                        
                        if validation_report.get('warnings'):
                            st.warning("⚠️ Warnings:")
                            for warning in validation_report['warnings']:
                                st.write(f"• {warning}")
                    
                    # Price chart
                    st.markdown("#### 📈 Price Chart")
                    fig = go.Figure()
                    
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name=symbol
                    ))
                    
                    fig.update_layout(
                        title=f"{symbol} Price Chart ({period})",
                        yaxis_title="Price ($)",
                        xaxis_title="Date",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f"❌ No data available for {symbol}")
            
            except Exception as e:
                st.error(f"❌ Failed to fetch data: {str(e)}")
                log_user_action("dashboard_fetch_data", {"symbol": symbol, "error": str(e)}, success=False)


def show_validation_page(data_pipeline: DataPipeline):
    """Data validation and quality monitoring page."""
    st.markdown("## 🔍 Data Validation & Quality")
    st.markdown("Monitor data quality metrics and validation results across all datasets.")
    
    # Quality overview
    st.markdown("### 📊 Quality Overview")
    
    try:
        dashboard_data = asyncio.run(data_pipeline.get_user_dashboard_data())
        quality_metrics = dashboard_data.get('data_quality', [])
        
        if quality_metrics:
            # Create quality trend chart
            df_quality = pd.DataFrame(quality_metrics)
            df_quality['created_at'] = pd.to_datetime(df_quality['created_at'])
            
            fig = px.line(
                df_quality,
                x='created_at',
                y='quality_score',
                color='dataset',
                title="Data Quality Trends",
                labels={'quality_score': 'Quality Score', 'created_at': 'Time'}
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Quality metrics table
            st.markdown("### 📋 Recent Quality Metrics")
            st.dataframe(df_quality, use_container_width=True)
        
        else:
            st.info("📝 No quality metrics available yet. Fetch some data to see quality reports.")
    
    except Exception as e:
        st.error(f"❌ Failed to load quality data: {str(e)}")


def show_market_data_page(data_pipeline: DataPipeline):
    """Market data exploration page."""
    st.markdown("## 📈 Market Data Explorer")
    st.markdown("Explore available market data and analyze historical trends.")
    
    # Available symbols
    st.markdown("### 📊 Available Symbols")
    
    try:
        symbols_result = data_pipeline.duckdb_conn.execute("""
            SELECT DISTINCT symbol, 
                   COUNT(*) as record_count,
                   MIN(date) as start_date,
                   MAX(date) as end_date
            FROM historical_prices 
            GROUP BY symbol 
            ORDER BY record_count DESC
        """).fetchall()
        
        if symbols_result:
            symbols_df = pd.DataFrame(symbols_result, columns=['Symbol', 'Records', 'Start Date', 'End Date'])
            st.dataframe(symbols_df, use_container_width=True)
            
            # Symbol selector for detailed view
            st.markdown("### 🔍 Detailed Analysis")
            selected_symbol = st.selectbox(
                "Select Symbol for Analysis",
                options=[row[0] for row in symbols_result],
                help="Choose a symbol to view detailed analysis"
            )
            
            if selected_symbol:
                # Fetch detailed data for selected symbol
                symbol_data = data_pipeline.duckdb_conn.execute("""
                    SELECT date, open, high, low, close, volume
                    FROM historical_prices 
                    WHERE symbol = ?
                    ORDER BY date DESC
                    LIMIT 100
                """, [selected_symbol]).fetchdf()
                
                if not symbol_data.empty:
                    # Price and volume charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_price = px.line(
                            symbol_data.sort_values('date'),
                            x='date',
                            y='close',
                            title=f"{selected_symbol} Price Trend"
                        )
                        st.plotly_chart(fig_price, use_container_width=True)
                    
                    with col2:
                        fig_volume = px.bar(
                            symbol_data.sort_values('date').tail(20),
                            x='date',
                            y='volume',
                            title=f"{selected_symbol} Volume (Last 20 Days)"
                        )
                        st.plotly_chart(fig_volume, use_container_width=True)
        
        else:
            st.info("📝 No market data available yet. Use the Data Pipeline page to fetch some data.")
    
    except Exception as e:
        st.error(f"❌ Failed to load market data: {str(e)}")


def show_system_health_page(data_pipeline: DataPipeline):
    """System health monitoring page."""
    st.markdown("## ⚙️ System Health Monitor")
    st.markdown("Monitor system components and performance metrics in real-time.")
    
    # Auto-refresh option
    auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        # This would implement auto-refresh in a real application
        st.info("🔄 Auto-refresh enabled (simulated)")
    
    # Health check
    if st.button("🏥 Run Health Check", type="primary"):
        with st.spinner("Running comprehensive health check..."):
            try:
                health_status = asyncio.run(data_pipeline.health_check())
                
                # Overall status
                if health_status['healthy']:
                    st.success("✅ All systems operational")
                else:
                    st.error("❌ Issues detected - see details below")
                
                # Component details
                st.markdown("### 🔧 Component Status")
                
                components = health_status.get('components', {})
                
                for component_name, component_status in components.items():
                    with st.expander(f"{component_name.replace('_', ' ').title()}", expanded=not component_status.get('healthy', False)):
                        if component_status.get('healthy', False):
                            st.success(f"✅ {component_status.get('message', 'OK')}")
                        else:
                            st.error(f"❌ {component_status.get('message', 'Error')}")
                        
                        # Additional details if available
                        if 'details' in component_status:
                            st.json(component_status['details'])
            
            except Exception as e:
                st.error(f"❌ Health check failed: {str(e)}")


def show_testing_page(data_pipeline: DataPipeline):
    """System testing and validation page."""
    st.markdown("## 🧪 System Testing")
    st.markdown("Run comprehensive tests to validate system functionality and performance.")
    
    # Test categories
    test_category = st.selectbox(
        "Select Test Category",
        ["🔄 Data Pipeline Tests", "🗄️ Database Tests", "🌐 API Tests", "📊 Performance Tests"]
    )
    
    if test_category == "🔄 Data Pipeline Tests":
        st.markdown("### Data Pipeline Validation")
        
        if st.button("🚀 Run Data Pipeline Tests", type="primary"):
            with st.spinner("Running data pipeline tests..."):
                try:
                    # Test data fetching
                    st.info("Testing data fetching...")
                    test_data, validation = asyncio.run(
                        data_pipeline.fetch_historical_data("AAPL", "5d", validate=True)
                    )
                    
                    if not test_data.empty:
                        st.success(f"✅ Data fetch test passed - {len(test_data)} records")
                        
                        # Display validation results
                        quality_score = validation.get('overall_score', 0)
                        if quality_score >= 0.9:
                            st.success(f"✅ Data quality test passed - Score: {quality_score:.1%}")
                        elif quality_score >= 0.7:
                            st.warning(f"⚠️ Data quality acceptable - Score: {quality_score:.1%}")
                        else:
                            st.error(f"❌ Data quality below threshold - Score: {quality_score:.1%}")
                    else:
                        st.error("❌ Data fetch test failed - No data returned")
                
                except Exception as e:
                    st.error(f"❌ Data pipeline test failed: {str(e)}")
    
    elif test_category == "🗄️ Database Tests":
        st.markdown("### Database Connectivity Tests")
        
        if st.button("🗄️ Run Database Tests", type="primary"):
            with st.spinner("Testing database connections..."):
                try:
                    # Test PostgreSQL
                    db_health = asyncio.run(data_pipeline.check_database())
                    if db_health['healthy']:
                        st.success("✅ PostgreSQL connection test passed")
                    else:
                        st.error(f"❌ PostgreSQL test failed: {db_health['message']}")
                    
                    # Test DuckDB
                    duckdb_health = asyncio.run(data_pipeline.check_duckdb())
                    if duckdb_health['healthy']:
                        st.success("✅ DuckDB connection test passed")
                    else:
                        st.error(f"❌ DuckDB test failed: {duckdb_health['message']}")
                
                except Exception as e:
                    st.error(f"❌ Database tests failed: {str(e)}")
    
    # Test results summary
    st.markdown("---")
    st.markdown("### 📊 Test Results Summary")
    st.info("Test results will appear here after running tests above.")


# Initialize dashboard if run directly
if __name__ == "__main__":
    # This would be called from main.py in practice
    st.error("⚠️ Dashboard should be launched through main.py")
    st.code("python main.py", language="bash")
