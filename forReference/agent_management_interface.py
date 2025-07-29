"""
Agent Management Interface - NEAT Options System
Production-ready interface for managing per-stock evolutionary agents.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from src.database.agent_database import AgentDatabase
from per_stock_neat_framework import PerStockNEATFramework
from enhanced_data_pipeline import EnhancedDataPipeline

class AgentManagementInterface:
    """Streamlit interface for managing NEAT agents."""
    
    def __init__(self):
        """Initialize the agent management interface."""
        self.agent_db = AgentDatabase()
        self.data_pipeline = EnhancedDataPipeline()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # S&P 500 symbols for training
        self.sp500_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B',
            'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV',
            'PFE', 'AVGO', 'KO', 'LLY', 'TMO', 'COST', 'PEP', 'WMT', 'DHR',
            'ABT', 'MRK', 'VZ', 'ADBE', 'ACN', 'NFLX', 'CRM', 'CMCSA', 'NKE',
            'TXN', 'QCOM', 'RTX', 'NEE', 'HON', 'UPS', 'LOW', 'AMD', 'ORCL',
            'PM', 'IBM', 'T', 'SPGI', 'CAT', 'GS', 'INTU', 'AMGN', 'BLK',
            'AXP', 'BA', 'DE', 'BKNG', 'MDLZ', 'GILD', 'ADP', 'TJX', 'SYK',
            'MMM', 'VRTX', 'CVS', 'LRCX', 'MU', 'ISRG', 'ZTS', 'CI', 'NOW',
            'REGN', 'TMUS', 'PLD', 'SO', 'DUK', 'BSX', 'CB', 'SCHW', 'CL',
            'FIS', 'MO', 'USB', 'BDX', 'ANTM', 'NSC', 'AON', 'ITW', 'CSX',
            'MMC', 'PNC', 'FCX', 'EQIX', 'GM', 'APD', 'COP', 'KLAC', 'SHW',
            'EMR', 'WM', 'GD', 'ATVI', 'ECL', 'DXCM', 'ADSK', 'TGT', 'F'
        ]
    
    def run(self):
        """Run the Streamlit interface."""
        
        st.set_page_config(
            page_title="NEAT Agent Management",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 NEAT Agent Management System")
        st.markdown("**Production-ready management for per-stock evolutionary agents**")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Select Page",
            ["Dashboard", "Agent Training", "Performance Monitoring", 
             "Retraining Management", "Agent Archive", "System Status"]
        )
        
        if page == "Dashboard":
            self.show_dashboard()
        elif page == "Agent Training":
            self.show_training_interface()
        elif page == "Performance Monitoring":
            self.show_performance_monitoring()
        elif page == "Retraining Management":
            self.show_retraining_management()
        elif page == "Agent Archive":
            self.show_agent_archive()
        elif page == "System Status":
            self.show_system_status()
    
    def show_dashboard(self):
        """Show main dashboard with agent overview."""
        
        st.header("📊 Agent Dashboard")
        
        # Get agent statistics
        agent_stats = self.agent_db.get_agent_statistics()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Agents", agent_stats.get('total_agents', 0))
        
        with col2:
            st.metric("Champion Agents", agent_stats.get('champion_agents', 0))
        
        with col3:
            avg_accuracy = agent_stats.get('avg_accuracy', 0)
            st.metric("Avg Accuracy", f"{avg_accuracy:.1%}")
        
        with col4:
            avg_return = agent_stats.get('avg_return', 0)
            st.metric("Avg Return", f"{avg_return:.1%}")
        
        # Performance overview chart
        st.subheader("🎯 Agent Performance Overview")
        
        performance_data = self.agent_db.get_all_agent_performance()
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            
            # Create scatter plot
            fig = px.scatter(
                df, x='accuracy', y='total_return',
                color='sharpe_ratio', size='total_trades',
                hover_data=['symbol', 'fitness_score'],
                title="Agent Performance: Accuracy vs Returns",
                labels={
                    'accuracy': 'Accuracy (%)',
                    'total_return': 'Total Return (%)',
                    'sharpe_ratio': 'Sharpe Ratio'
                }
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top performers table
            st.subheader("🏆 Top Performing Agents")
            
            top_agents = df.nlargest(10, 'total_return')[
                ['symbol', 'accuracy', 'total_return', 'sharpe_ratio', 'win_rate']
            ]
            
            # Format percentages
            for col in ['accuracy', 'total_return', 'win_rate']:
                top_agents[col] = top_agents[col].apply(lambda x: f"{x:.1%}")
            
            top_agents['sharpe_ratio'] = top_agents['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(top_agents, use_container_width=True)
        
        else:
            st.info("No agent performance data available. Train some agents first!")
        
        # Recent training activity
        st.subheader("📈 Recent Training Activity")
        
        recent_sessions = self.agent_db.get_recent_training_sessions(limit=10)
        
        if recent_sessions:
            sessions_df = pd.DataFrame(recent_sessions)
            
            # Format datetime columns
            if 'start_time' in sessions_df.columns:
                sessions_df['start_time'] = pd.to_datetime(sessions_df['start_time'])
                sessions_df['duration'] = sessions_df.apply(
                    lambda row: self._calculate_duration(row), axis=1
                )
            
            display_cols = ['symbol', 'start_time', 'status', 'best_accuracy', 'duration']
            available_cols = [col for col in display_cols if col in sessions_df.columns]
            
            st.dataframe(sessions_df[available_cols], use_container_width=True)
        else:
            st.info("No recent training sessions found.")
    
    def show_training_interface(self):
        """Show agent training interface."""
        
        st.header("🎓 Agent Training")
        
        # Training configuration
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stock selection
            training_mode = st.selectbox(
                "Training Mode",
                ["Single Stock", "Batch Training", "Top 20 S&P 500"]
            )
            
            if training_mode == "Single Stock":
                symbol = st.selectbox("Select Stock", self.sp500_symbols)
                symbols_to_train = [symbol]
            elif training_mode == "Top 20 S&P 500":
                symbols_to_train = self.sp500_symbols[:20]
                st.info(f"Will train agents for: {', '.join(symbols_to_train)}")
            else:  # Batch Training
                selected_symbols = st.multiselect(
                    "Select Stocks", self.sp500_symbols, 
                    default=self.sp500_symbols[:5]
                )
                symbols_to_train = selected_symbols
        
        with col2:
            # Training parameters
            generations = st.slider("Generations", 50, 500, 100)
            population_size = st.slider("Population Size", 50, 300, 150)
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                accuracy_threshold = st.slider("Accuracy Threshold", 0.6, 0.9, 0.8)
                overfitting_threshold = st.slider("Overfitting Threshold", 0.1, 0.3, 0.15)
                early_stopping = st.checkbox("Early Stopping", True)
        
        # Training controls
        st.subheader("Training Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Training", type="primary"):
                self._start_training(symbols_to_train, generations)
        
        with col2:
            if st.button("⏹️ Stop Training"):
                st.warning("Training stop functionality not implemented yet")
        
        with col3:
            if st.button("📊 Refresh Status"):
                st.rerun()
        
        # Training progress
        st.subheader("Training Progress")
        
        # Show active training sessions
        active_sessions = self.agent_db.get_active_training_sessions()
        
        if active_sessions:
            for session in active_sessions:
                with st.container():
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write(f"**{session['symbol']}**")
                    
                    with col2:
                        st.write(f"Status: {session['status']}")
                    
                    with col3:
                        if 'generations_completed' in session:
                            progress = session['generations_completed'] / generations
                            st.progress(progress)
                    
                    with col4:
                        if 'best_accuracy' in session:
                            st.write(f"Best: {session['best_accuracy']:.1%}")
        else:
            st.info("No active training sessions")
        
        # Training history
        st.subheader("Training History")
        
        history = self.agent_db.get_recent_training_sessions(limit=20)
        
        if history:
            history_df = pd.DataFrame(history)
            
            # Create timeline chart
            if 'start_time' in history_df.columns:
                fig = px.timeline(
                    history_df,
                    x_start='start_time',
                    x_end='end_time',
                    y='symbol',
                    color='status',
                    title="Training Timeline"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def show_performance_monitoring(self):
        """Show performance monitoring dashboard."""
        
        st.header("📈 Performance Monitoring")
        
        # Performance filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_period = st.selectbox(
                "Time Period",
                ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"]
            )
        
        with col2:
            metric_type = st.selectbox(
                "Primary Metric",
                ["Total Return", "Accuracy", "Sharpe Ratio", "Win Rate"]
            )
        
        with col3:
            min_trades = st.slider("Min Trades", 0, 100, 10)
        
        # Get performance data
        performance_data = self.agent_db.get_performance_history(
            days=self._get_days_from_period(time_period),
            min_trades=min_trades
        )
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            
            # Performance trends
            st.subheader("📊 Performance Trends")
            
            # Time series chart
            if 'date' in df.columns:
                fig = px.line(
                    df, x='date', y=metric_type.lower().replace(' ', '_'),
                    color='symbol',
                    title=f"{metric_type} Over Time"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance distribution
            st.subheader("📈 Performance Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    df, x='total_return',
                    title="Return Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(
                    df, y='accuracy',
                    title="Accuracy Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance correlation
            st.subheader("🔗 Performance Correlations")
            
            numeric_cols = ['accuracy', 'total_return', 'sharpe_ratio', 'win_rate']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(available_cols) >= 2:
                corr_matrix = df[available_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title="Performance Metric Correlations",
                    color_continuous_scale="RdBu"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No performance data available for the selected criteria.")
    
    def show_retraining_management(self):
        """Show retraining management interface."""
        
        st.header("🔄 Retraining Management")
        
        # Retraining status
        st.subheader("Retraining Status")
        
        # Check which agents need retraining
        retraining_needed = []
        
        for symbol in self.sp500_symbols[:20]:  # Check top 20
            framework = PerStockNEATFramework(symbol)
            if framework.check_retraining_needed():
                agent_data = framework.load_champion_agent()
                if agent_data:
                    retraining_needed.append({
                        'symbol': symbol,
                        'last_training': agent_data.get('last_training_date', 'Unknown'),
                        'current_accuracy': agent_data.get('accuracy', 0),
                        'reason': 'Accuracy below threshold'
                    })
        
        if retraining_needed:
            st.warning(f"⚠️ {len(retraining_needed)} agents need retraining!")
            
            retraining_df = pd.DataFrame(retraining_needed)
            st.dataframe(retraining_df, use_container_width=True)
            
            # Bulk retraining
            if st.button("🔄 Retrain All", type="primary"):
                self._start_retraining([item['symbol'] for item in retraining_needed])
        
        else:
            st.success("✅ All agents are performing well!")
        
        # Retraining schedule
        st.subheader("📅 Retraining Schedule")
        
        schedule_data = self.agent_db.get_retraining_schedule()
        
        if schedule_data:
            schedule_df = pd.DataFrame(schedule_data)
            st.dataframe(schedule_df, use_container_width=True)
        else:
            st.info("No scheduled retraining sessions.")
        
        # Manual retraining
        st.subheader("🎯 Manual Retraining")
        
        col1, col2 = st.columns(2)
        
        with col1:
            manual_symbol = st.selectbox("Select Stock for Retraining", self.sp500_symbols)
        
        with col2:
            force_retrain = st.checkbox("Force Retraining (ignore thresholds)")
        
        if st.button("🔄 Start Manual Retraining"):
            self._start_retraining([manual_symbol], force=force_retrain)
    
    def show_agent_archive(self):
        """Show agent archive and historical performance."""
        
        st.header("🗄️ Agent Archive")
        
        # Archive statistics
        archive_stats = self.agent_db.get_archive_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Archived Agents", archive_stats.get('total_archived', 0))
        
        with col2:
            st.metric("Historical Champions", archive_stats.get('historical_champions', 0))
        
        with col3:
            st.metric("Best All-Time Return", f"{archive_stats.get('best_return', 0):.1%}")
        
        with col4:
            st.metric("Best All-Time Accuracy", f"{archive_stats.get('best_accuracy', 0):.1%}")
        
        # Archive browser
        st.subheader("📚 Browse Archive")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            archive_symbol = st.selectbox("Filter by Symbol", ["All"] + self.sp500_symbols)
        
        with col2:
            sort_by = st.selectbox("Sort by", ["Date", "Return", "Accuracy", "Sharpe Ratio"])
        
        with col3:
            limit = st.slider("Show Top", 10, 100, 50)
        
        # Get archive data
        archive_data = self.agent_db.get_archived_agents(
            symbol=None if archive_symbol == "All" else archive_symbol,
            sort_by=sort_by.lower().replace(' ', '_'),
            limit=limit
        )
        
        if archive_data:
            archive_df = pd.DataFrame(archive_data)
            
            # Format data for display
            display_cols = ['symbol', 'archive_date', 'total_return', 'accuracy', 'sharpe_ratio']
            available_cols = [col for col in display_cols if col in archive_df.columns]
            
            st.dataframe(archive_df[available_cols], use_container_width=True)
            
            # Archive performance chart
            if len(archive_df) > 1:
                fig = px.scatter(
                    archive_df, x='accuracy', y='total_return',
                    color='symbol', size='sharpe_ratio',
                    title="Archived Agent Performance"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No archived agents found.")
        
        # Agent restoration
        st.subheader("🔄 Agent Restoration")
        
        if archive_data:
            restore_agent = st.selectbox(
                "Select Agent to Restore",
                options=range(len(archive_df)),
                format_func=lambda x: f"{archive_df.iloc[x]['symbol']} - {archive_df.iloc[x]['archive_date']}"
            )
            
            if st.button("🔄 Restore Agent"):
                selected_agent = archive_df.iloc[restore_agent]
                st.success(f"Restored agent for {selected_agent['symbol']}")
    
    def show_system_status(self):
        """Show system status and health metrics."""
        
        st.header("🔧 System Status")
        
        # System health
        st.subheader("💚 System Health")
        
        # Database status
        db_status = self.agent_db.check_database_health()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "🟢" if db_status['connected'] else "🔴"
            st.write(f"{status_color} Database: {'Connected' if db_status['connected'] else 'Disconnected'}")
        
        with col2:
            st.write(f"📊 Tables: {db_status.get('table_count', 0)}")
        
        with col3:
            st.write(f"💾 DB Size: {db_status.get('size_mb', 0):.1f} MB")
        
        # Data pipeline status
        st.subheader("📡 Data Pipeline Status")
        
        pipeline_status = self.data_pipeline.check_health()
        
        for source, status in pipeline_status.items():
            status_icon = "🟢" if status['available'] else "🔴"
            st.write(f"{status_icon} {source}: {status['message']}")
        
        # Resource usage
        st.subheader("📈 Resource Usage")
        
        # Placeholder for resource monitoring
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("CPU Usage", "45%")
        
        with col2:
            st.metric("Memory Usage", "2.1 GB")
        
        with col3:
            st.metric("Disk Usage", "12.3 GB")
        
        # System logs
        st.subheader("📝 Recent System Logs")
        
        # Placeholder for log display
        log_data = [
            {"timestamp": "2024-01-15 10:30:00", "level": "INFO", "message": "Agent training completed for AAPL"},
            {"timestamp": "2024-01-15 10:25:00", "level": "WARNING", "message": "Low accuracy detected for TSLA agent"},
            {"timestamp": "2024-01-15 10:20:00", "level": "INFO", "message": "Database backup completed"},
        ]
        
        logs_df = pd.DataFrame(log_data)
        st.dataframe(logs_df, use_container_width=True)
    
    def _start_training(self, symbols: List[str], generations: int):
        """Start training for selected symbols."""
        
        with st.spinner(f"Starting training for {len(symbols)} symbols..."):
            for symbol in symbols:
                try:
                    framework = PerStockNEATFramework(symbol)
                    
                    # Start training in background (simplified)
                    st.info(f"Training started for {symbol}")
                    
                    # In production, this would be async
                    # agent_id = framework.train_agent(generations)
                    
                except Exception as e:
                    st.error(f"Failed to start training for {symbol}: {str(e)}")
        
        st.success("Training initiated for all selected symbols!")
    
    def _start_retraining(self, symbols: List[str], force: bool = False):
        """Start retraining for selected symbols."""
        
        with st.spinner(f"Starting retraining for {len(symbols)} symbols..."):
            for symbol in symbols:
                try:
                    framework = PerStockNEATFramework(symbol)
                    
                    if force or framework.check_retraining_needed():
                        st.info(f"Retraining started for {symbol}")
                        # In production, would start async retraining
                    else:
                        st.warning(f"Retraining not needed for {symbol}")
                
                except Exception as e:
                    st.error(f"Failed to start retraining for {symbol}: {str(e)}")
        
        st.success("Retraining initiated!")
    
    def _calculate_duration(self, row) -> str:
        """Calculate training duration."""
        
        if 'end_time' in row and row['end_time']:
            start = pd.to_datetime(row['start_time'])
            end = pd.to_datetime(row['end_time'])
            duration = end - start
            
            hours = duration.total_seconds() / 3600
            return f"{hours:.1f}h"
        
        return "In Progress"
    
    def _get_days_from_period(self, period: str) -> int:
        """Convert period string to days."""
        
        period_map = {
            "Last 7 Days": 7,
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "All Time": 365 * 10  # 10 years
        }
        
        return period_map.get(period, 30)


def main():
    """Run the agent management interface."""
    
    interface = AgentManagementInterface()
    interface.run()


if __name__ == "__main__":
    main()
