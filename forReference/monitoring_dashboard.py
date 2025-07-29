"""
Real-time Monitoring Dashboard - NEAT Options System
Streamlit dashboard for monitoring NEAT agent performance in real-time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List

from realtime_performance_monitor import RealTimePerformanceMonitor

# Configure Streamlit page
st.set_page_config(
    page_title="NEAT Options - Real-time Monitoring",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MonitoringDashboard:
    """Streamlit dashboard for real-time monitoring."""
    
    def __init__(self):
        """Initialize monitoring dashboard."""
        self.monitor = RealTimePerformanceMonitor()
        
        # Initialize session state
        if 'monitoring_active' not in st.session_state:
            st.session_state.monitoring_active = False
        
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
    
    def render_dashboard(self):
        """Render the main monitoring dashboard."""
        
        # Header
        st.title("🔍 NEAT Options System - Real-time Monitoring")
        st.markdown("Monitor agent performance, alerts, and system health in real-time")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main content
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            self._render_system_status()
        
        with col2:
            self._render_alert_summary()
        
        with col3:
            self._render_quick_stats()
        
        # Performance metrics
        st.markdown("---")
        self._render_performance_metrics()
        
        # Agent details
        st.markdown("---")
        self._render_agent_details()
        
        # Alerts and logs
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_recent_alerts()
        
        with col2:
            self._render_system_logs()
    
    def _render_sidebar(self):
        """Render sidebar controls."""
        
        st.sidebar.title("🎛️ Monitoring Controls")
        
        # Start/Stop monitoring
        if st.sidebar.button("🟢 Start Monitoring" if not st.session_state.monitoring_active else "🔴 Stop Monitoring"):
            if not st.session_state.monitoring_active:
                success = self.monitor.start_monitoring()
                if success:
                    st.session_state.monitoring_active = True
                    st.sidebar.success("✅ Monitoring started!")
                else:
                    st.sidebar.error("❌ Failed to start monitoring")
            else:
                self.monitor.stop_monitoring()
                st.session_state.monitoring_active = False
                st.sidebar.info("🛑 Monitoring stopped")
        
        # Auto-refresh toggle
        st.session_state.auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=st.session_state.auto_refresh)
        
        # Refresh interval
        refresh_interval = st.sidebar.selectbox(
            "Refresh Interval",
            [5, 10, 30, 60],
            index=2,
            format_func=lambda x: f"{x} seconds"
        )
        
        # Manual refresh
        if st.sidebar.button("🔄 Refresh Now"):
            st.session_state.last_update = datetime.now()
            st.rerun()
        
        # Generate report
        if st.sidebar.button("📊 Generate Report"):
            with st.sidebar:
                with st.spinner("Generating report..."):
                    report = self.monitor.generate_monitoring_report()
                    if report:
                        st.success("✅ Report generated!")
                        st.download_button(
                            "📥 Download Report",
                            data=json.dumps(report, indent=2, default=str),
                            file_name=f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
        
        # Configuration
        st.sidebar.markdown("---")
        st.sidebar.markdown("**⚙️ Configuration**")
        
        # Alert thresholds
        with st.sidebar.expander("🚨 Alert Thresholds"):
            accuracy_threshold = st.slider("Accuracy Drop (%)", 1, 20, 5)
            volatility_threshold = st.slider("Volatility Spike (x)", 1.5, 5.0, 2.0, 0.1)
            correlation_threshold = st.slider("Correlation Break (%)", 10, 50, 30)
        
        # Auto-refresh logic
        if st.session_state.auto_refresh and st.session_state.monitoring_active:
            time.sleep(refresh_interval)
            st.rerun()
    
    def _render_system_status(self):
        """Render system status overview."""
        
        st.subheader("🖥️ System Status")
        
        # Get current status
        status = self.monitor.get_current_status()
        
        # Status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "🟢" if status['monitoring_active'] else "🔴"
            st.metric(
                "Monitoring Status",
                f"{status_color} {'Active' if status['monitoring_active'] else 'Inactive'}"
            )
        
        with col2:
            st.metric(
                "Agents Monitored",
                status['agents_monitored']
            )
        
        with col3:
            st.metric(
                "Alerts (24h)",
                status['alert_count_24h']
            )
        
        # Last update
        st.caption(f"Last updated: {status['last_update']}")
    
    def _render_alert_summary(self):
        """Render alert summary."""
        
        st.subheader("🚨 Alert Summary")
        
        # Mock alert data for demonstration
        alert_data = {
            'High': 2,
            'Medium': 5,
            'Low': 3
        }
        
        # Create alert severity chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(alert_data.keys()),
                y=list(alert_data.values()),
                marker_color=['red', 'orange', 'yellow']
            )
        ])
        
        fig.update_layout(
            title="Alert Severity (24h)",
            xaxis_title="Severity",
            yaxis_title="Count",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_quick_stats(self):
        """Render quick statistics."""
        
        st.subheader("📊 Quick Stats")
        
        # Performance summary
        perf_summary = self.monitor._get_performance_summary()
        
        if perf_summary:
            st.metric(
                "Healthy Agents",
                f"{perf_summary.get('healthy_agents', 0)}/{perf_summary.get('total_agents', 0)}"
            )
            
            st.metric(
                "Avg Accuracy",
                f"{perf_summary.get('average_accuracy', 0):.1%}"
            )
            
            st.metric(
                "Avg Volatility",
                f"{perf_summary.get('average_volatility', 0):.1%}"
            )
        else:
            st.info("No performance data available")
    
    def _render_performance_metrics(self):
        """Render performance metrics charts."""
        
        st.subheader("📈 Performance Metrics")
        
        # Create tabs for different metrics
        tab1, tab2, tab3, tab4 = st.tabs(["Accuracy", "Returns", "Volatility", "Correlation"])
        
        with tab1:
            self._render_accuracy_chart()
        
        with tab2:
            self._render_returns_chart()
        
        with tab3:
            self._render_volatility_chart()
        
        with tab4:
            self._render_correlation_chart()
    
    def _render_accuracy_chart(self):
        """Render accuracy performance chart."""
        
        # Generate sample data for demonstration
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
        
        fig = go.Figure()
        
        for symbol in symbols:
            # Generate realistic accuracy data
            base_accuracy = np.random.uniform(0.75, 0.85)
            noise = np.random.normal(0, 0.02, len(dates))
            accuracy_data = np.clip(base_accuracy + noise, 0.5, 1.0)
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=accuracy_data,
                mode='lines',
                name=symbol,
                line=dict(width=2)
            ))
        
        # Add threshold line
        fig.add_hline(y=0.70, line_dash="dash", line_color="red", 
                     annotation_text="Retraining Threshold")
        
        fig.update_layout(
            title="Agent Accuracy Over Time",
            xaxis_title="Time",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0.5, 1.0], tickformat='.0%'),
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_returns_chart(self):
        """Render returns performance chart."""
        
        # Generate sample returns data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        fig = go.Figure()
        
        for symbol in symbols:
            # Generate cumulative returns
            daily_returns = np.random.normal(0.001, 0.02, len(dates))  # ~0.1% daily return with 2% volatility
            cumulative_returns = np.cumprod(1 + daily_returns) - 1
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns * 100,
                mode='lines',
                name=symbol,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Cumulative Returns (30 days)",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_volatility_chart(self):
        """Render volatility chart."""
        
        # Generate sample volatility data
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
        
        fig = go.Figure()
        
        for symbol in symbols:
            # Generate volatility data
            base_vol = np.random.uniform(0.15, 0.25)
            vol_changes = np.random.normal(0, 0.01, len(dates))
            volatility_data = np.clip(base_vol + np.cumsum(vol_changes) * 0.1, 0.05, 0.5)
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=volatility_data * 100,
                mode='lines',
                name=symbol,
                line=dict(width=2)
            ))
        
        # Add alert threshold
        fig.add_hline(y=40, line_dash="dash", line_color="orange", 
                     annotation_text="High Volatility Alert")
        
        fig.update_layout(
            title="Realized Volatility (Annualized)",
            xaxis_title="Time",
            yaxis_title="Volatility (%)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_correlation_chart(self):
        """Render correlation heatmap."""
        
        # Generate sample correlation matrix
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B']
        
        # Create realistic correlation matrix
        np.random.seed(42)  # For reproducible results
        base_corr = np.random.uniform(0.3, 0.8, (len(symbols), len(symbols)))
        correlation_matrix = (base_corr + base_corr.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1.0)  # Diagonal = 1
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=symbols,
            y=symbols,
            colorscale='RdYlBu_r',
            zmid=0.5,
            text=np.round(correlation_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Agent Prediction Correlation Matrix",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_agent_details(self):
        """Render detailed agent information."""
        
        st.subheader("🤖 Agent Details")
        
        # Agent selection
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B']
        selected_symbol = st.selectbox("Select Agent", symbols)
        
        if selected_symbol:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Accuracy", "78.5%", "-2.1%")
                st.metric("Prediction Confidence", "84.2%", "+1.3%")
            
            with col2:
                st.metric("Daily Return", "0.12%", "+0.08%")
                st.metric("Sharpe Ratio", "1.42", "-0.15")
            
            with col3:
                st.metric("Health Status", "🟡 Warning")
                st.metric("Last Retrain", "3 days ago")
            
            # Agent performance history
            st.markdown("**Performance History**")
            
            # Generate sample performance data
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            performance_data = pd.DataFrame({
                'Date': dates,
                'Accuracy': np.random.uniform(0.75, 0.85, len(dates)),
                'Return': np.random.normal(0.001, 0.02, len(dates)),
                'Confidence': np.random.uniform(0.8, 0.9, len(dates))
            })
            
            # Create performance chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=performance_data['Date'],
                y=performance_data['Accuracy'],
                mode='lines',
                name='Accuracy',
                yaxis='y'
            ))
            
            fig.add_trace(go.Scatter(
                x=performance_data['Date'],
                y=performance_data['Confidence'],
                mode='lines',
                name='Confidence',
                yaxis='y'
            ))
            
            fig.update_layout(
                title=f"{selected_symbol} Performance Trends",
                xaxis_title="Date",
                yaxis=dict(title="Accuracy/Confidence", tickformat='.0%'),
                height=300,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_recent_alerts(self):
        """Render recent alerts."""
        
        st.subheader("🚨 Recent Alerts")
        
        # Sample alert data
        alerts = [
            {
                'timestamp': datetime.now() - timedelta(minutes=15),
                'symbol': 'AAPL',
                'type': 'accuracy_drop',
                'severity': 'high',
                'message': 'Accuracy dropped from 82.1% to 76.8%'
            },
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'symbol': 'MSFT',
                'type': 'volatility_spike',
                'severity': 'medium',
                'message': 'Volatility spiked to 35.2% (normal: 18.1%)'
            },
            {
                'timestamp': datetime.now() - timedelta(hours=4),
                'symbol': 'GOOGL',
                'type': 'correlation_break',
                'severity': 'medium',
                'message': 'Market correlation dropped to 28%'
            }
        ]
        
        for alert in alerts:
            severity_color = {'high': '🔴', 'medium': '🟠', 'low': '🟡'}
            color = severity_color.get(alert['severity'], '⚠️')
            
            with st.container():
                st.markdown(f"""
                **{color} {alert['symbol']} - {alert['type'].replace('_', ' ').title()}**  
                {alert['message']}  
                *{alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}*
                """)
                st.markdown("---")
    
    def _render_system_logs(self):
        """Render system logs."""
        
        st.subheader("📋 System Logs")
        
        # Sample log entries
        logs = [
            {'time': datetime.now() - timedelta(minutes=5), 'level': 'INFO', 'message': 'Performance metrics updated for all agents'},
            {'time': datetime.now() - timedelta(minutes=10), 'level': 'WARNING', 'message': 'AAPL agent accuracy below threshold'},
            {'time': datetime.now() - timedelta(minutes=15), 'level': 'INFO', 'message': 'Market data refresh completed'},
            {'time': datetime.now() - timedelta(minutes=20), 'level': 'ERROR', 'message': 'Failed to fetch options data for TSLA'},
            {'time': datetime.now() - timedelta(minutes=25), 'level': 'INFO', 'message': 'Monitoring system started'},
        ]
        
        # Create log display
        log_container = st.container()
        
        with log_container:
            for log in logs:
                level_color = {
                    'INFO': '🔵',
                    'WARNING': '🟠', 
                    'ERROR': '🔴',
                    'DEBUG': '⚪'
                }
                
                color = level_color.get(log['level'], '⚪')
                
                st.text(f"{color} {log['time'].strftime('%H:%M:%S')} [{log['level']}] {log['message']}")


def main():
    """Main dashboard application."""
    
    dashboard = MonitoringDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()
