"""
Real-time Performance Monitor - NEAT Options System
Monitor agent performance in real-time with alerts and auto-retraining triggers.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import logging
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

from src.database.agent_database import AgentDatabase
from per_stock_neat_framework import PerStockNEATFramework
from enhanced_data_pipeline import EnhancedDataPipeline

class RealTimePerformanceMonitor:
    """Real-time monitoring system for NEAT agents."""
    
    def __init__(self, db_path: str = "data/neat_agents.db"):
        """Initialize real-time performance monitor."""
        self.db_path = db_path
        self.agent_db = AgentDatabase(db_path)
        self.data_pipeline = EnhancedDataPipeline()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Monitoring configuration
        self.monitor_config = {
            'update_frequency': 300,  # 5 minutes
            'alert_thresholds': {
                'accuracy_drop': 0.05,      # 5% accuracy drop
                'drawdown_limit': 0.15,     # 15% max drawdown
                'consecutive_losses': 5,     # 5 consecutive losses
                'volatility_spike': 2.0,    # 2x normal volatility
                'correlation_break': 0.3    # Correlation drops below 30%
            },
            'retraining_triggers': {
                'accuracy_threshold': 0.70,  # Retrain if accuracy < 70%
                'performance_decline': 0.10, # 10% performance decline
                'market_regime_change': True,
                'scheduled_interval': 7      # Weekly retraining check
            },
            'notification_channels': ['console', 'file'],  # Could add email, Slack, etc.
        }
        
        # Performance tracking
        self.performance_data = {}
        self.alert_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Alert callbacks
        self.alert_callbacks = []
    
    def start_monitoring(self, symbols: List[str] = None) -> bool:
        """Start real-time monitoring for champion agents."""
        
        self.logger.info("🔍 Starting real-time performance monitoring...")
        
        # Get champion agents to monitor
        champions = self._get_champions_to_monitor(symbols)
        
        if not champions:
            self.logger.error("No champion agents found to monitor")
            return False
        
        # Initialize performance tracking
        self._initialize_performance_tracking(champions)
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(champions,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"✅ Monitoring started for {len(champions)} agents")
        return True
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        
        self.logger.info("🛑 Real-time monitoring stopped")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def get_current_status(self) -> Dict:
        """Get current monitoring status."""
        
        return {
            'monitoring_active': self.monitoring_active,
            'agents_monitored': len(self.performance_data),
            'last_update': datetime.now().isoformat(),
            'alert_count_24h': len([
                alert for alert in self.alert_history 
                if datetime.fromisoformat(alert['timestamp']) > datetime.now() - timedelta(days=1)
            ]),
            'performance_summary': self._get_performance_summary()
        }
    
    def _get_champions_to_monitor(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Get champion agents to monitor."""
        
        champions = {}
        
        # Default to all champions if no symbols specified
        if symbols is None:
            symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
                'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV'
            ]
        
        for symbol in symbols:
            try:
                agent_data = self.agent_db.load_agent(symbol)
                if agent_data and agent_data.get('is_champion', False):
                    champions[symbol] = agent_data
            except Exception as e:
                self.logger.warning(f"Could not load champion for {symbol}: {str(e)}")
        
        return champions
    
    def _initialize_performance_tracking(self, champions: Dict):
        """Initialize performance tracking for champions."""
        
        for symbol, agent_data in champions.items():
            self.performance_data[symbol] = {
                'agent_id': agent_data['agent_id'],
                'baseline_metrics': {
                    'accuracy': agent_data.get('accuracy', 0),
                    'total_return': agent_data.get('total_return', 0),
                    'sharpe_ratio': agent_data.get('sharpe_ratio', 0),
                    'max_drawdown': agent_data.get('max_drawdown', 0)
                },
                'current_metrics': {},
                'performance_history': [],
                'alerts': [],
                'last_update': datetime.now(),
                'consecutive_losses': 0,
                'retraining_needed': False
            }
    
    def _monitoring_loop(self, champions: Dict):
        """Main monitoring loop."""
        
        self.logger.info("🔄 Starting monitoring loop...")
        
        while self.monitoring_active:
            try:
                # Update performance metrics for all agents
                self._update_all_performance_metrics(champions)
                
                # Check for alerts
                self._check_alert_conditions()
                
                # Check retraining triggers
                self._check_retraining_triggers()
                
                # Log monitoring status
                self._log_monitoring_status()
                
                # Wait for next update
                time.sleep(self.monitor_config['update_frequency'])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _update_all_performance_metrics(self, champions: Dict):
        """Update performance metrics for all monitored agents."""
        
        for symbol in champions.keys():
            try:
                self._update_agent_performance(symbol)
            except Exception as e:
                self.logger.error(f"Failed to update performance for {symbol}: {str(e)}")
    
    def _update_agent_performance(self, symbol: str):
        """Update performance metrics for a single agent."""
        
        # Get recent market data
        market_data = self._get_recent_market_data(symbol)
        
        if market_data.empty:
            return
        
        # Calculate current performance metrics
        current_metrics = self._calculate_current_metrics(symbol, market_data)
        
        # Update performance data
        perf_data = self.performance_data[symbol]
        perf_data['current_metrics'] = current_metrics
        perf_data['last_update'] = datetime.now()
        
        # Add to history
        perf_data['performance_history'].append({
            'timestamp': datetime.now().isoformat(),
            'metrics': current_metrics.copy()
        })
        
        # Keep only last 100 history entries
        if len(perf_data['performance_history']) > 100:
            perf_data['performance_history'] = perf_data['performance_history'][-100:]
        
        # Log performance to database
        self._log_performance_to_db(symbol, current_metrics)
    
    def _get_recent_market_data(self, symbol: str, days: int = 5) -> pd.DataFrame:
        """Get recent market data for performance calculation."""
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            return data
            
        except Exception as e:
            self.logger.warning(f"Could not get market data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_current_metrics(self, symbol: str, market_data: pd.DataFrame) -> Dict:
        """Calculate current performance metrics."""
        
        if len(market_data) < 2:
            return {}
        
        # Calculate basic metrics
        current_price = market_data['Close'].iloc[-1]
        prev_price = market_data['Close'].iloc[-2]
        
        daily_return = (current_price / prev_price) - 1
        
        # Calculate volatility (5-day)
        returns = market_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Simulate agent prediction accuracy (in production, would use actual predictions)
        baseline_accuracy = self.performance_data[symbol]['baseline_metrics']['accuracy']
        
        # Add some realistic noise to simulate real performance
        accuracy_noise = np.random.normal(0, 0.02)  # 2% standard deviation
        current_accuracy = max(0.5, min(1.0, baseline_accuracy + accuracy_noise))
        
        # Calculate performance metrics
        metrics = {
            'current_price': current_price,
            'daily_return': daily_return,
            'volatility': volatility,
            'accuracy': current_accuracy,
            'prediction_confidence': np.random.uniform(0.6, 0.9),  # Simulated
            'market_correlation': np.random.uniform(0.7, 0.95),    # Simulated
            'sharpe_estimate': (daily_return - 0.04/252) / (volatility/np.sqrt(252)) if volatility > 0 else 0
        }
        
        return metrics
    
    def _log_performance_to_db(self, symbol: str, metrics: Dict):
        """Log performance metrics to database."""
        
        try:
            agent_id = self.performance_data[symbol]['agent_id']
            
            performance_data = {
                'date': datetime.now().date(),
                'fitness_score': metrics.get('sharpe_estimate', 0) * 100,
                'accuracy': metrics.get('accuracy', 0),
                'total_return': metrics.get('daily_return', 0),
                'sharpe_ratio': metrics.get('sharpe_estimate', 0),
                'total_trades': 1  # Simplified
            }
            
            self.agent_db.log_performance(agent_id, symbol, performance_data)
            
        except Exception as e:
            self.logger.error(f"Failed to log performance to DB for {symbol}: {str(e)}")
    
    def _check_alert_conditions(self):
        """Check for alert conditions across all monitored agents."""
        
        for symbol, perf_data in self.performance_data.items():
            try:
                self._check_agent_alerts(symbol, perf_data)
            except Exception as e:
                self.logger.error(f"Error checking alerts for {symbol}: {str(e)}")
    
    def _check_agent_alerts(self, symbol: str, perf_data: Dict):
        """Check alert conditions for a single agent."""
        
        current_metrics = perf_data.get('current_metrics', {})
        baseline_metrics = perf_data['baseline_metrics']
        
        if not current_metrics:
            return
        
        alerts = []
        
        # Check accuracy drop
        current_accuracy = current_metrics.get('accuracy', 0)
        baseline_accuracy = baseline_metrics.get('accuracy', 0)
        
        if baseline_accuracy > 0 and (baseline_accuracy - current_accuracy) > self.monitor_config['alert_thresholds']['accuracy_drop']:
            alerts.append({
                'type': 'accuracy_drop',
                'severity': 'high',
                'message': f"Accuracy dropped from {baseline_accuracy:.1%} to {current_accuracy:.1%}",
                'current_value': current_accuracy,
                'threshold': baseline_accuracy - self.monitor_config['alert_thresholds']['accuracy_drop']
            })
        
        # Check volatility spike
        current_volatility = current_metrics.get('volatility', 0)
        normal_volatility = 0.20  # Assume 20% normal volatility
        
        if current_volatility > normal_volatility * self.monitor_config['alert_thresholds']['volatility_spike']:
            alerts.append({
                'type': 'volatility_spike',
                'severity': 'medium',
                'message': f"Volatility spiked to {current_volatility:.1%} (normal: {normal_volatility:.1%})",
                'current_value': current_volatility,
                'threshold': normal_volatility * self.monitor_config['alert_thresholds']['volatility_spike']
            })
        
        # Check correlation breakdown
        current_correlation = current_metrics.get('market_correlation', 1.0)
        
        if current_correlation < self.monitor_config['alert_thresholds']['correlation_break']:
            alerts.append({
                'type': 'correlation_break',
                'severity': 'medium',
                'message': f"Market correlation dropped to {current_correlation:.1%}",
                'current_value': current_correlation,
                'threshold': self.monitor_config['alert_thresholds']['correlation_break']
            })
        
        # Process alerts
        for alert in alerts:
            self._trigger_alert(symbol, alert)
    
    def _trigger_alert(self, symbol: str, alert: Dict):
        """Trigger an alert for an agent."""
        
        # Add timestamp and symbol to alert
        alert['symbol'] = symbol
        alert['timestamp'] = datetime.now().isoformat()
        
        # Add to alert history
        self.alert_history.append(alert)
        
        # Add to agent's alert list
        self.performance_data[symbol]['alerts'].append(alert)
        
        # Log alert
        severity_emoji = {'low': '🟡', 'medium': '🟠', 'high': '🔴'}
        emoji = severity_emoji.get(alert['severity'], '⚠️')
        
        self.logger.warning(f"{emoji} ALERT [{symbol}] {alert['type'].upper()}: {alert['message']}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(symbol, alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {str(e)}")
        
        # Save alert to file
        self._save_alert_to_file(alert)
    
    def _save_alert_to_file(self, alert: Dict):
        """Save alert to file for persistence."""
        
        try:
            import os
            alert_dir = "data/alerts"
            os.makedirs(alert_dir, exist_ok=True)
            
            alert_file = f"{alert_dir}/alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            with open(alert_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to save alert to file: {str(e)}")
    
    def _check_retraining_triggers(self):
        """Check for retraining triggers across all agents."""
        
        for symbol, perf_data in self.performance_data.items():
            try:
                if self._should_retrain_agent(symbol, perf_data):
                    self._trigger_retraining(symbol)
            except Exception as e:
                self.logger.error(f"Error checking retraining for {symbol}: {str(e)}")
    
    def _should_retrain_agent(self, symbol: str, perf_data: Dict) -> bool:
        """Check if agent should be retrained."""
        
        current_metrics = perf_data.get('current_metrics', {})
        
        if not current_metrics:
            return False
        
        # Check accuracy threshold
        current_accuracy = current_metrics.get('accuracy', 0)
        if current_accuracy < self.monitor_config['retraining_triggers']['accuracy_threshold']:
            return True
        
        # Check consecutive losses (simplified)
        if perf_data['consecutive_losses'] >= self.monitor_config['alert_thresholds']['consecutive_losses']:
            return True
        
        # Check scheduled retraining
        last_update = perf_data['last_update']
        days_since_update = (datetime.now() - last_update).days
        
        if days_since_update >= self.monitor_config['retraining_triggers']['scheduled_interval']:
            return True
        
        return False
    
    def _trigger_retraining(self, symbol: str):
        """Trigger retraining for an agent."""
        
        if self.performance_data[symbol]['retraining_needed']:
            return  # Already triggered
        
        self.performance_data[symbol]['retraining_needed'] = True
        
        self.logger.info(f"🔄 RETRAINING TRIGGERED for {symbol}")
        
        # Create retraining alert
        alert = {
            'type': 'retraining_triggered',
            'severity': 'high',
            'symbol': symbol,
            'message': f"Agent retraining triggered due to performance degradation",
            'timestamp': datetime.now().isoformat()
        }
        
        self.alert_history.append(alert)
        
        # In production, would trigger actual retraining process
        # For now, just log the trigger
        self._log_retraining_trigger(symbol)
    
    def _log_retraining_trigger(self, symbol: str):
        """Log retraining trigger to database."""
        
        try:
            # Log to training sessions table
            session_data = {
                'start_time': datetime.now(),
                'status': 'TRIGGERED',
                'trigger_reason': 'performance_degradation',
                'population_size': 150,
                'hyperparameters': {'triggered_by_monitor': True}
            }
            
            self.agent_db.log_training_session(symbol, session_data)
            
        except Exception as e:
            self.logger.error(f"Failed to log retraining trigger for {symbol}: {str(e)}")
    
    def _log_monitoring_status(self):
        """Log periodic monitoring status."""
        
        active_agents = len(self.performance_data)
        recent_alerts = len([
            alert for alert in self.alert_history 
            if datetime.fromisoformat(alert['timestamp']) > datetime.now() - timedelta(hours=1)
        ])
        
        agents_needing_retraining = len([
            perf for perf in self.performance_data.values() 
            if perf['retraining_needed']
        ])
        
        self.logger.info(f"📊 Monitor Status: {active_agents} agents, "
                        f"{recent_alerts} alerts (1h), "
                        f"{agents_needing_retraining} need retraining")
    
    def _get_performance_summary(self) -> Dict:
        """Get performance summary for all monitored agents."""
        
        if not self.performance_data:
            return {}
        
        summary = {
            'total_agents': len(self.performance_data),
            'healthy_agents': 0,
            'agents_with_alerts': 0,
            'agents_needing_retraining': 0,
            'average_accuracy': 0,
            'average_volatility': 0
        }
        
        accuracies = []
        volatilities = []
        
        for symbol, perf_data in self.performance_data.items():
            current_metrics = perf_data.get('current_metrics', {})
            
            if current_metrics:
                accuracies.append(current_metrics.get('accuracy', 0))
                volatilities.append(current_metrics.get('volatility', 0))
            
            if perf_data['alerts']:
                summary['agents_with_alerts'] += 1
            
            if perf_data['retraining_needed']:
                summary['agents_needing_retraining'] += 1
            
            if not perf_data['alerts'] and not perf_data['retraining_needed']:
                summary['healthy_agents'] += 1
        
        if accuracies:
            summary['average_accuracy'] = np.mean(accuracies)
        
        if volatilities:
            summary['average_volatility'] = np.mean(volatilities)
        
        return summary
    
    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report."""
        
        report = {
            'report_date': datetime.now().isoformat(),
            'monitoring_period': '24_hours',
            'status': self.get_current_status(),
            'performance_summary': self._get_performance_summary(),
            'alert_summary': self._get_alert_summary(),
            'agent_details': {},
            'recommendations': []
        }
        
        # Agent details
        for symbol, perf_data in self.performance_data.items():
            report['agent_details'][symbol] = {
                'current_metrics': perf_data.get('current_metrics', {}),
                'baseline_metrics': perf_data['baseline_metrics'],
                'recent_alerts': perf_data['alerts'][-5:],  # Last 5 alerts
                'retraining_needed': perf_data['retraining_needed'],
                'health_status': self._get_agent_health_status(symbol, perf_data)
            }
        
        # Recommendations
        report['recommendations'] = self._generate_monitoring_recommendations()
        
        # Save report
        self._save_monitoring_report(report)
        
        return report
    
    def _get_alert_summary(self) -> Dict:
        """Get alert summary for the report."""
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if datetime.fromisoformat(alert['timestamp']) > datetime.now() - timedelta(days=1)
        ]
        
        alert_types = {}
        severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        
        for alert in recent_alerts:
            alert_type = alert.get('type', 'unknown')
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            
            severity = alert.get('severity', 'medium')
            severity_counts[severity] += 1
        
        return {
            'total_alerts_24h': len(recent_alerts),
            'alert_types': alert_types,
            'severity_breakdown': severity_counts,
            'most_common_alert': max(alert_types.items(), key=lambda x: x[1])[0] if alert_types else None
        }
    
    def _get_agent_health_status(self, symbol: str, perf_data: Dict) -> str:
        """Get health status for an agent."""
        
        if perf_data['retraining_needed']:
            return 'CRITICAL'
        
        if len(perf_data['alerts']) > 0:
            recent_alerts = [
                alert for alert in perf_data['alerts']
                if datetime.fromisoformat(alert['timestamp']) > datetime.now() - timedelta(hours=1)
            ]
            
            if recent_alerts:
                return 'WARNING'
        
        current_metrics = perf_data.get('current_metrics', {})
        baseline_metrics = perf_data['baseline_metrics']
        
        if current_metrics:
            accuracy_ratio = current_metrics.get('accuracy', 0) / max(baseline_metrics.get('accuracy', 1), 0.01)
            
            if accuracy_ratio > 0.95:
                return 'HEALTHY'
            elif accuracy_ratio > 0.85:
                return 'GOOD'
            else:
                return 'DEGRADED'
        
        return 'UNKNOWN'
    
    def _generate_monitoring_recommendations(self) -> List[str]:
        """Generate recommendations based on monitoring data."""
        
        recommendations = []
        
        # Check for widespread issues
        agents_needing_retraining = sum(1 for perf in self.performance_data.values() if perf['retraining_needed'])
        
        if agents_needing_retraining > len(self.performance_data) * 0.3:
            recommendations.append("High number of agents need retraining - consider market regime change")
        
        # Check for alert patterns
        recent_alerts = [
            alert for alert in self.alert_history 
            if datetime.fromisoformat(alert['timestamp']) > datetime.now() - timedelta(hours=4)
        ]
        
        if len(recent_alerts) > 10:
            recommendations.append("High alert frequency - investigate market conditions")
        
        # Check performance degradation
        degraded_agents = [
            symbol for symbol, perf in self.performance_data.items()
            if self._get_agent_health_status(symbol, perf) in ['CRITICAL', 'DEGRADED']
        ]
        
        if degraded_agents:
            recommendations.append(f"Review agents with degraded performance: {', '.join(degraded_agents)}")
        
        if not recommendations:
            recommendations.append("System operating normally - continue monitoring")
        
        return recommendations
    
    def _save_monitoring_report(self, report: Dict):
        """Save monitoring report to file."""
        
        try:
            import os
            report_dir = "data/monitoring_reports"
            os.makedirs(report_dir, exist_ok=True)
            
            report_path = f"{report_dir}/monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"📄 Monitoring report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save monitoring report: {str(e)}")


def run_realtime_monitoring():
    """Run real-time performance monitoring."""
    
    print("🔍 NEAT Options System - Real-time Performance Monitoring")
    print("="*60)
    
    monitor = RealTimePerformanceMonitor()
    
    try:
        # Start monitoring
        success = monitor.start_monitoring()
        
        if success:
            print("✅ Real-time monitoring started!")
            print("Press Ctrl+C to stop monitoring...")
            
            # Run for demonstration (in production, would run continuously)
            time.sleep(30)  # Monitor for 30 seconds
            
            # Generate report
            report = monitor.generate_monitoring_report()
            
            print(f"\n📊 Monitoring Report Generated:")
            print(f"Agents monitored: {report['status']['agents_monitored']}")
            print(f"Alerts (24h): {report['status']['alert_count_24h']}")
            print(f"Healthy agents: {report['performance_summary']['healthy_agents']}")
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            return report
        else:
            print("❌ Failed to start monitoring")
            return None
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        monitor.stop_monitoring()
    except Exception as e:
        print(f"\n❌ Monitoring failed: {str(e)}")
        monitor.stop_monitoring()
        raise


if __name__ == "__main__":
    run_realtime_monitoring()
