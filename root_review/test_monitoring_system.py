"""
Test Suite for Real-time Monitoring System - NEAT Options
Comprehensive tests for monitoring, alerts, and dashboard functionality.
"""

import unittest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import json
import threading
import time
from unittest.mock import Mock, patch, MagicMock

from realtime_performance_monitor import RealTimePerformanceMonitor
from src.database.agent_database import AgentDatabase

class TestRealTimeMonitoring(unittest.TestCase):
    """Test real-time monitoring functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.monitor = RealTimePerformanceMonitor(db_path=self.temp_db.name)
        
        # Create test agent data
        self.test_agents = {
            'AAPL': {
                'agent_id': 'test_aapl_001',
                'is_champion': True,
                'accuracy': 0.82,
                'total_return': 0.15,
                'sharpe_ratio': 1.45,
                'max_drawdown': 0.08
            },
            'MSFT': {
                'agent_id': 'test_msft_001',
                'is_champion': True,
                'accuracy': 0.78,
                'total_return': 0.12,
                'sharpe_ratio': 1.32,
                'max_drawdown': 0.06
            }
        }
    
    def tearDown(self):
        """Clean up test environment."""
        self.monitor.stop_monitoring()
        
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        self.assertIsNotNone(self.monitor.agent_db)
        self.assertIsNotNone(self.monitor.data_pipeline)
        self.assertFalse(self.monitor.monitoring_active)
        self.assertEqual(len(self.monitor.performance_data), 0)
        self.assertEqual(len(self.monitor.alert_history), 0)
    
    @patch('realtime_performance_monitor.RealTimePerformanceMonitor._get_champions_to_monitor')
    def test_start_monitoring_success(self, mock_get_champions):
        """Test successful monitoring start."""
        mock_get_champions.return_value = self.test_agents
        
        success = self.monitor.start_monitoring(['AAPL', 'MSFT'])
        
        self.assertTrue(success)
        self.assertTrue(self.monitor.monitoring_active)
        self.assertEqual(len(self.monitor.performance_data), 2)
        self.assertIn('AAPL', self.monitor.performance_data)
        self.assertIn('MSFT', self.monitor.performance_data)
    
    @patch('realtime_performance_monitor.RealTimePerformanceMonitor._get_champions_to_monitor')
    def test_start_monitoring_no_champions(self, mock_get_champions):
        """Test monitoring start with no champions."""
        mock_get_champions.return_value = {}
        
        success = self.monitor.start_monitoring(['AAPL'])
        
        self.assertFalse(success)
        self.assertFalse(self.monitor.monitoring_active)
    
    def test_stop_monitoring(self):
        """Test monitoring stop."""
        self.monitor.monitoring_active = True
        self.monitor.monitor_thread = Mock()
        self.monitor.monitor_thread.is_alive.return_value = True
        
        self.monitor.stop_monitoring()
        
        self.assertFalse(self.monitor.monitoring_active)
        self.monitor.monitor_thread.join.assert_called_once()
    
    def test_performance_tracking_initialization(self):
        """Test performance tracking initialization."""
        self.monitor._initialize_performance_tracking(self.test_agents)
        
        self.assertEqual(len(self.monitor.performance_data), 2)
        
        aapl_data = self.monitor.performance_data['AAPL']
        self.assertEqual(aapl_data['agent_id'], 'test_aapl_001')
        self.assertEqual(aapl_data['baseline_metrics']['accuracy'], 0.82)
        self.assertEqual(aapl_data['consecutive_losses'], 0)
        self.assertFalse(aapl_data['retraining_needed'])
    
    @patch('yfinance.download')
    def test_market_data_retrieval(self, mock_yf_download):
        """Test market data retrieval."""
        # Mock market data
        mock_data = pd.DataFrame({
            'Close': [150.0, 152.0, 151.5, 153.0, 154.2],
            'Volume': [1000000, 1100000, 950000, 1200000, 1050000]
        }, index=pd.date_range(start='2024-01-01', periods=5))
        
        mock_yf_download.return_value = mock_data
        
        data = self.monitor._get_recent_market_data('AAPL')
        
        self.assertFalse(data.empty)
        self.assertEqual(len(data), 5)
        mock_yf_download.assert_called_once()
    
    @patch('yfinance.download')
    def test_market_data_failure(self, mock_yf_download):
        """Test market data retrieval failure."""
        mock_yf_download.side_effect = Exception("API Error")
        
        data = self.monitor._get_recent_market_data('INVALID')
        
        self.assertTrue(data.empty)
    
    def test_current_metrics_calculation(self):
        """Test current metrics calculation."""
        # Create test market data
        market_data = pd.DataFrame({
            'Close': [150.0, 152.0, 151.5, 153.0, 154.2],
        }, index=pd.date_range(start='2024-01-01', periods=5))
        
        # Initialize performance data
        self.monitor._initialize_performance_tracking(self.test_agents)
        
        metrics = self.monitor._calculate_current_metrics('AAPL', market_data)
        
        self.assertIn('current_price', metrics)
        self.assertIn('daily_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('prediction_confidence', metrics)
        
        self.assertEqual(metrics['current_price'], 154.2)
        self.assertAlmostEqual(metrics['daily_return'], (154.2 / 153.0) - 1, places=4)
    
    def test_alert_condition_accuracy_drop(self):
        """Test accuracy drop alert condition."""
        # Initialize performance data
        self.monitor._initialize_performance_tracking(self.test_agents)
        
        # Simulate accuracy drop
        perf_data = self.monitor.performance_data['AAPL']
        perf_data['current_metrics'] = {
            'accuracy': 0.75,  # Dropped from 0.82
            'volatility': 0.20,
            'market_correlation': 0.85
        }
        
        # Check alerts
        self.monitor._check_agent_alerts('AAPL', perf_data)
        
        # Should have generated accuracy drop alert
        self.assertTrue(len(perf_data['alerts']) > 0)
        alert = perf_data['alerts'][0]
        self.assertEqual(alert['type'], 'accuracy_drop')
        self.assertEqual(alert['severity'], 'high')
    
    def test_alert_condition_volatility_spike(self):
        """Test volatility spike alert condition."""
        # Initialize performance data
        self.monitor._initialize_performance_tracking(self.test_agents)
        
        # Simulate volatility spike
        perf_data = self.monitor.performance_data['AAPL']
        perf_data['current_metrics'] = {
            'accuracy': 0.82,
            'volatility': 0.45,  # High volatility (2x normal)
            'market_correlation': 0.85
        }
        
        # Check alerts
        self.monitor._check_agent_alerts('AAPL', perf_data)
        
        # Should have generated volatility spike alert
        alerts = [alert for alert in perf_data['alerts'] if alert['type'] == 'volatility_spike']
        self.assertTrue(len(alerts) > 0)
        self.assertEqual(alerts[0]['severity'], 'medium')
    
    def test_alert_condition_correlation_break(self):
        """Test correlation breakdown alert condition."""
        # Initialize performance data
        self.monitor._initialize_performance_tracking(self.test_agents)
        
        # Simulate correlation breakdown
        perf_data = self.monitor.performance_data['AAPL']
        perf_data['current_metrics'] = {
            'accuracy': 0.82,
            'volatility': 0.20,
            'market_correlation': 0.25  # Low correlation
        }
        
        # Check alerts
        self.monitor._check_agent_alerts('AAPL', perf_data)
        
        # Should have generated correlation break alert
        alerts = [alert for alert in perf_data['alerts'] if alert['type'] == 'correlation_break']
        self.assertTrue(len(alerts) > 0)
        self.assertEqual(alerts[0]['severity'], 'medium')
    
    def test_retraining_trigger_accuracy_threshold(self):
        """Test retraining trigger for accuracy threshold."""
        # Initialize performance data
        self.monitor._initialize_performance_tracking(self.test_agents)
        
        # Set low accuracy
        perf_data = self.monitor.performance_data['AAPL']
        perf_data['current_metrics'] = {'accuracy': 0.65}  # Below 0.70 threshold
        
        should_retrain = self.monitor._should_retrain_agent('AAPL', perf_data)
        
        self.assertTrue(should_retrain)
    
    def test_retraining_trigger_consecutive_losses(self):
        """Test retraining trigger for consecutive losses."""
        # Initialize performance data
        self.monitor._initialize_performance_tracking(self.test_agents)
        
        # Set consecutive losses
        perf_data = self.monitor.performance_data['AAPL']
        perf_data['consecutive_losses'] = 6  # Above threshold of 5
        perf_data['current_metrics'] = {'accuracy': 0.80}
        
        should_retrain = self.monitor._should_retrain_agent('AAPL', perf_data)
        
        self.assertTrue(should_retrain)
    
    def test_retraining_trigger_scheduled_interval(self):
        """Test retraining trigger for scheduled interval."""
        # Initialize performance data
        self.monitor._initialize_performance_tracking(self.test_agents)
        
        # Set old last update
        perf_data = self.monitor.performance_data['AAPL']
        perf_data['last_update'] = datetime.now() - timedelta(days=8)  # 8 days ago
        perf_data['current_metrics'] = {'accuracy': 0.80}
        
        should_retrain = self.monitor._should_retrain_agent('AAPL', perf_data)
        
        self.assertTrue(should_retrain)
    
    def test_alert_callback_functionality(self):
        """Test alert callback functionality."""
        callback_called = False
        callback_symbol = None
        callback_alert = None
        
        def test_callback(symbol, alert):
            nonlocal callback_called, callback_symbol, callback_alert
            callback_called = True
            callback_symbol = symbol
            callback_alert = alert
        
        # Initialize performance data first
        self.monitor._initialize_performance_tracking(self.test_agents)
        
        self.monitor.add_alert_callback(test_callback)
        
        # Trigger an alert
        alert = {
            'type': 'test_alert',
            'severity': 'medium',
            'message': 'Test alert message'
        }
        
        self.monitor._trigger_alert('AAPL', alert)
        
        self.assertTrue(callback_called)
        self.assertEqual(callback_symbol, 'AAPL')
        self.assertEqual(callback_alert['type'], 'test_alert')
    
    def test_performance_summary_generation(self):
        """Test performance summary generation."""
        # Initialize performance data
        self.monitor._initialize_performance_tracking(self.test_agents)
        
        # Add some current metrics
        self.monitor.performance_data['AAPL']['current_metrics'] = {
            'accuracy': 0.80,
            'volatility': 0.22
        }
        self.monitor.performance_data['MSFT']['current_metrics'] = {
            'accuracy': 0.75,
            'volatility': 0.18
        }
        
        summary = self.monitor._get_performance_summary()
        
        self.assertEqual(summary['total_agents'], 2)
        self.assertEqual(summary['healthy_agents'], 2)  # No alerts or retraining needed
        self.assertAlmostEqual(summary['average_accuracy'], 0.775, places=3)
        self.assertAlmostEqual(summary['average_volatility'], 0.20, places=2)
    
    def test_monitoring_report_generation(self):
        """Test monitoring report generation."""
        # Initialize performance data
        self.monitor._initialize_performance_tracking(self.test_agents)
        
        # Add some test data
        self.monitor.performance_data['AAPL']['current_metrics'] = {'accuracy': 0.80}
        self.monitor.alert_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'test_alert',
            'severity': 'medium'
        })
        
        report = self.monitor.generate_monitoring_report()
        
        self.assertIn('report_date', report)
        self.assertIn('status', report)
        self.assertIn('performance_summary', report)
        self.assertIn('alert_summary', report)
        self.assertIn('agent_details', report)
        self.assertIn('recommendations', report)
        
        self.assertEqual(len(report['agent_details']), 2)
        self.assertIn('AAPL', report['agent_details'])
        self.assertIn('MSFT', report['agent_details'])
    
    def test_agent_health_status_assessment(self):
        """Test agent health status assessment."""
        # Initialize performance data
        self.monitor._initialize_performance_tracking(self.test_agents)
        
        # Test healthy agent
        perf_data = self.monitor.performance_data['AAPL']
        perf_data['current_metrics'] = {'accuracy': 0.82}
        
        health_status = self.monitor._get_agent_health_status('AAPL', perf_data)
        self.assertEqual(health_status, 'HEALTHY')
        
        # Test degraded agent (accuracy ratio < 0.85)
        perf_data['current_metrics'] = {'accuracy': 0.65}  # Much lower than baseline 0.82
        health_status = self.monitor._get_agent_health_status('AAPL', perf_data)
        self.assertEqual(health_status, 'DEGRADED')
        
        # Test critical agent
        perf_data['retraining_needed'] = True
        health_status = self.monitor._get_agent_health_status('AAPL', perf_data)
        self.assertEqual(health_status, 'CRITICAL')
    
    def test_current_status_retrieval(self):
        """Test current status retrieval."""
        status = self.monitor.get_current_status()
        
        self.assertIn('monitoring_active', status)
        self.assertIn('agents_monitored', status)
        self.assertIn('last_update', status)
        self.assertIn('alert_count_24h', status)
        self.assertIn('performance_summary', status)
        
        self.assertFalse(status['monitoring_active'])
        self.assertEqual(status['agents_monitored'], 0)


class TestMonitoringIntegration(unittest.TestCase):
    """Test monitoring system integration."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Initialize database with test data
        self.agent_db = AgentDatabase(self.temp_db.name)
        self._create_test_agents()
        
        self.monitor = RealTimePerformanceMonitor(db_path=self.temp_db.name)
    
    def tearDown(self):
        """Clean up integration test environment."""
        self.monitor.stop_monitoring()
        
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def _create_test_agents(self):
        """Create test agents in database."""
        test_symbols = ['AAPL', 'MSFT']
        
        for symbol in test_symbols:
            agent_data = {
                'symbol': symbol,
                'generation': 100,
                'fitness_score': 85.5,
                'network_config': {'input_nodes': 10, 'output_nodes': 3},
                'is_champion': True
            }
            
            performance_metrics = {
                'fitness_score': 85.5,
                'accuracy': 0.82,
                'total_return': 0.15,
                'sharpe_ratio': 1.45,
                'max_drawdown': 0.08,
                'win_rate': 0.75,
                'total_trades': 100,
                'validation_accuracy': 0.80
            }
            
            training_info = {
                'generations': 100,
                'population_size': 150,
                'training_time': 3600,
                'data_start': '2023-01-01',
                'data_end': '2024-01-01'
            }
            
            self.agent_db.save_agent(f"test_{symbol.lower()}_001", symbol, agent_data, performance_metrics, training_info)
    
    @patch('yfinance.download')
    def test_end_to_end_monitoring(self, mock_yf_download):
        """Test end-to-end monitoring workflow."""
        # Mock market data
        mock_data = pd.DataFrame({
            'Close': [150.0, 152.0, 151.5, 153.0, 154.2],
        }, index=pd.date_range(start='2024-01-01', periods=5))
        
        mock_yf_download.return_value = mock_data
        
        # Start monitoring
        success = self.monitor.start_monitoring(['AAPL', 'MSFT'])
        self.assertTrue(success)
        
        # Let monitoring run briefly
        time.sleep(1)
        
        # Check that performance data was initialized
        self.assertEqual(len(self.monitor.performance_data), 2)
        self.assertIn('AAPL', self.monitor.performance_data)
        self.assertIn('MSFT', self.monitor.performance_data)
        
        # Generate report
        report = self.monitor.generate_monitoring_report()
        self.assertIsNotNone(report)
        self.assertEqual(len(report['agent_details']), 2)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring_active)


def run_monitoring_tests():
    """Run all monitoring system tests."""
    
    print("🧪 NEAT Options System - Monitoring Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestRealTimeMonitoring))
    test_suite.addTest(unittest.makeSuite(TestMonitoringIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\n🚨 Errors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n✅ Success Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_monitoring_tests()
