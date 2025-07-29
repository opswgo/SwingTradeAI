#!/usr/bin/env python3
"""
Comprehensive Test Suite for Trading System
Tests all critical functionality with real data validation.
"""

import unittest
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from practical_neat_system import PracticalNEATTrader

class TestTradingSystem(unittest.TestCase):
    """Comprehensive tests for the trading system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.trader = PracticalNEATTrader()
        cls.test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        cls.test_data = {}
        
        # Fetch real test data
        print("\n🔄 Fetching test data...")
        for symbol in cls.test_symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="2y", auto_adjust=True)
                if len(data) >= 500:
                    cls.test_data[symbol] = data
                    print(f"✅ {symbol}: {len(data)} days of data")
                else:
                    print(f"❌ {symbol}: Insufficient data ({len(data)} days)")
            except Exception as e:
                print(f"❌ {symbol}: Data fetch failed - {e}")
    
    def test_01_data_availability(self):
        """Test that we have sufficient real market data."""
        self.assertGreater(len(self.test_data), 0, "No test data available")
        
        for symbol, data in self.test_data.items():
            with self.subTest(symbol=symbol):
                self.assertGreaterEqual(len(data), 500, 
                                      f"{symbol} has insufficient data: {len(data)} days")
                self.assertIn('Close', data.columns, f"{symbol} missing Close prices")
                self.assertIn('Volume', data.columns, f"{symbol} missing Volume data")
                
                # Check for data quality
                self.assertFalse(data['Close'].isna().all(), f"{symbol} has no valid prices")
                self.assertGreater(data['Close'].iloc[-1], 0, f"{symbol} has invalid current price")
    
    def test_02_feature_engineering(self):
        """Test feature engineering with real data."""
        for symbol, data in self.test_data.items():
            with self.subTest(symbol=symbol):
                features = self.trader.engineer_features(data)
                
                # Basic feature validation
                self.assertIsInstance(features, pd.DataFrame, f"{symbol} features not DataFrame")
                self.assertGreater(len(features), 0, f"{symbol} no features generated")
                self.assertGreater(len(features.columns), 5, f"{symbol} insufficient feature columns")
                
                # Check for NaN/inf values
                self.assertFalse(features.isna().all().all(), f"{symbol} all features are NaN")
                self.assertFalse(np.isinf(features).any().any(), f"{symbol} features contain infinity")
                
                # Validate feature ranges (normalized features should be reasonable)
                for col in features.columns:
                    col_data = features[col].dropna()
                    if len(col_data) > 0:
                        self.assertLess(abs(col_data.mean()), 10, 
                                      f"{symbol} feature {col} mean too extreme: {col_data.mean()}")
    
    def test_03_label_creation(self):
        """Test weekly label creation."""
        for symbol, data in self.test_data.items():
            with self.subTest(symbol=symbol):
                labels = self.trader.create_weekly_labels(data, horizon_weeks=4)
                
                self.assertIsInstance(labels, np.ndarray, f"{symbol} labels not numpy array")
                self.assertGreater(len(labels), 0, f"{symbol} no labels generated")
                
                # Check label distribution
                unique_labels = np.unique(labels[~np.isnan(labels)])
                self.assertGreater(len(unique_labels), 1, f"{symbol} labels have no variation")
                
                # Labels should be in reasonable range
                valid_labels = labels[~np.isnan(labels)]
                self.assertTrue(np.all(np.abs(valid_labels) <= 2), 
                              f"{symbol} labels outside expected range")
    
    def test_04_model_training(self):
        """Test model training with real data."""
        # Test with one symbol to avoid long test times
        test_symbol = 'AAPL'
        if test_symbol not in self.test_data:
            self.skipTest(f"No data available for {test_symbol}")
        
        data = self.test_data[test_symbol]
        
        # Train model
        result = self.trader.train_weekly_agent(test_symbol, data, horizon_weeks=4)
        
        self.assertIsInstance(result, dict, "Training result not a dictionary")
        self.assertIn('success', result, "Training result missing success indicator")
        
        if result['success']:
            self.assertIn('performance', result, "Successful training missing performance metrics")
            self.assertIn(test_symbol, self.trader.models, f"Model not saved for {test_symbol}")
            
            # Validate performance metrics
            perf = result['performance']
            self.assertIn('total_return', perf, "Performance missing total_return")
            self.assertIn('win_rate', perf, "Performance missing win_rate")
            self.assertIsInstance(perf['total_return'], (int, float), "Invalid total_return type")
            self.assertIsInstance(perf['win_rate'], (int, float), "Invalid win_rate type")
            
            # Performance should be reasonable
            self.assertGreaterEqual(perf['win_rate'], 0, "Win rate cannot be negative")
            self.assertLessEqual(perf['win_rate'], 1, "Win rate cannot exceed 100%")
        else:
            self.assertIn('error', result, "Failed training should include error message")
    
    def test_05_prediction_functionality(self):
        """Test prediction functionality."""
        # First ensure we have a trained model
        test_symbol = 'AAPL'
        if test_symbol not in self.test_data:
            self.skipTest(f"No data available for {test_symbol}")
        
        # Train if not already trained
        if test_symbol not in self.trader.models:
            data = self.test_data[test_symbol]
            result = self.trader.train_weekly_agent(test_symbol, data)
            if not result.get('success', False):
                self.skipTest(f"Could not train model for {test_symbol}")
        
        # Test prediction
        prediction = self.trader.predict_next_weeks(test_symbol, weeks_ahead=4)
        
        self.assertIsInstance(prediction, dict, "Prediction not a dictionary")
        
        if 'error' in prediction:
            self.fail(f"Prediction failed: {prediction['error']}")
        
        # Validate prediction structure
        required_fields = ['symbol', 'current_price', 'prediction_signal', 
                          'confidence', 'recommendation', 'horizon_weeks']
        for field in required_fields:
            self.assertIn(field, prediction, f"Prediction missing {field}")
        
        # Validate prediction values
        self.assertEqual(prediction['symbol'], test_symbol)
        self.assertGreater(prediction['current_price'], 0, "Invalid current price")
        self.assertIsInstance(prediction['prediction_signal'], (int, float), "Invalid prediction signal")
        self.assertIn(prediction['confidence'], ['Low', 'Medium', 'High'], "Invalid confidence level")
        self.assertIsInstance(prediction['recommendation'], str, "Invalid recommendation type")
        self.assertEqual(prediction['horizon_weeks'], 4, "Incorrect horizon weeks")
    
    def test_06_historical_validation(self):
        """Test historical validation functionality."""
        test_symbol = 'AAPL'
        if test_symbol not in self.test_data:
            self.skipTest(f"No data available for {test_symbol}")
        
        # Ensure model is trained
        if test_symbol not in self.trader.models:
            data = self.test_data[test_symbol]
            result = self.trader.train_weekly_agent(test_symbol, data)
            if not result.get('success', False):
                self.skipTest(f"Could not train model for {test_symbol}")
        
        # Test historical validation
        test_date = "2023-01-01"
        validation = self.trader.validate_historical_prediction(test_symbol, test_date, weeks_ahead=4)
        
        self.assertIsInstance(validation, dict, "Validation not a dictionary")
        
        if 'error' in validation:
            # This might be expected if the date is not available
            self.assertIsInstance(validation['error'], str, "Error should be a string")
        else:
            # Validate successful validation structure
            required_fields = ['symbol', 'test_date', 'weeks_ahead', 'predicted_signal',
                             'predicted_direction', 'actual_return', 'actual_direction',
                             'prediction_correct', 'test_price', 'outcome_price']
            for field in required_fields:
                self.assertIn(field, validation, f"Validation missing {field}")
            
            # Validate values
            self.assertEqual(validation['symbol'], test_symbol)
            self.assertEqual(validation['weeks_ahead'], 4)
            self.assertIn(validation['predicted_direction'], ['UP', 'DOWN', 'NEUTRAL'])
            self.assertIn(validation['actual_direction'], ['UP', 'DOWN', 'NEUTRAL'])
            self.assertIsInstance(validation['prediction_correct'], bool)
            self.assertGreater(validation['test_price'], 0)
            self.assertGreater(validation['outcome_price'], 0)
    
    def test_07_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test prediction with untrained symbol
        invalid_prediction = self.trader.predict_next_weeks('INVALID_SYMBOL')
        self.assertIn('error', invalid_prediction, "Should return error for untrained symbol")
        
        # Test historical validation with invalid symbol
        invalid_validation = self.trader.validate_historical_prediction('INVALID_SYMBOL', '2023-01-01')
        self.assertIn('error', invalid_validation, "Should return error for invalid symbol")
        
        # Test with invalid date
        if 'AAPL' in self.trader.models:
            invalid_date_validation = self.trader.validate_historical_prediction('AAPL', '1900-01-01')
            self.assertIn('error', invalid_date_validation, "Should return error for invalid date")
    
    def test_08_data_consistency(self):
        """Test data consistency across different operations."""
        test_symbol = 'AAPL'
        if test_symbol not in self.test_data:
            self.skipTest(f"No data available for {test_symbol}")
        
        data = self.test_data[test_symbol]
        
        # Test that features and labels have consistent lengths
        features = self.trader.engineer_features(data)
        labels = self.trader.create_weekly_labels(data, horizon_weeks=4)
        
        # Allow for some difference due to forward-looking labels
        max_len = max(len(features), len(labels))
        min_len = min(len(features), len(labels))
        
        self.assertLessEqual(max_len - min_len, 50, 
                           f"Features ({len(features)}) and labels ({len(labels)}) length mismatch too large")
    
    def test_09_performance_benchmarks(self):
        """Test that the system meets basic performance benchmarks."""
        test_symbol = 'AAPL'
        if test_symbol not in self.test_data:
            self.skipTest(f"No data available for {test_symbol}")
        
        # Train model if needed
        if test_symbol not in self.trader.models:
            data = self.test_data[test_symbol]
            result = self.trader.train_weekly_agent(test_symbol, data)
            if not result.get('success', False):
                self.skipTest(f"Could not train model for {test_symbol}")
        
        # Get performance metrics
        if test_symbol in self.trader.models:
            model_info = self.trader.models[test_symbol]
            if 'performance' in model_info:
                perf = model_info['performance']
                
                # Basic performance checks
                if 'win_rate' in perf:
                    self.assertGreaterEqual(perf['win_rate'], 0.3, 
                                          f"Win rate too low: {perf['win_rate']:.2%}")
                
                if 'total_return' in perf:
                    # Should at least not lose everything
                    self.assertGreater(perf['total_return'], -0.5, 
                                     f"Total return too negative: {perf['total_return']:.2%}")

def run_comprehensive_tests():
    """Run all tests and provide detailed report."""
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE TRADING SYSTEM TESTS")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTradingSystem)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"  • {test}: {error_msg}")
    
    if result.errors:
        print(f"\n🔥 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"  • {test}: {error_msg}")
    
    if not result.failures and not result.errors:
        print("\n✅ ALL TESTS PASSED!")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
