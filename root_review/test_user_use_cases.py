#!/usr/bin/env python3
"""
Comprehensive User Use Case Tests
Validates ALL user workflows with real data only - no synthetic/mock data.
"""

import unittest
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os
import time
import requests

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robust_trading_ui import RobustTradingSystem

class TestUserUseCases(unittest.TestCase):
    """Comprehensive tests for ALL user use cases with real data only."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with real data."""
        cls.system = RobustTradingSystem()
        cls.test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        cls.ui_url = "http://localhost:8504"
        
        print("\n🔄 Setting up user use case tests with real data...")
        
        # Validate real data availability
        cls.real_data = {}
        for symbol in cls.test_symbols:
            try:
                data = cls.system.fetch_data_safely(symbol, "2y")
                if len(data) >= 500:
                    cls.real_data[symbol] = data
                    print(f"✅ {symbol}: {len(data)} days of real data")
                else:
                    print(f"❌ {symbol}: Insufficient real data ({len(data)} days)")
            except Exception as e:
                print(f"❌ {symbol}: Real data fetch failed - {e}")
        
        # Check if UI is running
        try:
            response = requests.get(cls.ui_url, timeout=5)
            cls.ui_available = response.status_code == 200
            print(f"✅ UI available at {cls.ui_url}")
        except Exception:
            cls.ui_available = False
            print(f"❌ UI not available at {cls.ui_url}")
    
    def test_01_user_case_train_model_real_data(self):
        """USER CASE: Train a model with real market data."""
        print("\n📋 USER CASE: Training model with real data")
        
        test_symbol = 'AAPL'
        self.assertIn(test_symbol, self.real_data, f"No real data for {test_symbol}")
        
        # User trains model
        result = self.system.train_model(test_symbol, horizon_weeks=4)
        
        # Validate training success with real data
        self.assertTrue(result.get('success', False), f"Training failed: {result.get('error', 'Unknown')}")
        self.assertIn(test_symbol, self.system.trader.models, f"Model not saved for {test_symbol}")
        
        # Validate model uses real data
        model_data = self.system.trader.models[test_symbol]
        self.assertIn('performance', model_data, "No performance metrics")
        self.assertIn('training_period', model_data, "No training period recorded")
        
        # Validate performance metrics are realistic (not synthetic)
        perf = model_data['performance']
        self.assertIsInstance(perf.get('win_rate', 0), (int, float), "Invalid win rate type")
        self.assertIsInstance(perf.get('total_return', 0), (int, float), "Invalid return type")
        self.assertGreaterEqual(perf.get('win_rate', 0), 0, "Win rate cannot be negative")
        self.assertLessEqual(perf.get('win_rate', 0), 1, "Win rate cannot exceed 100%")
        
        print(f"✅ Model trained successfully with real data: {perf}")
    
    def test_02_user_case_model_persistence_cross_session(self):
        """USER CASE: Models persist between sessions and appear in all tabs."""
        print("\n📋 USER CASE: Model persistence across sessions")
        
        test_symbol = 'MSFT'
        
        # Train model
        if test_symbol not in self.system.trader.models:
            result = self.system.train_model(test_symbol, horizon_weeks=4)
            self.assertTrue(result.get('success', False), "Training failed")
        
        # Simulate session restart
        original_models = dict(self.system.trader.models)
        self.system.trader.models = {}  # Clear in-memory models
        
        # Load models from persistence
        loaded_count = self.system.load_all_models()
        
        # Validate model persistence
        self.assertGreater(loaded_count, 0, "No models loaded from persistence")
        self.assertIn(test_symbol, self.system.trader.models, f"Model not persisted for {test_symbol}")
        
        # Validate model data integrity
        loaded_model = self.system.trader.models[test_symbol]
        self.assertIn('saved_at', loaded_model, "No save timestamp")
        self.assertIn('performance', loaded_model, "Performance data lost")
        
        print(f"✅ Model persisted and loaded successfully: {loaded_count} models")
    
    def test_03_user_case_model_quality_filtering(self):
        """USER CASE: System only offers acceptable models, filters out poor performers."""
        print("\n📋 USER CASE: Model quality filtering")
        
        # Ensure we have at least one model
        test_symbol = 'GOOGL'
        if test_symbol not in self.system.trader.models:
            result = self.system.train_model(test_symbol, horizon_weeks=4)
            self.assertTrue(result.get('success', False), "Training failed")
        
        # Test model acceptability
        for symbol in self.system.trader.models.keys():
            acceptable, reason = self.system.is_model_acceptable(symbol)
            
            if acceptable:
                # Validate acceptable models meet criteria
                model_data = self.system.trader.models[symbol]
                perf = model_data.get('performance', {})
                
                win_rate = perf.get('win_rate', 0)
                total_return = perf.get('total_return', 0)
                
                self.assertGreaterEqual(win_rate, self.system.min_win_rate, 
                                      f"Acceptable model {symbol} has low win rate: {win_rate:.1%}")
                self.assertGreaterEqual(total_return, self.system.min_return,
                                      f"Acceptable model {symbol} has low return: {total_return:.1%}")
                
                print(f"✅ {symbol} meets quality criteria: {win_rate:.1%} win rate, {total_return:.1%} return")
            else:
                print(f"⚠️ {symbol} filtered out: {reason}")
        
        print("✅ Model quality filtering working correctly")
    
    def test_04_user_case_8_week_prediction_chart(self):
        """USER CASE: Generate 8-week prediction chart with confidence bands."""
        print("\n📋 USER CASE: 8-week prediction chart generation")
        
        # Find an acceptable model
        acceptable_models = [s for s in self.system.trader.models.keys() 
                           if self.system.is_model_acceptable(s)[0]]
        
        if not acceptable_models:
            # Train one if needed
            result = self.system.train_model('AAPL', horizon_weeks=4)
            self.assertTrue(result.get('success', False), "Training failed")
            acceptable_models = ['AAPL']
        
        test_symbol = acceptable_models[0]
        
        # Generate prediction chart
        fig, result = self.system.create_8_week_prediction_chart(test_symbol)
        
        # Validate chart generation
        self.assertIsNotNone(fig, f"Chart generation failed: {result.get('error', 'Unknown')}")
        self.assertIn('predictions', result, "No predictions in result")
        self.assertIn('current_price', result, "No current price in result")
        
        # Validate prediction data
        predictions = result['predictions']
        self.assertEqual(len(predictions), 8, "Should have 8 weeks of predictions")
        
        for i, pred in enumerate(predictions):
            self.assertEqual(pred['week'], i + 1, f"Week numbering incorrect: {pred['week']}")
            self.assertGreater(pred['predicted_price'], 0, f"Invalid predicted price: {pred['predicted_price']}")
            self.assertGreater(pred['upper_band'], pred['predicted_price'], "Upper band should be above prediction")
            self.assertLess(pred['lower_band'], pred['predicted_price'], "Lower band should be below prediction")
            self.assertIn('signal', pred, "Missing signal data")
            self.assertIn('confidence', pred, "Missing confidence data")
        
        current_price = result['current_price']
        self.assertGreater(current_price, 0, "Invalid current price")
        
        print(f"✅ 8-week prediction chart generated successfully for {test_symbol}")
        print(f"   Current price: ${current_price:.2f}")
        print(f"   Week 8 prediction: ${predictions[7]['predicted_price']:.2f}")
    
    def test_05_user_case_historical_validation_real_data(self):
        """USER CASE: Historical validation with real data fetch and 8-week comparison."""
        print("\n📋 USER CASE: Historical validation with real data")
        
        # Find an acceptable model
        acceptable_models = [s for s in self.system.trader.models.keys() 
                           if self.system.is_model_acceptable(s)[0]]
        
        if not acceptable_models:
            result = self.system.train_model('AAPL', horizon_weeks=4)
            self.assertTrue(result.get('success', False), "Training failed")
            acceptable_models = ['AAPL']
        
        test_symbol = acceptable_models[0]
        test_date = "2023-01-01"
        
        # Run historical validation
        fig, result = self.system.create_validation_chart(test_symbol, test_date, weeks_ahead=8)
        
        # Validate validation success
        self.assertIsNotNone(fig, f"Validation failed: {result.get('error', 'Unknown')}")
        self.assertIn('mae', result, "No accuracy metrics")
        self.assertIn('mape', result, "No percentage error metrics")
        self.assertIn('weeks_validated', result, "No weeks validated count")
        
        # Validate real data was fetched and used
        weeks_validated = result['weeks_validated']
        self.assertGreater(weeks_validated, 0, "No weeks validated")
        self.assertLessEqual(weeks_validated, 8, "Too many weeks validated")
        
        # Validate accuracy metrics are realistic
        mae = result['mae']
        mape = result['mape']
        self.assertGreater(mae, 0, "MAE should be positive")
        self.assertGreater(mape, 0, "MAPE should be positive")
        self.assertLess(mape, 1000, "MAPE too high - likely calculation error")
        
        print(f"✅ Historical validation completed for {test_symbol}")
        print(f"   Test date: {test_date}")
        print(f"   Weeks validated: {weeks_validated}")
        print(f"   MAE: ${mae:.2f}, MAPE: {mape:.1f}%")
    
    def test_06_user_case_model_update_outdated_detection(self):
        """USER CASE: Detect outdated models and update them."""
        print("\n📋 USER CASE: Model update and outdated detection")
        
        test_symbol = 'AAPL'
        
        # Ensure we have a model
        if test_symbol not in self.system.trader.models:
            result = self.system.train_model(test_symbol, horizon_weeks=4)
            self.assertTrue(result.get('success', False), "Training failed")
        
        # Test outdated detection
        outdated, age_info = self.system.is_model_outdated(test_symbol)
        
        # Validate age detection works
        self.assertIsInstance(outdated, bool, "Outdated detection should return boolean")
        self.assertIsInstance(age_info, str, "Age info should be string")
        
        # If model is fresh, it shouldn't be outdated
        model_data = self.system.trader.models[test_symbol]
        saved_at = model_data.get('saved_at', '')
        
        if saved_at:
            try:
                saved_time = datetime.fromisoformat(saved_at)
                age_hours = (datetime.now() - saved_time).total_seconds() / 3600
                
                if age_hours < 1:  # Less than 1 hour old
                    self.assertFalse(outdated, f"Fresh model marked as outdated: {age_info}")
                
                print(f"✅ Model age detection working: {age_info}")
            except Exception as e:
                self.fail(f"Invalid save date format: {saved_at}")
        
        # Test model update
        original_save_time = saved_at
        result = self.system.train_model(test_symbol, horizon_weeks=4)
        
        if result.get('success', False):
            new_model_data = self.system.trader.models[test_symbol]
            new_save_time = new_model_data.get('saved_at', '')
            
            self.assertNotEqual(original_save_time, new_save_time, "Model save time should update")
            print(f"✅ Model updated successfully: {original_save_time} -> {new_save_time}")
    
    def test_07_user_case_performance_table_unified_view(self):
        """USER CASE: View unified performance table with all models."""
        print("\n📋 USER CASE: Unified performance table")
        
        # Ensure we have multiple models
        required_symbols = ['AAPL', 'MSFT']
        for symbol in required_symbols:
            if symbol not in self.system.trader.models:
                result = self.system.train_model(symbol, horizon_weeks=4)
                self.assertTrue(result.get('success', False), f"Training failed for {symbol}")
        
        # Generate performance table data
        available_models = list(self.system.trader.models.keys())
        self.assertGreater(len(available_models), 0, "No models available for performance table")
        
        perf_data = []
        for symbol in available_models:
            model_data = self.system.trader.models[symbol]
            perf = model_data.get('performance', {})
            
            acceptable, reason = self.system.is_model_acceptable(symbol)
            outdated, age_info = self.system.is_model_outdated(symbol)
            
            row = {
                'Symbol': symbol,
                'Win Rate': perf.get('win_rate', 0),
                'Total Return': perf.get('total_return', 0),
                'Buy & Hold': perf.get('buy_hold_return', 0),
                'Total Trades': perf.get('total_trades', 0),
                'Status': 'Good' if acceptable and not outdated else 'Poor',
                'Issues': reason if not acceptable else age_info if outdated else 'None'
            }
            perf_data.append(row)
        
        # Validate performance table data
        self.assertGreater(len(perf_data), 0, "No performance data generated")
        
        for row in perf_data:
            # Validate data types and ranges
            self.assertIsInstance(row['Win Rate'], (int, float), f"Invalid win rate for {row['Symbol']}")
            self.assertIsInstance(row['Total Return'], (int, float), f"Invalid return for {row['Symbol']}")
            self.assertIsInstance(row['Total Trades'], (int, float), f"Invalid trade count for {row['Symbol']}")
            
            self.assertGreaterEqual(row['Win Rate'], 0, f"Negative win rate for {row['Symbol']}")
            self.assertLessEqual(row['Win Rate'], 1, f"Win rate > 100% for {row['Symbol']}")
            self.assertGreaterEqual(row['Total Trades'], 0, f"Negative trade count for {row['Symbol']}")
        
        # Calculate summary stats
        good_models = [row for row in perf_data if row['Status'] == 'Good']
        success_rate = len(good_models) / len(perf_data) if perf_data else 0
        
        print(f"✅ Performance table generated successfully")
        print(f"   Total models: {len(perf_data)}")
        print(f"   Good models: {len(good_models)}")
        print(f"   Success rate: {success_rate:.0%}")
        
        for row in perf_data:
            print(f"   {row['Symbol']}: {row['Win Rate']:.1%} win rate, {row['Total Return']:.1%} return, {row['Status']}")
    
    def test_08_user_case_meaningful_metrics_interpretation(self):
        """USER CASE: Metrics are interpretable and actionable for users."""
        print("\n📋 USER CASE: Meaningful metrics interpretation")
        
        # Ensure we have a model
        test_symbol = 'AAPL'
        if test_symbol not in self.system.trader.models:
            result = self.system.train_model(test_symbol, horizon_weeks=4)
            self.assertTrue(result.get('success', False), "Training failed")
        
        # Test prediction with interpretable metrics
        prediction = self.system.trader.predict_next_weeks(test_symbol, weeks_ahead=4)
        
        self.assertNotIn('error', prediction, f"Prediction failed: {prediction.get('error', '')}")
        
        # Validate interpretable metrics
        signal = prediction.get('prediction_signal', 0)
        confidence = prediction.get('confidence', '')
        recommendation = prediction.get('recommendation', '')
        
        # Signal should be in reasonable range
        self.assertIsInstance(signal, (int, float), "Signal should be numeric")
        self.assertGreaterEqual(signal, -2, "Signal too negative")
        self.assertLessEqual(signal, 2, "Signal too positive")
        
        # Confidence should be interpretable
        self.assertIn(confidence, ['Low', 'Medium', 'High'], f"Invalid confidence: {confidence}")
        
        # Recommendation should be actionable
        self.assertIsInstance(recommendation, str, "Recommendation should be string")
        self.assertGreater(len(recommendation), 10, "Recommendation too short")
        
        # Test signal strength interpretation
        if abs(signal) < 0.3:
            expected_action = "HOLD"
        elif signal > 0.3:
            expected_action = "BUY"
        else:
            expected_action = "SELL"
        
        self.assertIn(expected_action, recommendation, f"Recommendation doesn't match signal: {signal} -> {recommendation}")
        
        print(f"✅ Metrics are interpretable and actionable")
        print(f"   Signal: {signal:.3f}")
        print(f"   Confidence: {confidence}")
        print(f"   Recommendation: {recommendation}")
    
    def test_09_user_case_real_data_validation_no_synthetic(self):
        """USER CASE: System uses only real data, no synthetic/mock data."""
        print("\n📋 USER CASE: Real data validation - no synthetic data")
        
        test_symbol = 'MSFT'
        
        # Test data fetching
        data = self.system.fetch_data_safely(test_symbol, "1y")
        
        self.assertGreater(len(data), 0, "No data fetched")
        
        # Validate data is real market data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            self.assertIn(col, data.columns, f"Missing required column: {col}")
            self.assertFalse(data[col].isna().all(), f"All {col} values are NaN")
        
        # Validate realistic market data characteristics
        prices = data['Close']
        returns = prices.pct_change().dropna()
        
        # Returns should be realistic (not synthetic patterns)
        self.assertGreater(returns.std(), 0.001, "Returns too stable - likely synthetic")
        self.assertLess(returns.std(), 0.5, "Returns too volatile - likely corrupted")
        
        # Prices should be positive and reasonable
        self.assertTrue((prices > 0).all(), "Negative prices detected")
        self.assertLess(prices.max() / prices.min(), 100, "Price range too extreme")
        
        # Volume should be realistic
        volumes = data['Volume']
        self.assertTrue((volumes >= 0).all(), "Negative volume detected")
        self.assertGreater(volumes.mean(), 1000, "Volume too low - likely synthetic")
        
        print(f"✅ Real data validation passed for {test_symbol}")
        print(f"   Data points: {len(data)}")
        print(f"   Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        print(f"   Return volatility: {returns.std():.3f}")
        print(f"   Average volume: {volumes.mean():,.0f}")
    
    def test_10_user_case_ui_integration_workflow(self):
        """USER CASE: Complete UI workflow integration test."""
        print("\n📋 USER CASE: UI integration workflow")
        
        # This test would require Selenium WebDriver setup
        # For now, just validate UI is accessible
        try:
            response = requests.get(self.ui_url, timeout=10)
            self.assertEqual(response.status_code, 200, "UI not accessible")
            
            # Validate UI contains expected elements
            content = response.text.lower()
            expected_elements = ['training', 'predictions', 'validation', 'performance']
            
            for element in expected_elements:
                self.assertIn(element, content, f"UI missing {element} section")
            
            print(f"✅ UI integration test passed")
            print(f"   URL: {self.ui_url}")
            print(f"   Status: {response.status_code}")
            print(f"   Contains all expected sections")
            
        except Exception as e:
            self.fail(f"UI integration test failed: {e}")

def run_user_use_case_tests():
    """Run all user use case tests with comprehensive reporting."""
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE USER USE CASE TESTS")
    print("="*80)
    print("Testing ALL user workflows with real data only")
    print("No synthetic/mock data allowed - professional standards")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUserUseCases)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("📊 USER USE CASE TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
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
        print("\n✅ ALL USER USE CASES PASS!")
        print("🎯 System ready for production use")
    else:
        print("\n⚠️ SOME USER USE CASES FAILED")
        print("🔧 Fix required before production deployment")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_user_use_case_tests()
    sys.exit(0 if success else 1)
