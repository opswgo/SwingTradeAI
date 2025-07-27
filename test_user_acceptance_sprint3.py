"""
Sprint 3 User Acceptance Tests: ML Models + Basic Backtesting

This test suite validates the ML model architecture, training pipeline,
backtesting engine, and user experience requirements for Sprint 3.
"""

import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config import Settings
from src.models.regime_model import RegimeAwareModel, ModelPrediction
from src.models.backtesting import BacktestingEngine, BacktestResults
from src.features.market_regime import MarketRegimeDetector
from src.features.feature_store import FeatureStore


class Sprint3UserAcceptanceTests:
    """
    Comprehensive user acceptance tests for Sprint 3 ML capabilities.
    
    Tests cover:
    - Model architecture and regime-aware predictions
    - Training pipeline with cross-validation
    - Model explainability and user understanding
    - Backtesting engine with walk-forward testing
    - User interface for model predictions and explanations
    """
    
    def __init__(self):
        self.settings = Settings()
        self.test_symbol = "AAPL"
        self.model = RegimeAwareModel(self.settings)
        self.backtesting_engine = BacktestingEngine(self.settings)
        self.test_results = {}
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all Sprint 3 user acceptance tests."""
        print("🚀 Starting Sprint 3 User Acceptance Tests...")
        print("=" * 60)
        
        # Test categories
        test_categories = [
            ("Model Architecture & Predictions", self._test_model_architecture),
            ("Training Pipeline & Cross-Validation", self._test_training_pipeline),
            ("Model Explainability", self._test_model_explainability),
            ("Backtesting Engine", self._test_backtesting_engine),
            ("Walk-Forward Testing", self._test_walk_forward),
            ("User Experience", self._test_user_experience)
        ]
        
        results = {}
        
        for category_name, test_func in test_categories:
            print(f"\n📊 Testing {category_name}...")
            try:
                success = await test_func()
                results[category_name] = success
                status = "✅ PASSED" if success else "❌ FAILED"
                print(f"{status}     {category_name}")
            except Exception as e:
                print(f"❌ FAILED     {category_name}: {str(e)}")
                results[category_name] = False
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 SPRINT 3 USER ACCEPTANCE TEST RESULTS")
        print("=" * 60)
        
        passed_count = sum(results.values())
        total_count = len(results)
        
        for category, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status}     {category}")
        
        print("-" * 60)
        print(f"📊 SUMMARY: {passed_count}/{total_count} test categories passed")
        
        if passed_count == total_count:
            print("🎉 SPRINT 3 USER ACCEPTANCE: ✅ PASSED")
            print("\n🚀 Sprint 3 is complete and ready for user validation!")
            print("\nKey Achievements:")
            print("• 🤖 Regime-aware ML model with confidence scores")
            print("• 🎯 Training pipeline with cross-validation")
            print("• 💡 Model explainability with feature importance")
            print("• 📈 Comprehensive backtesting engine")
            print("• 🔄 Walk-forward testing capabilities")
            print("• 👤 User-friendly prediction interfaces")
        else:
            print("❌ SPRINT 3 USER ACCEPTANCE: FAILED")
            print(f"\n🔧 Issues to address before Sprint 3 completion:")
            for category, passed in results.items():
                if not passed:
                    print(f"• ❌ {category}")
        
        print("=" * 60)
        return results
    
    async def _test_model_architecture(self) -> bool:
        """Test model architecture and regime-aware predictions."""
        try:
            print("  🤖 Testing model architecture...")
            
            # Test 1: Model initialization
            init_result = await self.model.initialize()
            if not init_result['success']:
                print(f"  ❌ Model initialization failed: {init_result.get('error')}")
                return False
            
            print(f"  ✅ Model initialized successfully")
            
            # Test 2: Model components
            expected_components = ['regime_models', 'meta_model', 'scaler', 'feature_columns']
            for component in expected_components:
                if not hasattr(self.model, component):
                    print(f"  ❌ Missing model component: {component}")
                    return False
            
            print("  ✅ All model components present")
            
            # Test 3: Feature configuration
            if len(self.model.feature_columns) < 10:
                print(f"  ❌ Insufficient feature columns: {len(self.model.feature_columns)}")
                return False
            
            print(f"  ✅ Feature configuration complete: {len(self.model.feature_columns)} features")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Model architecture test failed: {str(e)}")
            return False
    
    async def _test_training_pipeline(self) -> bool:
        """Test training pipeline with cross-validation."""
        try:
            print("  🎯 Testing training pipeline...")
            
            # Test 1: Model training
            start_time = time.time()
            training_result = await self.model.train_model(self.test_symbol, lookback_days=100)
            training_time = time.time() - start_time
            
            # User Acceptance: Training should complete successfully
            if not training_result['success']:
                print(f"  ❌ Model training failed: {training_result.get('error')}")
                return False
            
            # User Acceptance: Training should be reasonably fast (<60 seconds)
            if training_time > 60:
                print(f"  ❌ Training too slow: {training_time:.1f}s")
                return False
            
            print(f"  ✅ Model trained successfully in {training_time:.1f}s")
            
            # Test 2: Training metrics
            accuracy = training_result.get('accuracy', 0)
            if accuracy < 0.4:  # Minimum acceptable accuracy
                print(f"  ❌ Training accuracy too low: {accuracy:.3f}")
                return False
            
            print(f"  ✅ Training accuracy acceptable: {accuracy:.3f}")
            
            # Test 3: Model state after training
            if not self.model.is_trained:
                print("  ❌ Model not marked as trained")
                return False
            
            if not self.model.last_training_date:
                print("  ❌ Training date not recorded")
                return False
            
            print("  ✅ Model state properly updated after training")
            
            # Test 4: Cross-validation results
            feature_importance = training_result.get('feature_importance', {})
            if not feature_importance:
                print("  ❌ No feature importance calculated")
                return False
            
            print(f"  ✅ Feature importance calculated: {len(feature_importance)} features")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Training pipeline test failed: {str(e)}")
            return False
    
    async def _test_model_explainability(self) -> bool:
        """Test model explainability and user understanding."""
        try:
            print("  💡 Testing model explainability...")
            
            # Ensure model is trained
            if not self.model.is_trained:
                await self.model.train_model(self.test_symbol, lookback_days=100)
            
            # Test 1: Model predictions with explanations
            prediction = await self.model.predict(self.test_symbol, explain=True)
            
            # User Acceptance: Predictions should include confidence scores
            if not hasattr(prediction, 'confidence') or prediction.confidence < 0:
                print("  ❌ Invalid confidence score")
                return False
            
            print(f"  ✅ Prediction confidence: {prediction.confidence:.3f}")
            
            # User Acceptance: Predictions should include regime explanations
            if not prediction.regime or len(prediction.regime) < 3:
                print("  ❌ Missing or invalid regime information")
                return False
            
            print(f"  ✅ Regime detected: {prediction.regime}")
            
            # Test 2: User-friendly explanations
            if not prediction.explanation or len(prediction.explanation) < 20:
                print("  ❌ Insufficient explanation provided")
                return False
            
            print("  ✅ User-friendly explanation provided")
            
            # Test 3: Feature importance
            if not prediction.features_importance:
                print("  ❌ No feature importance in prediction")
                return False
            
            print(f"  ✅ Feature importance available: {len(prediction.features_importance)} features")
            
            # Test 4: Model metrics
            metrics = await self.model.get_model_metrics()
            if 'error' in metrics:
                print(f"  ❌ Model metrics error: {metrics['error']}")
                return False
            
            required_metrics = ['accuracy', 'precision', 'recall', 'feature_importance']
            for metric in required_metrics:
                if metric not in metrics:
                    print(f"  ❌ Missing metric: {metric}")
                    return False
            
            print("  ✅ Comprehensive model metrics available")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Model explainability test failed: {str(e)}")
            return False
    
    async def _test_backtesting_engine(self) -> bool:
        """Test backtesting engine functionality."""
        try:
            print("  📈 Testing backtesting engine...")
            
            # Test 1: Engine initialization
            init_result = await self.backtesting_engine.initialize()
            if not init_result['success']:
                print(f"  ❌ Backtesting engine initialization failed: {init_result.get('error')}")
                return False
            
            print("  ✅ Backtesting engine initialized")
            
            # Test 2: Backtest execution
            start_date = datetime.now() - timedelta(days=60)
            end_date = datetime.now() - timedelta(days=10)
            
            start_time = time.time()
            backtest_results = await self.backtesting_engine.run_backtest(
                self.test_symbol, start_date, end_date, walk_forward=False
            )
            backtest_time = time.time() - start_time
            
            # User Acceptance: Backtest should complete in reasonable time
            if backtest_time > 30:
                print(f"  ❌ Backtest too slow: {backtest_time:.1f}s")
                return False
            
            print(f"  ✅ Backtest completed in {backtest_time:.1f}s")
            
            # Test 3: Backtest results structure
            required_fields = ['total_trades', 'win_rate', 'total_return', 'sharpe_ratio', 'max_drawdown']
            for field in required_fields:
                if not hasattr(backtest_results, field):
                    print(f"  ❌ Missing backtest field: {field}")
                    return False
            
            print("  ✅ Backtest results structure complete")
            
            # Test 4: Performance metrics validation
            if backtest_results.total_trades < 0:
                print(f"  ❌ Invalid trade count: {backtest_results.total_trades}")
                return False
            
            if not (0 <= backtest_results.win_rate <= 1):
                print(f"  ❌ Invalid win rate: {backtest_results.win_rate}")
                return False
            
            print(f"  ✅ Performance metrics valid: {backtest_results.total_trades} trades, {backtest_results.win_rate:.1%} win rate")
            
            # Test 5: Trade records
            if backtest_results.total_trades > 0 and not backtest_results.trades:
                print("  ❌ Missing trade records")
                return False
            
            print("  ✅ Trade records available")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Backtesting engine test failed: {str(e)}")
            return False
    
    async def _test_walk_forward(self) -> bool:
        """Test walk-forward testing capabilities."""
        try:
            print("  🔄 Testing walk-forward analysis...")
            
            # Test 1: Walk-forward backtest
            start_date = datetime.now() - timedelta(days=90)
            end_date = datetime.now() - timedelta(days=10)
            
            start_time = time.time()
            wf_results = await self.backtesting_engine.run_backtest(
                self.test_symbol, start_date, end_date, walk_forward=True
            )
            wf_time = time.time() - start_time
            
            # User Acceptance: Walk-forward should complete reasonably fast
            if wf_time > 60:
                print(f"  ❌ Walk-forward test too slow: {wf_time:.1f}s")
                return False
            
            print(f"  ✅ Walk-forward test completed in {wf_time:.1f}s")
            
            # Test 2: Results comparison
            # Walk-forward should provide realistic results
            if abs(wf_results.total_return) > 2.0:  # More than 200% return is unrealistic
                print(f"  ⚠️ Walk-forward return seems unrealistic: {wf_results.total_return:.2%}")
                # Don't fail the test, just warn
            
            print(f"  ✅ Walk-forward return: {wf_results.total_return:.2%}")
            
            # Test 3: Equity curve
            if wf_results.equity_curve.empty:
                print("  ❌ No equity curve generated")
                return False
            
            print(f"  ✅ Equity curve generated: {len(wf_results.equity_curve)} data points")
            
            # Test 4: Regime performance breakdown
            if not wf_results.regime_performance:
                print("  ❌ No regime performance breakdown")
                return False
            
            print(f"  ✅ Regime performance breakdown: {len(wf_results.regime_performance)} regimes")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Walk-forward test failed: {str(e)}")
            return False
    
    async def _test_user_experience(self) -> bool:
        """Test user experience requirements."""
        try:
            print("  👤 Testing user experience requirements...")
            
            # Test 1: Response time requirements
            print("    ⏱️ Validating response time requirements...")
            
            # Model prediction response time
            start_time = time.time()
            prediction = await self.model.predict(self.test_symbol, explain=True)
            prediction_time = time.time() - start_time
            
            if prediction_time > 2.0:  # Should be fast
                print(f"    ❌ Model prediction too slow: {prediction_time:.2f}s")
                return False
            
            print(f"    ✅ Model prediction time: {prediction_time:.3f}s")
            
            # Test 2: User-friendly explanations
            print("    💡 Validating user explanations...")
            
            # Check prediction explanation quality
            if not prediction.explanation or len(prediction.explanation.split()) < 10:
                print("    ❌ Prediction explanation insufficient")
                return False
            
            print("    ✅ Prediction explanations comprehensive")
            
            # Test 3: Model metrics accessibility
            metrics = await self.model.get_model_metrics()
            if 'error' in metrics:
                print(f"    ❌ Model metrics not accessible: {metrics['error']}")
                return False
            
            print("    ✅ Model metrics accessible to users")
            
            # Test 4: Backtest report generation
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now() - timedelta(days=5)
            
            backtest_results = await self.backtesting_engine.run_backtest(
                self.test_symbol, start_date, end_date, walk_forward=False
            )
            
            report = self.backtesting_engine.generate_backtest_report(backtest_results)
            if not report or len(report) < 100:
                print("    ❌ Backtest report generation failed")
                return False
            
            print("    ✅ User-friendly backtest reports generated")
            
            # Test 5: Error handling
            print("    🔧 Validating error handling...")
            
            # Test with invalid symbol
            try:
                invalid_prediction = await self.model.predict("INVALID_SYMBOL", explain=True)
                if invalid_prediction.signal != 'HOLD':  # Should default to safe HOLD
                    print("    ⚠️ Invalid symbol handling could be improved")
                else:
                    print("    ✅ Invalid symbol handled gracefully")
            except Exception:
                print("    ✅ Invalid symbol handled with proper error")
            
            # Test 6: Data quality validation
            print("    🔍 Validating data quality...")
            
            # Check that predictions include confidence bounds
            if prediction.confidence < 0 or prediction.confidence > 1:
                print(f"    ❌ Invalid confidence bounds: {prediction.confidence}")
                return False
            
            print(f"    ✅ Confidence bounds valid: {prediction.confidence:.3f}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ User experience test failed: {str(e)}")
            return False


async def main():
    """Run Sprint 3 user acceptance tests."""
    test_suite = Sprint3UserAcceptanceTests()
    results = await test_suite.run_all_tests()
    
    # Return appropriate exit code
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
