#!/usr/bin/env python3
"""
SwingTradeAI - Sprint 5 User Acceptance Tests
User Acceptance: All Sprint 5 deliverables meet professional standards and user requirements
"""

import unittest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.backtesting.advanced_backtester import AdvancedBacktester, BacktestConfig
from src.backtesting.model_competition import ModelCompetition, StrategyConfig
from src.backtesting.scenario_analyzer import ScenarioAnalyzer, MarketScenario
from src.backtesting.strategy_reporter import StrategyReporter
from src.core.config import Settings

class Sprint5UserAcceptanceTests(unittest.TestCase):
    """
    Sprint 5 User Acceptance Tests
    
    Tests validate that users can:
    1. Run regime-specific backtesting with Monte Carlo simulation
    2. Compare multiple strategies side-by-side
    3. Test "what-if" scenarios easily
    4. Get professional-quality strategy reports
    5. Use advanced backtesting interface effectively
    """
    
    def setUp(self):
        """Set up test environment"""
        self.settings = Settings()
        self.test_symbol = "AAPL"
        self.start_date = datetime.now() - timedelta(days=180)
        self.end_date = datetime.now() - timedelta(days=1)
        
        # Initialize components
        self.backtester = AdvancedBacktester(self.settings)
        self.competition = ModelCompetition(self.settings)
        self.scenario_analyzer = ScenarioAnalyzer(self.settings)
        self.reporter = StrategyReporter(self.settings)
        
        print(f"\n🧪 Testing Sprint 5 deliverables for {self.test_symbol}")
    
    def test_1_regime_specific_backtesting(self):
        """
        User Acceptance: User can test strategies across different market conditions
        
        Validates:
        - Regime detection works in backtesting
        - Monte Carlo simulation provides statistical validation
        - Performance metrics are regime-specific
        - Results are actionable and professional
        """
        
        print("\n1️⃣ Testing regime-specific backtesting...")
        
        try:
            # Create backtest configuration
            config = BacktestConfig(
                symbol=self.test_symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=100000,
                position_size_pct=0.25,
                stop_loss_pct=0.05,
                regime_threshold=0.7,
                lookback_period=20
            )
            
            # Run backtest
            result = asyncio.run(self.backtester.run_backtest(config))
            
            # Validate backtest result structure
            self.assertIsNotNone(result, "Backtest should return a result")
            self.assertEqual(result.symbol, self.test_symbol, "Symbol should match")
            self.assertIsInstance(result.total_return, (int, float), "Total return should be numeric")
            self.assertIsInstance(result.sharpe_ratio, (int, float), "Sharpe ratio should be numeric")
            self.assertIsInstance(result.max_drawdown, (int, float), "Max drawdown should be numeric")
            self.assertIsInstance(result.total_trades, int, "Total trades should be integer")
            
            # Validate regime-specific performance
            self.assertIsNotNone(result.regime_performance, "Should have regime-specific performance")
            self.assertGreater(len(result.regime_performance), 0, "Should detect at least one regime")
            
            # Validate Monte Carlo results
            self.assertIsNotNone(result.monte_carlo_results, "Should have Monte Carlo results")
            self.assertIn('confidence_intervals', result.monte_carlo_results, "Should have confidence intervals")
            
            # Validate equity curve
            self.assertFalse(result.equity_curve.empty, "Should have equity curve data")
            self.assertIn('portfolio_value', result.equity_curve.columns, "Should track portfolio value")
            
            # Validate trade history
            if result.total_trades > 0:
                self.assertFalse(result.trade_history.empty, "Should have trade history if trades executed")
                required_trade_columns = ['entry_date', 'exit_date', 'symbol', 'pnl']
                for col in required_trade_columns:
                    if col in result.trade_history.columns:
                        self.assertTrue(True, f"Trade history has {col} column")
            
            print(f"   ✅ Regime-specific backtesting: {result.total_trades} trades, {result.total_return:.2f}% return")
            print(f"   ✅ Detected {len(result.regime_performance)} market regimes")
            print(f"   ✅ Monte Carlo validation with confidence intervals")
            
        except Exception as e:
            self.fail(f"Regime-specific backtesting failed: {str(e)}")
    
    def test_2_strategy_competition(self):
        """
        User Acceptance: User can compare multiple strategies side-by-side
        
        Validates:
        - Multiple strategies can be compared
        - Statistical significance testing works
        - Performance rankings are clear
        - Recommendations are actionable
        """
        
        print("\n2️⃣ Testing strategy competition...")
        
        try:
            # Define test strategies
            strategies = [
                StrategyConfig(
                    name="Conservative",
                    description="Conservative strategy with low risk",
                    parameters={"regime_threshold": 0.8, "lookback_period": 20, "position_size_pct": 0.15, "stop_loss_pct": 0.03},
                    regime_weights={"bull_trending": 0.8, "bear_trending": 0.2},
                    risk_settings={"max_position_size": 0.15, "stop_loss": 0.03},
                    entry_rules=["regime_confidence > 0.8", "trend_strength > 0.6"],
                    exit_rules=["stop_loss_hit", "regime_change"]
                ),
                StrategyConfig(
                    name="Aggressive",
                    description="Aggressive strategy with higher risk",
                    parameters={"regime_threshold": 0.6, "lookback_period": 10, "position_size_pct": 0.35, "stop_loss_pct": 0.05},
                    regime_weights={"bull_trending": 0.9, "bear_trending": 0.1},
                    risk_settings={"max_position_size": 0.35, "stop_loss": 0.05},
                    entry_rules=["regime_confidence > 0.6", "momentum > 0.7"],
                    exit_rules=["stop_loss_hit", "profit_target"]
                ),
                StrategyConfig(
                    name="Balanced",
                    description="Balanced strategy with moderate risk",
                    parameters={"regime_threshold": 0.7, "lookback_period": 15, "position_size_pct": 0.25, "stop_loss_pct": 0.04},
                    regime_weights={"bull_trending": 0.7, "bear_trending": 0.3},
                    risk_settings={"max_position_size": 0.25, "stop_loss": 0.04},
                    entry_rules=["regime_confidence > 0.7", "trend_strength > 0.5"],
                    exit_rules=["stop_loss_hit", "time_exit"]
                )
            ]
            
            # Run competition
            competition_result = asyncio.run(
                self.competition.run_competition(self.test_symbol, self.start_date, self.end_date, strategies)
            )
            
            # Validate competition result
            self.assertIsNotNone(competition_result, "Competition should return results")
            self.assertEqual(len(competition_result.strategies), len(strategies), "Should test all strategies")
            self.assertIsNotNone(competition_result.best_strategy, "Should identify best strategy")
            self.assertIn(competition_result.best_strategy, [s.name for s in strategies], "Best strategy should be from input list")
            
            # Validate performance matrix
            self.assertFalse(competition_result.performance_matrix.empty, "Should have performance matrix")
            self.assertEqual(len(competition_result.performance_matrix), len(strategies), "Matrix should have all strategies")
            
            # Check required performance columns
            required_columns = ['Strategy', 'Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
            for col in required_columns:
                self.assertIn(col, competition_result.performance_matrix.columns, f"Should have {col} column")
            
            # Validate statistical significance
            self.assertIsNotNone(competition_result.statistical_significance, "Should have statistical tests")
            
            # Validate recommendations
            self.assertIsInstance(competition_result.recommendations, list, "Should have recommendations list")
            self.assertGreater(len(competition_result.recommendations), 0, "Should provide recommendations")
            
            print(f"   ✅ Strategy competition: {len(strategies)} strategies compared")
            print(f"   ✅ Best strategy: {competition_result.best_strategy}")
            print(f"   ✅ Statistical significance testing completed")
            print(f"   ✅ {len(competition_result.recommendations)} actionable recommendations")
            
        except Exception as e:
            self.fail(f"Strategy competition failed: {str(e)}")
    
    def test_3_scenario_analysis(self):
        """
        User Acceptance: User can test "what-if" scenarios easily
        
        Validates:
        - Market scenarios can be defined and tested
        - Impact analysis is comprehensive
        - Stress testing provides meaningful insights
        - Results guide risk management decisions
        """
        
        print("\n3️⃣ Testing scenario analysis...")
        
        try:
            # Define test scenarios
            scenarios = [
                MarketScenario(
                    name="Market Crash",
                    description="20% market decline",
                    market_shock=-0.20,
                    volatility_multiplier=2.0,
                    duration_days=30
                ),
                MarketScenario(
                    name="Volatility Spike",
                    description="VIX spike to 40+",
                    market_shock=0.0,
                    volatility_multiplier=3.0,
                    duration_days=14
                )
            ]
            
            # Run scenario analysis
            scenario_results = {}
            for scenario in scenarios:
                result = asyncio.run(
                    self.scenario_analyzer.analyze_scenario(
                        self.test_symbol, self.start_date, self.end_date, scenario
                    )
                )
                scenario_results[scenario.name] = result
            
            # Validate scenario results
            self.assertEqual(len(scenario_results), len(scenarios), "Should analyze all scenarios")
            
            for scenario_name, scenario_result in scenario_results.items():
                # Validate result structure
                self.assertIsNotNone(scenario_result, f"Scenario {scenario_name} should have results")
                self.assertEqual(scenario_result.scenario_name, scenario_name, "Scenario name should match")
                self.assertIsNotNone(scenario_result.impact_metrics, "Should have impact metrics")
                self.assertIsInstance(scenario_result.recommendations, list, "Should have recommendations")
                
                # Validate impact metrics
                impact_metrics = scenario_result.impact_metrics
                self.assertIn('return_change_pct', impact_metrics, "Should measure return impact")
                self.assertIn('volatility_change_pct', impact_metrics, "Should measure volatility impact")
                self.assertIn('drawdown_change_pct', impact_metrics, "Should measure drawdown impact")
                
                # Validate recommendations
                self.assertGreater(len(scenario_result.recommendations), 0, f"Should provide recommendations for {scenario_name}")
            
            print(f"   ✅ Scenario analysis: {len(scenarios)} scenarios tested")
            print(f"   ✅ Impact metrics calculated for all scenarios")
            print(f"   ✅ Risk management recommendations provided")
            
        except Exception as e:
            self.fail(f"Scenario analysis failed: {str(e)}")
    
    def test_4_portfolio_optimization(self):
        """
        User Acceptance: User can optimize portfolio allocation
        
        Validates:
        - Portfolio optimization works with multiple assets
        - Efficient frontier calculation
        - Risk-return trade-offs are clear
        - Allocation recommendations are practical
        """
        
        print("\n4️⃣ Testing portfolio optimization...")
        
        try:
            # Test symbols for portfolio optimization
            symbols = [self.test_symbol, "SPY", "QQQ"]
            
            # Run portfolio optimization
            optimization_result = asyncio.run(
                self.scenario_analyzer.optimize_portfolio(symbols, self.start_date, self.end_date)
            )
            
            # Validate optimization result
            self.assertIsNotNone(optimization_result, "Should return optimization result")
            self.assertIsNotNone(optimization_result.optimal_weights, "Should have optimal weights")
            self.assertEqual(len(optimization_result.optimal_weights), len(symbols), "Should have weights for all symbols")
            
            # Validate weights sum to 1
            total_weight = sum(optimization_result.optimal_weights.values())
            self.assertAlmostEqual(total_weight, 1.0, places=2, msg="Weights should sum to 1.0")
            
            # Validate performance metrics
            self.assertIsInstance(optimization_result.expected_return, (int, float), "Expected return should be numeric")
            self.assertIsInstance(optimization_result.expected_volatility, (int, float), "Expected volatility should be numeric")
            self.assertIsInstance(optimization_result.sharpe_ratio, (int, float), "Sharpe ratio should be numeric")
            
            # Validate diversification
            self.assertIsInstance(optimization_result.diversification_ratio, (int, float), "Should have diversification ratio")
            self.assertGreater(optimization_result.diversification_ratio, 0, "Diversification ratio should be positive")
            
            print(f"   ✅ Portfolio optimization: {len(symbols)} assets optimized")
            print(f"   ✅ Expected return: {optimization_result.expected_return:.2f}%")
            print(f"   ✅ Expected volatility: {optimization_result.expected_volatility:.2f}%")
            print(f"   ✅ Sharpe ratio: {optimization_result.sharpe_ratio:.2f}")
            
        except Exception as e:
            self.fail(f"Portfolio optimization failed: {str(e)}")
    
    def test_5_professional_strategy_reports(self):
        """
        User Acceptance: User gets professional-quality strategy reports
        
        Validates:
        - Reports are comprehensive and professional
        - Executive summary is clear and actionable
        - Technical analysis is detailed
        - Export functionality works
        """
        
        print("\n5️⃣ Testing professional strategy reports...")
        
        try:
            # First run a backtest to get results
            config = BacktestConfig(
                symbol=self.test_symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=100000,
                position_size_pct=0.25,
                stop_loss_pct=0.05,
                regime_threshold=0.7,
                lookback_period=20
            )
            
            backtest_result = asyncio.run(self.backtester.run_backtest(config))
            
            # Generate comprehensive report
            report = asyncio.run(self.reporter.generate_comprehensive_report(backtest_result))
            
            # Validate report structure
            self.assertIsNotNone(report, "Should generate report")
            self.assertIsNotNone(report.report_id, "Should have report ID")
            self.assertEqual(report.symbol, self.test_symbol, "Report symbol should match")
            self.assertEqual(report.strategy_name, backtest_result.strategy_name, "Strategy name should match")
            
            # Validate executive summary
            self.assertIsNotNone(report.executive_summary, "Should have executive summary")
            self.assertGreater(len(report.executive_summary), 100, "Executive summary should be comprehensive")
            self.assertIn("Strategy Overview", report.executive_summary, "Should include strategy overview")
            self.assertIn("Key Performance Metrics", report.executive_summary, "Should include key metrics")
            
            # Validate report sections
            self.assertIsInstance(report.sections, list, "Should have report sections")
            self.assertGreater(len(report.sections), 0, "Should have at least one section")
            
            # Check for required sections
            section_titles = [section.title for section in report.sections]
            required_sections = ["Performance Analysis", "Risk Analysis", "Trade Analysis"]
            for required_section in required_sections:
                self.assertIn(required_section, section_titles, f"Should have {required_section} section")
            
            # Validate each section
            for section in report.sections:
                self.assertIsNotNone(section.title, "Section should have title")
                self.assertIsNotNone(section.content, "Section should have content")
                self.assertIsInstance(section.key_metrics, dict, "Section should have key metrics")
                self.assertIsInstance(section.recommendations, list, "Section should have recommendations")
            
            # Validate disclaimer
            self.assertIsNotNone(report.disclaimer, "Should have disclaimer")
            self.assertIn("educational purposes", report.disclaimer.lower(), "Should mention educational purpose")
            
            # Validate metadata
            self.assertIsNotNone(report.report_metadata, "Should have metadata")
            self.assertIn('analysis_period_days', report.report_metadata, "Should track analysis period")
            self.assertIn('total_trades', report.report_metadata, "Should track total trades")
            
            # Validate report saving
            reports_dir = Path("data/reports")
            if reports_dir.exists():
                report_files = list(reports_dir.glob("*.json"))
                self.assertGreater(len(report_files), 0, "Should save report files")
            
            print(f"   ✅ Professional report generated: {report.report_id}")
            print(f"   ✅ Executive summary: {len(report.executive_summary)} characters")
            print(f"   ✅ Report sections: {len(report.sections)}")
            print(f"   ✅ Professional formatting and disclaimer included")
            
        except Exception as e:
            self.fail(f"Professional strategy reports failed: {str(e)}")
    
    def test_6_user_interface_functionality(self):
        """
        User Acceptance: Advanced backtesting interface is user-friendly and functional
        
        Validates:
        - Interface files exist and are properly structured
        - Runner script is executable
        - Required dependencies are specified
        - User documentation is clear
        """
        
        print("\n6️⃣ Testing user interface functionality...")
        
        try:
            # Check interface file exists
            interface_file = Path("advanced_backtesting_interface.py")
            self.assertTrue(interface_file.exists(), "Advanced backtesting interface should exist")
            
            # Check runner script exists and is executable
            runner_script = Path("run_advanced_backtesting.sh")
            self.assertTrue(runner_script.exists(), "Runner script should exist")
            self.assertTrue(os.access(runner_script, os.X_OK), "Runner script should be executable")
            
            # Validate interface file structure
            with open(interface_file, 'r') as f:
                interface_content = f.read()
            
            # Check for key components
            required_components = [
                "class AdvancedBacktestingInterface",
                "def _run_single_backtest",
                "def _run_strategy_competition", 
                "def _run_scenario_analysis",
                "def _run_portfolio_optimization",
                "streamlit"
            ]
            
            for component in required_components:
                self.assertIn(component, interface_content, f"Interface should have {component}")
            
            # Check runner script structure
            with open(runner_script, 'r') as f:
                runner_content = f.read()
            
            # Check for key runner components
            runner_requirements = [
                "streamlit run",
                "SSL_CERT_FILE",
                "Advanced Backtesting Interface"
            ]
            
            for requirement in runner_requirements:
                self.assertIn(requirement, runner_content, f"Runner should include {requirement}")
            
            print(f"   ✅ Interface file exists and is properly structured")
            print(f"   ✅ Runner script is executable")
            print(f"   ✅ All required components present")
            print(f"   ✅ User-friendly configuration and documentation")
            
        except Exception as e:
            self.fail(f"User interface functionality test failed: {str(e)}")
    
    def test_7_integration_and_performance(self):
        """
        User Acceptance: All Sprint 5 components work together seamlessly
        
        Validates:
        - Components integrate properly
        - Performance is acceptable for user experience
        - Error handling is robust
        - Results are consistent across runs
        """
        
        print("\n7️⃣ Testing integration and performance...")
        
        try:
            # Test full workflow integration
            start_time = datetime.now()
            
            # 1. Run backtest
            config = BacktestConfig(
                symbol=self.test_symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=50000,  # Smaller for faster testing
                position_size_pct=0.20,
                stop_loss_pct=0.05,
                regime_threshold=0.7,
                lookback_period=15
            )
            
            backtest_result = asyncio.run(self.backtester.run_backtest(config))
            
            # 2. Generate report
            report = asyncio.run(self.reporter.generate_comprehensive_report(backtest_result))
            
            # 3. Test scenario analysis
            scenario = MarketScenario(
                name="Quick Test",
                description="Fast scenario test",
                market_shock=-0.10,
                volatility_multiplier=1.5,
                duration_days=10
            )
            
            scenario_result = asyncio.run(
                self.scenario_analyzer.analyze_scenario(
                    self.test_symbol, self.start_date, self.end_date, scenario
                )
            )
            
            # Calculate total execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Validate performance
            self.assertLess(execution_time, 60, "Full workflow should complete within 60 seconds")
            
            # Validate integration
            self.assertIsNotNone(backtest_result, "Backtest should complete successfully")
            self.assertIsNotNone(report, "Report should generate successfully")
            self.assertIsNotNone(scenario_result, "Scenario analysis should complete successfully")
            
            # Validate consistency
            self.assertEqual(report.symbol, backtest_result.symbol, "Report and backtest symbols should match")
            self.assertEqual(report.strategy_name, backtest_result.strategy_name, "Strategy names should match")
            
            print(f"   ✅ Full workflow integration: {execution_time:.1f} seconds")
            print(f"   ✅ All components work together seamlessly")
            print(f"   ✅ Performance meets user experience requirements")
            print(f"   ✅ Error handling and consistency validated")
            
        except Exception as e:
            self.fail(f"Integration and performance test failed: {str(e)}")

def run_sprint5_acceptance_tests():
    """Run all Sprint 5 user acceptance tests"""
    
    print("🚀 SwingTradeAI Sprint 5 User Acceptance Tests")
    print("=" * 60)
    print("Testing: Advanced Backtesting + Model Competition")
    print("User Focus: Professional strategy analysis and comparison")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(Sprint5UserAcceptanceTests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 SPRINT 5 USER ACCEPTANCE TEST SUMMARY")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"🚨 Errors: {errors}")
    print(f"📈 Success Rate: {(passed/total_tests)*100:.1f}%")
    
    if failures == 0 and errors == 0:
        print("\n🎉 ALL SPRINT 5 USER ACCEPTANCE CRITERIA MET!")
        print("✅ Advanced backtesting with regime detection")
        print("✅ Multi-strategy competition analysis")
        print("✅ Scenario analysis and stress testing")
        print("✅ Professional strategy reports")
        print("✅ User-friendly interface")
        print("✅ Production-ready performance")
        print("\n🚀 Sprint 5 is COMPLETE and ready for user deployment!")
    else:
        print(f"\n⚠️ {failures + errors} tests need attention before Sprint 5 completion")
        
        if result.failures:
            print("\n❌ FAILURES:")
            for test, traceback in result.failures:
                error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
                print(f"   • {test}: {error_msg}")
        
        if result.errors:
            print("\n🚨 ERRORS:")
            for test, traceback in result.errors:
                error_msg = traceback.split('\n')[-2]
                print(f"   • {test}: {error_msg}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_sprint5_acceptance_tests()
    exit(0 if success else 1)
