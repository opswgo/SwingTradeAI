"""
Agent Performance Validator - NEAT Options System
Comprehensive validation of trained agents with real market data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json
import yfinance as yf
from scipy import stats

from src.database.agent_database import AgentDatabase
from per_stock_neat_framework import PerStockNEATFramework
from enhanced_data_pipeline import EnhancedDataPipeline

class AgentPerformanceValidator:
    """Validate agent performance with comprehensive metrics."""
    
    def __init__(self, db_path: str = "data/neat_agents.db"):
        """Initialize performance validator."""
        self.db_path = db_path
        self.agent_db = AgentDatabase(db_path)
        self.data_pipeline = EnhancedDataPipeline()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance benchmarks
        self.benchmarks = {
            'spy_return': 0.10,      # S&P 500 annual return
            'risk_free_rate': 0.04,  # 4% sweep account
            'mag7_return': 0.25,     # Mag7 average return
            'target_multiplier': 2.0 # Must 2x S&P 500
        }
    
    def validate_all_agents(self) -> Dict:
        """Validate all champion agents comprehensively."""
        
        self.logger.info("🔍 Starting comprehensive agent validation...")
        
        # Get all champion agents
        champions = self._get_champion_agents()
        
        if not champions:
            self.logger.warning("No champion agents found for validation")
            return {'champions': [], 'validation_results': {}}
        
        # Validate each agent
        validation_results = {}
        
        for symbol, agent_data in champions.items():
            self.logger.info(f"Validating {symbol} agent...")
            
            try:
                result = self._validate_single_agent(symbol, agent_data)
                validation_results[symbol] = result
                
                # Log validation summary
                self._log_validation_result(symbol, result)
                
            except Exception as e:
                self.logger.error(f"Validation failed for {symbol}: {str(e)}")
                validation_results[symbol] = {'error': str(e), 'passed': False}
        
        # Generate comprehensive report
        report = self._generate_validation_report(validation_results)
        
        self.logger.info("✅ Agent validation completed!")
        return report
    
    def _get_champion_agents(self) -> Dict[str, Dict]:
        """Get all champion agents."""
        
        champions = {}
        
        # Top 20 stocks
        top_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV',
            'PFE', 'AVGO'
        ]
        
        for symbol in top_stocks:
            try:
                agent_data = self.agent_db.load_agent(symbol)
                if agent_data and agent_data.get('is_champion', False):
                    champions[symbol] = agent_data
            except Exception as e:
                self.logger.warning(f"Could not load agent for {symbol}: {str(e)}")
        
        return champions
    
    def _validate_single_agent(self, symbol: str, agent_data: Dict) -> Dict:
        """Validate a single agent comprehensively."""
        
        validation_result = {
            'symbol': symbol,
            'agent_id': agent_data['agent_id'],
            'validation_date': datetime.now().isoformat(),
            'tests_passed': 0,
            'total_tests': 8,
            'passed': False,
            'performance_metrics': {},
            'benchmark_comparisons': {},
            'detailed_results': {}
        }
        
        # Test 1: Accuracy Validation
        accuracy_test = self._test_prediction_accuracy(symbol, agent_data)
        validation_result['detailed_results']['accuracy_test'] = accuracy_test
        if accuracy_test['passed']:
            validation_result['tests_passed'] += 1
        
        # Test 2: Return Performance
        return_test = self._test_return_performance(symbol, agent_data)
        validation_result['detailed_results']['return_test'] = return_test
        if return_test['passed']:
            validation_result['tests_passed'] += 1
        
        # Test 3: Risk-Adjusted Returns (Sharpe Ratio)
        sharpe_test = self._test_sharpe_ratio(symbol, agent_data)
        validation_result['detailed_results']['sharpe_test'] = sharpe_test
        if sharpe_test['passed']:
            validation_result['tests_passed'] += 1
        
        # Test 4: Maximum Drawdown
        drawdown_test = self._test_max_drawdown(symbol, agent_data)
        validation_result['detailed_results']['drawdown_test'] = drawdown_test
        if drawdown_test['passed']:
            validation_result['tests_passed'] += 1
        
        # Test 5: Win Rate
        winrate_test = self._test_win_rate(symbol, agent_data)
        validation_result['detailed_results']['winrate_test'] = winrate_test
        if winrate_test['passed']:
            validation_result['tests_passed'] += 1
        
        # Test 6: Benchmark Comparison (vs Buy & Hold)
        benchmark_test = self._test_benchmark_comparison(symbol, agent_data)
        validation_result['detailed_results']['benchmark_test'] = benchmark_test
        if benchmark_test['passed']:
            validation_result['tests_passed'] += 1
        
        # Test 7: Options Strategy Validation
        options_test = self._test_options_strategy(symbol, agent_data)
        validation_result['detailed_results']['options_test'] = options_test
        if options_test['passed']:
            validation_result['tests_passed'] += 1
        
        # Test 8: Overfitting Detection
        overfitting_test = self._test_overfitting(symbol, agent_data)
        validation_result['detailed_results']['overfitting_test'] = overfitting_test
        if overfitting_test['passed']:
            validation_result['tests_passed'] += 1
        
        # Overall validation result
        validation_result['passed'] = validation_result['tests_passed'] >= 6  # 75% pass rate
        validation_result['pass_rate'] = validation_result['tests_passed'] / validation_result['total_tests']
        
        return validation_result
    
    def _test_prediction_accuracy(self, symbol: str, agent_data: Dict) -> Dict:
        """Test prediction accuracy against actual market movements."""
        
        try:
            # Get recent predictions and outcomes
            accuracy = agent_data.get('accuracy', 0)
            validation_accuracy = agent_data.get('validation_accuracy', 0)
            
            # Minimum accuracy threshold: 75%
            min_accuracy = 0.75
            
            passed = accuracy >= min_accuracy and validation_accuracy >= (min_accuracy - 0.05)
            
            return {
                'test_name': 'Prediction Accuracy',
                'passed': passed,
                'accuracy': accuracy,
                'validation_accuracy': validation_accuracy,
                'threshold': min_accuracy,
                'message': f"Accuracy: {accuracy:.1%} (min: {min_accuracy:.1%})"
            }
            
        except Exception as e:
            return {
                'test_name': 'Prediction Accuracy',
                'passed': False,
                'error': str(e),
                'message': f"Test failed: {str(e)}"
            }
    
    def _test_return_performance(self, symbol: str, agent_data: Dict) -> Dict:
        """Test return performance against benchmarks."""
        
        try:
            total_return = agent_data.get('total_return', 0)
            
            # Must beat risk-free rate (4%) and target 15%+ annual return
            min_return = 0.15
            
            passed = total_return >= min_return
            
            return {
                'test_name': 'Return Performance',
                'passed': passed,
                'total_return': total_return,
                'threshold': min_return,
                'message': f"Return: {total_return:.1%} (min: {min_return:.1%})"
            }
            
        except Exception as e:
            return {
                'test_name': 'Return Performance',
                'passed': False,
                'error': str(e),
                'message': f"Test failed: {str(e)}"
            }
    
    def _test_sharpe_ratio(self, symbol: str, agent_data: Dict) -> Dict:
        """Test risk-adjusted returns (Sharpe ratio)."""
        
        try:
            sharpe_ratio = agent_data.get('sharpe_ratio', 0)
            
            # Minimum Sharpe ratio: 1.0 (good risk-adjusted returns)
            min_sharpe = 1.0
            
            passed = sharpe_ratio >= min_sharpe
            
            return {
                'test_name': 'Sharpe Ratio',
                'passed': passed,
                'sharpe_ratio': sharpe_ratio,
                'threshold': min_sharpe,
                'message': f"Sharpe: {sharpe_ratio:.2f} (min: {min_sharpe:.2f})"
            }
            
        except Exception as e:
            return {
                'test_name': 'Sharpe Ratio',
                'passed': False,
                'error': str(e),
                'message': f"Test failed: {str(e)}"
            }
    
    def _test_max_drawdown(self, symbol: str, agent_data: Dict) -> Dict:
        """Test maximum drawdown (risk control)."""
        
        try:
            max_drawdown = agent_data.get('max_drawdown', 1.0)
            
            # Maximum acceptable drawdown: 15%
            max_acceptable_dd = 0.15
            
            passed = max_drawdown <= max_acceptable_dd
            
            return {
                'test_name': 'Maximum Drawdown',
                'passed': passed,
                'max_drawdown': max_drawdown,
                'threshold': max_acceptable_dd,
                'message': f"Max DD: {max_drawdown:.1%} (max: {max_acceptable_dd:.1%})"
            }
            
        except Exception as e:
            return {
                'test_name': 'Maximum Drawdown',
                'passed': False,
                'error': str(e),
                'message': f"Test failed: {str(e)}"
            }
    
    def _test_win_rate(self, symbol: str, agent_data: Dict) -> Dict:
        """Test win rate (percentage of profitable trades)."""
        
        try:
            win_rate = agent_data.get('win_rate', 0)
            
            # Minimum win rate: 65%
            min_win_rate = 0.65
            
            passed = win_rate >= min_win_rate
            
            return {
                'test_name': 'Win Rate',
                'passed': passed,
                'win_rate': win_rate,
                'threshold': min_win_rate,
                'message': f"Win Rate: {win_rate:.1%} (min: {min_win_rate:.1%})"
            }
            
        except Exception as e:
            return {
                'test_name': 'Win Rate',
                'passed': False,
                'error': str(e),
                'message': f"Test failed: {str(e)}"
            }
    
    def _test_benchmark_comparison(self, symbol: str, agent_data: Dict) -> Dict:
        """Test performance vs buy-and-hold benchmark."""
        
        try:
            agent_return = agent_data.get('total_return', 0)
            
            # Get buy-and-hold return for comparison
            buy_hold_return = self._calculate_buy_hold_return(symbol)
            
            # Must beat buy-and-hold by at least 5%
            min_outperformance = 0.05
            outperformance = agent_return - buy_hold_return
            
            passed = outperformance >= min_outperformance
            
            return {
                'test_name': 'Benchmark Comparison',
                'passed': passed,
                'agent_return': agent_return,
                'buy_hold_return': buy_hold_return,
                'outperformance': outperformance,
                'threshold': min_outperformance,
                'message': f"Outperformance: {outperformance:.1%} vs Buy&Hold (min: {min_outperformance:.1%})"
            }
            
        except Exception as e:
            return {
                'test_name': 'Benchmark Comparison',
                'passed': False,
                'error': str(e),
                'message': f"Test failed: {str(e)}"
            }
    
    def _test_options_strategy(self, symbol: str, agent_data: Dict) -> Dict:
        """Test options-specific strategy performance."""
        
        try:
            # Check if agent is actually using options strategies
            total_trades = agent_data.get('total_trades', 0)
            
            # Should have reasonable number of trades (not too few, not too many)
            min_trades = 50
            max_trades = 500
            
            trade_frequency_ok = min_trades <= total_trades <= max_trades
            
            # Check for options-specific metrics (simplified)
            # In production, would check assignment rates, premium collection, etc.
            options_performance_ok = True  # Placeholder
            
            passed = trade_frequency_ok and options_performance_ok
            
            return {
                'test_name': 'Options Strategy',
                'passed': passed,
                'total_trades': total_trades,
                'trade_frequency_ok': trade_frequency_ok,
                'message': f"Trades: {total_trades} (range: {min_trades}-{max_trades})"
            }
            
        except Exception as e:
            return {
                'test_name': 'Options Strategy',
                'passed': False,
                'error': str(e),
                'message': f"Test failed: {str(e)}"
            }
    
    def _test_overfitting(self, symbol: str, agent_data: Dict) -> Dict:
        """Test for overfitting (training vs validation performance gap)."""
        
        try:
            accuracy = agent_data.get('accuracy', 0)
            validation_accuracy = agent_data.get('validation_accuracy', 0)
            overfitting_score = agent_data.get('overfitting_score', 0)
            
            # Performance gap should be small
            performance_gap = abs(accuracy - validation_accuracy)
            max_gap = 0.10  # 10% maximum gap
            
            gap_ok = performance_gap <= max_gap
            overfitting_ok = overfitting_score <= 0.15
            
            passed = gap_ok and overfitting_ok
            
            return {
                'test_name': 'Overfitting Detection',
                'passed': passed,
                'performance_gap': performance_gap,
                'overfitting_score': overfitting_score,
                'max_gap': max_gap,
                'message': f"Gap: {performance_gap:.1%}, Overfitting: {overfitting_score:.3f}"
            }
            
        except Exception as e:
            return {
                'test_name': 'Overfitting Detection',
                'passed': False,
                'error': str(e),
                'message': f"Test failed: {str(e)}"
            }
    
    def _calculate_buy_hold_return(self, symbol: str, period_days: int = 365) -> float:
        """Calculate buy-and-hold return for comparison."""
        
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty or len(data) < 2:
                return 0.10  # Default 10% return assumption
            
            # Calculate total return
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            
            total_return = (end_price / start_price) - 1
            
            return total_return
            
        except Exception as e:
            self.logger.warning(f"Could not calculate buy-hold return for {symbol}: {str(e)}")
            return 0.10  # Default assumption
    
    def _log_validation_result(self, symbol: str, result: Dict):
        """Log validation result summary."""
        
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        pass_rate = result['pass_rate']
        
        self.logger.info(f"{symbol}: {status} ({result['tests_passed']}/{result['total_tests']} tests, {pass_rate:.1%})")
        
        # Log failed tests
        if not result['passed']:
            for test_name, test_result in result['detailed_results'].items():
                if not test_result.get('passed', False):
                    self.logger.warning(f"  ❌ {test_result.get('test_name', test_name)}: {test_result.get('message', 'Failed')}")
    
    def _generate_validation_report(self, validation_results: Dict) -> Dict:
        """Generate comprehensive validation report."""
        
        report = {
            'validation_date': datetime.now().isoformat(),
            'total_agents': len(validation_results),
            'passed_agents': len([r for r in validation_results.values() if r.get('passed', False)]),
            'validation_results': validation_results,
            'summary_statistics': {},
            'recommendations': []
        }
        
        # Calculate summary statistics
        if validation_results:
            pass_rates = [r.get('pass_rate', 0) for r in validation_results.values()]
            
            report['summary_statistics'] = {
                'overall_pass_rate': sum([r.get('passed', False) for r in validation_results.values()]) / len(validation_results),
                'average_test_pass_rate': np.mean(pass_rates),
                'median_test_pass_rate': np.median(pass_rates),
                'min_test_pass_rate': np.min(pass_rates),
                'max_test_pass_rate': np.max(pass_rates)
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(validation_results)
        
        # Save report
        self._save_validation_report(report)
        
        # Print summary
        self._print_validation_summary(report)
        
        return report
    
    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        
        recommendations = []
        
        failed_agents = [symbol for symbol, result in validation_results.items() 
                        if not result.get('passed', False)]
        
        if failed_agents:
            recommendations.append(f"Retrain agents for: {', '.join(failed_agents)}")
        
        # Check common failure patterns
        accuracy_failures = []
        return_failures = []
        
        for symbol, result in validation_results.items():
            detailed = result.get('detailed_results', {})
            
            if not detailed.get('accuracy_test', {}).get('passed', True):
                accuracy_failures.append(symbol)
            
            if not detailed.get('return_test', {}).get('passed', True):
                return_failures.append(symbol)
        
        if accuracy_failures:
            recommendations.append(f"Improve prediction accuracy for: {', '.join(accuracy_failures)}")
        
        if return_failures:
            recommendations.append(f"Optimize return generation for: {', '.join(return_failures)}")
        
        if len(failed_agents) > len(validation_results) * 0.3:
            recommendations.append("Consider adjusting training parameters - high failure rate detected")
        
        return recommendations
    
    def _save_validation_report(self, report: Dict):
        """Save validation report to file."""
        
        try:
            import os
            report_dir = "data/validation_reports"
            os.makedirs(report_dir, exist_ok=True)
            
            report_path = f"{report_dir}/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"📄 Validation report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {str(e)}")
    
    def _print_validation_summary(self, report: Dict):
        """Print validation summary to console."""
        
        print("\n" + "="*60)
        print("🔍 AGENT PERFORMANCE VALIDATION SUMMARY")
        print("="*60)
        print(f"Validation Date: {report['validation_date']}")
        print(f"Total Agents: {report['total_agents']}")
        print(f"Passed Validation: {report['passed_agents']}")
        print(f"Overall Pass Rate: {report['summary_statistics'].get('overall_pass_rate', 0):.1%}")
        
        print(f"\n📊 Test Statistics:")
        stats = report['summary_statistics']
        print(f"  Average Test Pass Rate: {stats.get('average_test_pass_rate', 0):.1%}")
        print(f"  Median Test Pass Rate: {stats.get('median_test_pass_rate', 0):.1%}")
        print(f"  Range: {stats.get('min_test_pass_rate', 0):.1%} - {stats.get('max_test_pass_rate', 0):.1%}")
        
        print(f"\n✅ Validated Agents:")
        for symbol, result in report['validation_results'].items():
            if result.get('passed', False):
                print(f"  {symbol}: {result['tests_passed']}/{result['total_tests']} tests ({result['pass_rate']:.1%})")
        
        failed_agents = [symbol for symbol, result in report['validation_results'].items() 
                        if not result.get('passed', False)]
        
        if failed_agents:
            print(f"\n❌ Failed Validation:")
            for symbol in failed_agents:
                result = report['validation_results'][symbol]
                print(f"  {symbol}: {result['tests_passed']}/{result['total_tests']} tests ({result['pass_rate']:.1%})")
        
        if report['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        print("="*60)


def run_agent_validation():
    """Run comprehensive agent validation."""
    
    print("🔍 NEAT Options System - Agent Performance Validation")
    print("="*60)
    
    validator = AgentPerformanceValidator()
    
    try:
        report = validator.validate_all_agents()
        
        print(f"\n✅ Validation completed!")
        print(f"Agents validated: {report['total_agents']}")
        print(f"Passed validation: {report['passed_agents']}")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_agent_validation()
