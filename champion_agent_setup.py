"""
Champion Agent Setup System - NEAT Options System
Set up and manage champion agents for production trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import asyncio

from src.database.agent_database import AgentDatabase
from per_stock_neat_framework import PerStockNEATFramework
from agent_performance_validator import AgentPerformanceValidator
from enhanced_data_pipeline import EnhancedDataPipeline

class ChampionAgentSetup:
    """Setup and manage champion agents for production trading."""
    
    def __init__(self, db_path: str = "data/neat_agents.db"):
        """Initialize champion agent setup system."""
        self.db_path = db_path
        self.agent_db = AgentDatabase(db_path)
        self.validator = AgentPerformanceValidator(db_path)
        self.data_pipeline = EnhancedDataPipeline()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Champion criteria
        self.champion_criteria = {
            'min_accuracy': 0.78,
            'min_return': 0.15,
            'min_sharpe': 1.0,
            'max_drawdown': 0.15,
            'min_win_rate': 0.70,
            'min_trades': 50,
            'max_overfitting': 0.10
        }
        
        # Top stocks for champion setup
        self.champion_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV',
            'PFE', 'AVGO'
        ]
    
    def setup_all_champions(self) -> Dict:
        """Set up champion agents for all stocks."""
        
        self.logger.info("👑 Setting up champion agents for production trading...")
        
        # Get trained agents
        trained_agents = self._get_trained_agents()
        
        if not trained_agents:
            raise RuntimeError("No trained agents found. Run initial training first.")
        
        # Validate agents
        validation_results = self.validator.validate_all_agents()
        
        # Select and set up champions
        champions = self._select_champions(trained_agents, validation_results)
        
        # Configure champions for production
        production_config = self._configure_champions_for_production(champions)
        
        # Generate champion report
        report = self._generate_champion_report(champions, production_config)
        
        self.logger.info("✅ Champion agent setup completed!")
        return report
    
    def _get_trained_agents(self) -> Dict[str, Dict]:
        """Get all trained agents."""
        
        trained_agents = {}
        
        for symbol in self.champion_stocks:
            try:
                agent_data = self.agent_db.load_agent(symbol)
                if agent_data:
                    trained_agents[symbol] = agent_data
                    self.logger.info(f"Found trained agent for {symbol}")
                else:
                    self.logger.warning(f"No trained agent found for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Error loading agent for {symbol}: {str(e)}")
        
        return trained_agents
    
    def _select_champions(self, trained_agents: Dict, validation_results: Dict) -> Dict[str, Dict]:
        """Select champion agents based on performance criteria."""
        
        champions = {}
        
        for symbol, agent_data in trained_agents.items():
            try:
                # Check if agent passed validation
                validation_result = validation_results.get('validation_results', {}).get(symbol, {})
                
                if not validation_result.get('passed', False):
                    self.logger.warning(f"Agent {symbol} failed validation - not eligible for champion status")
                    continue
                
                # Check champion criteria
                if self._meets_champion_criteria(agent_data):
                    # Set as champion in database
                    self.agent_db.set_champion_agent(symbol, agent_data['agent_id'])
                    
                    champions[symbol] = agent_data
                    self.logger.info(f"👑 {symbol} agent selected as champion")
                else:
                    self.logger.warning(f"Agent {symbol} does not meet champion criteria")
                    
            except Exception as e:
                self.logger.error(f"Error selecting champion for {symbol}: {str(e)}")
        
        return champions
    
    def _meets_champion_criteria(self, agent_data: Dict) -> bool:
        """Check if agent meets champion criteria."""
        
        criteria_checks = {
            'accuracy': agent_data.get('accuracy', 0) >= self.champion_criteria['min_accuracy'],
            'return': agent_data.get('total_return', 0) >= self.champion_criteria['min_return'],
            'sharpe': agent_data.get('sharpe_ratio', 0) >= self.champion_criteria['min_sharpe'],
            'drawdown': agent_data.get('max_drawdown', 1.0) <= self.champion_criteria['max_drawdown'],
            'win_rate': agent_data.get('win_rate', 0) >= self.champion_criteria['min_win_rate'],
            'trades': agent_data.get('total_trades', 0) >= self.champion_criteria['min_trades'],
            'overfitting': agent_data.get('overfitting_score', 1.0) <= self.champion_criteria['max_overfitting']
        }
        
        passed_criteria = sum(criteria_checks.values())
        total_criteria = len(criteria_checks)
        
        # Must pass at least 6 out of 7 criteria (85%)
        meets_criteria = passed_criteria >= 6
        
        if not meets_criteria:
            self.logger.info(f"Champion criteria check: {passed_criteria}/{total_criteria}")
            for criterion, passed in criteria_checks.items():
                if not passed:
                    self.logger.info(f"  ❌ {criterion}: {agent_data.get(criterion.replace('_', ''), 'N/A')}")
        
        return meets_criteria
    
    def _configure_champions_for_production(self, champions: Dict) -> Dict:
        """Configure champion agents for production trading."""
        
        production_config = {}
        
        for symbol, agent_data in champions.items():
            try:
                # Create production configuration
                config = {
                    'symbol': symbol,
                    'agent_id': agent_data['agent_id'],
                    'trading_enabled': True,
                    'risk_parameters': self._get_risk_parameters(symbol, agent_data),
                    'position_sizing': self._get_position_sizing(symbol, agent_data),
                    'options_strategy': self._get_options_strategy(symbol, agent_data),
                    'monitoring_thresholds': self._get_monitoring_thresholds(symbol, agent_data),
                    'retraining_triggers': self._get_retraining_triggers(symbol, agent_data)
                }
                
                production_config[symbol] = config
                
                # Save production config to database
                self._save_production_config(symbol, config)
                
                self.logger.info(f"✅ Production config created for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error configuring {symbol} for production: {str(e)}")
        
        return production_config
    
    def _get_risk_parameters(self, symbol: str, agent_data: Dict) -> Dict:
        """Get risk parameters for production trading."""
        
        # Base risk parameters
        base_risk = {
            'max_position_size': 0.05,  # 5% of portfolio max
            'stop_loss': 0.10,          # 10% stop loss
            'max_daily_loss': 0.02,     # 2% max daily loss
            'max_drawdown': 0.15,       # 15% max drawdown
            'volatility_adjustment': True
        }
        
        # Adjust based on agent performance
        sharpe_ratio = agent_data.get('sharpe_ratio', 1.0)
        max_drawdown = agent_data.get('max_drawdown', 0.15)
        
        # Higher Sharpe ratio allows slightly larger positions
        if sharpe_ratio > 1.5:
            base_risk['max_position_size'] *= 1.2
        elif sharpe_ratio < 0.8:
            base_risk['max_position_size'] *= 0.8
        
        # Lower historical drawdown allows slightly more risk
        if max_drawdown < 0.08:
            base_risk['max_daily_loss'] *= 1.1
        elif max_drawdown > 0.12:
            base_risk['max_daily_loss'] *= 0.9
        
        return base_risk
    
    def _get_position_sizing(self, symbol: str, agent_data: Dict) -> Dict:
        """Get position sizing parameters."""
        
        return {
            'base_size': 0.02,          # 2% base position size
            'confidence_scaling': True,  # Scale based on prediction confidence
            'volatility_scaling': True,  # Scale based on volatility
            'kelly_criterion': False,    # Conservative approach
            'max_size': 0.05,           # 5% maximum
            'min_size': 0.01            # 1% minimum
        }
    
    def _get_options_strategy(self, symbol: str, agent_data: Dict) -> Dict:
        """Get options strategy configuration."""
        
        return {
            'primary_strategy': 'cash_secured_puts',
            'strike_selection': {
                'base_delta': 0.30,      # 30 delta puts
                'confidence_adjustment': True,
                'min_delta': 0.20,
                'max_delta': 0.40
            },
            'expiration': {
                'target_dte': 30,        # 30 days to expiration
                'min_dte': 21,
                'max_dte': 45
            },
            'premium_targets': {
                'min_weekly_yield': 0.015,  # 1.5% weekly
                'target_annualized': 0.78   # 78% annualized
            },
            'assignment_management': {
                'max_assignment_rate': 0.05,  # 5% max
                'early_close_threshold': 0.50, # Close at 50% profit
                'roll_threshold': 0.21         # Roll at 21 DTE
            }
        }
    
    def _get_monitoring_thresholds(self, symbol: str, agent_data: Dict) -> Dict:
        """Get monitoring thresholds for production."""
        
        return {
            'accuracy_threshold': 0.75,     # Trigger alert if accuracy drops below 75%
            'return_threshold': -0.05,      # Alert if 5% loss
            'drawdown_threshold': 0.10,     # Alert if 10% drawdown
            'volatility_threshold': 2.0,    # Alert if volatility 2x normal
            'correlation_threshold': 0.8,   # Alert if correlation breaks down
            'prediction_confidence_threshold': 0.6  # Alert if confidence drops
        }
    
    def _get_retraining_triggers(self, symbol: str, agent_data: Dict) -> Dict:
        """Get retraining trigger conditions."""
        
        return {
            'accuracy_degradation': 0.05,   # Retrain if accuracy drops 5%
            'consecutive_losses': 10,       # Retrain after 10 consecutive losses
            'market_regime_change': True,   # Retrain on regime change
            'scheduled_retraining': 30,     # Retrain every 30 days
            'performance_threshold': 0.70   # Retrain if performance drops below 70%
        }
    
    def _save_production_config(self, symbol: str, config: Dict):
        """Save production configuration to database."""
        
        try:
            # Save to stock metadata table
            self.agent_db.update_stock_metadata(symbol, {
                'production_config': json.dumps(config),
                'champion_status': True,
                'last_config_update': datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Failed to save production config for {symbol}: {str(e)}")
    
    def _generate_champion_report(self, champions: Dict, production_config: Dict) -> Dict:
        """Generate comprehensive champion report."""
        
        report = {
            'setup_date': datetime.now().isoformat(),
            'total_stocks': len(self.champion_stocks),
            'champions_selected': len(champions),
            'champion_rate': len(champions) / len(self.champion_stocks),
            'champions': champions,
            'production_config': production_config,
            'performance_summary': {},
            'risk_summary': {},
            'readiness_assessment': {}
        }
        
        # Performance summary
        if champions:
            accuracies = [agent['accuracy'] for agent in champions.values()]
            returns = [agent['total_return'] for agent in champions.values()]
            sharpes = [agent['sharpe_ratio'] for agent in champions.values()]
            
            report['performance_summary'] = {
                'average_accuracy': np.mean(accuracies),
                'average_return': np.mean(returns),
                'average_sharpe': np.mean(sharpes),
                'min_accuracy': np.min(accuracies),
                'max_return': np.max(returns),
                'total_expected_return': np.sum(returns) / len(returns)  # Portfolio return estimate
            }
        
        # Risk summary
        if production_config:
            max_position_sizes = [config['risk_parameters']['max_position_size'] 
                                for config in production_config.values()]
            
            report['risk_summary'] = {
                'total_max_exposure': sum(max_position_sizes),
                'average_position_size': np.mean(max_position_sizes),
                'risk_diversification': len(production_config),
                'max_portfolio_risk': min(sum(max_position_sizes), 1.0)
            }
        
        # Readiness assessment
        report['readiness_assessment'] = self._assess_production_readiness(champions, production_config)
        
        # Save report
        self._save_champion_report(report)
        
        # Print summary
        self._print_champion_summary(report)
        
        return report
    
    def _assess_production_readiness(self, champions: Dict, production_config: Dict) -> Dict:
        """Assess readiness for production trading."""
        
        readiness_checks = {
            'sufficient_champions': len(champions) >= 15,  # At least 15 champions
            'performance_adequate': all(
                agent['accuracy'] >= 0.75 and agent['total_return'] >= 0.12
                for agent in champions.values()
            ),
            'risk_controlled': all(
                config['risk_parameters']['max_position_size'] <= 0.06
                for config in production_config.values()
            ),
            'diversification_adequate': len(champions) >= 10,
            'monitoring_configured': all(
                'monitoring_thresholds' in config
                for config in production_config.values()
            )
        }
        
        readiness_score = sum(readiness_checks.values()) / len(readiness_checks)
        
        return {
            'overall_ready': readiness_score >= 0.8,
            'readiness_score': readiness_score,
            'checks': readiness_checks,
            'recommendations': self._get_readiness_recommendations(readiness_checks)
        }
    
    def _get_readiness_recommendations(self, checks: Dict) -> List[str]:
        """Get recommendations for production readiness."""
        
        recommendations = []
        
        if not checks['sufficient_champions']:
            recommendations.append("Train more agents to reach minimum 15 champions")
        
        if not checks['performance_adequate']:
            recommendations.append("Improve agent performance - some agents below minimum thresholds")
        
        if not checks['risk_controlled']:
            recommendations.append("Review and adjust risk parameters - some positions too large")
        
        if not checks['diversification_adequate']:
            recommendations.append("Increase diversification - need more champion agents")
        
        if not checks['monitoring_configured']:
            recommendations.append("Complete monitoring configuration for all agents")
        
        if not recommendations:
            recommendations.append("System ready for production deployment!")
        
        return recommendations
    
    def _save_champion_report(self, report: Dict):
        """Save champion report to file."""
        
        try:
            import os
            report_dir = "data/champion_reports"
            os.makedirs(report_dir, exist_ok=True)
            
            report_path = f"{report_dir}/champions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"📄 Champion report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save champion report: {str(e)}")
    
    def _print_champion_summary(self, report: Dict):
        """Print champion summary to console."""
        
        print("\n" + "="*60)
        print("👑 CHAMPION AGENT SETUP SUMMARY")
        print("="*60)
        print(f"Setup Date: {report['setup_date']}")
        print(f"Total Stocks: {report['total_stocks']}")
        print(f"Champions Selected: {report['champions_selected']}")
        print(f"Champion Rate: {report['champion_rate']:.1%}")
        
        if 'performance_summary' in report and report['performance_summary']:
            perf = report['performance_summary']
            print(f"\n📊 Performance Summary:")
            print(f"  Average Accuracy: {perf['average_accuracy']:.1%}")
            print(f"  Average Return: {perf['average_return']:.1%}")
            print(f"  Average Sharpe: {perf['average_sharpe']:.2f}")
            print(f"  Expected Portfolio Return: {perf['total_expected_return']:.1%}")
        
        if 'risk_summary' in report and report['risk_summary']:
            risk = report['risk_summary']
            print(f"\n⚠️ Risk Summary:")
            print(f"  Total Max Exposure: {risk['total_max_exposure']:.1%}")
            print(f"  Average Position Size: {risk['average_position_size']:.1%}")
            print(f"  Risk Diversification: {risk['risk_diversification']} stocks")
        
        print(f"\n👑 Champion Agents:")
        for symbol, agent_data in report['champions'].items():
            print(f"  {symbol}: Acc {agent_data['accuracy']:.1%}, "
                  f"Ret {agent_data['total_return']:.1%}, "
                  f"Sharpe {agent_data['sharpe_ratio']:.2f}")
        
        # Readiness assessment
        readiness = report['readiness_assessment']
        status = "✅ READY" if readiness['overall_ready'] else "⚠️ NOT READY"
        print(f"\n🚀 Production Readiness: {status} ({readiness['readiness_score']:.1%})")
        
        if readiness['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in readiness['recommendations']:
                print(f"  • {rec}")
        
        print("="*60)
    
    def get_champion_status(self) -> Dict:
        """Get current champion status."""
        
        status = {
            'champions': {},
            'total_champions': 0,
            'production_ready': False
        }
        
        for symbol in self.champion_stocks:
            try:
                agent_data = self.agent_db.load_agent(symbol)
                if agent_data and agent_data.get('is_champion', False):
                    status['champions'][symbol] = {
                        'agent_id': agent_data['agent_id'],
                        'accuracy': agent_data.get('accuracy', 0),
                        'total_return': agent_data.get('total_return', 0),
                        'sharpe_ratio': agent_data.get('sharpe_ratio', 0)
                    }
                    status['total_champions'] += 1
            except Exception as e:
                self.logger.warning(f"Could not get status for {symbol}: {str(e)}")
        
        status['production_ready'] = status['total_champions'] >= 15
        
        return status


def run_champion_setup():
    """Run champion agent setup."""
    
    print("👑 NEAT Options System - Champion Agent Setup")
    print("="*60)
    
    setup = ChampionAgentSetup()
    
    try:
        report = setup.setup_all_champions()
        
        print(f"\n✅ Champion setup completed!")
        print(f"Champions selected: {report['champions_selected']}")
        print(f"Production ready: {report['readiness_assessment']['overall_ready']}")
        
        return report
        
    except Exception as e:
        print(f"\n❌ Champion setup failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_champion_setup()
