"""
Initial Agent Training Pipeline - NEAT Options System
Train agents for top 20 S&P 500 stocks with 100+ generations each.
"""

import asyncio
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import time

from per_stock_neat_framework import PerStockNEATFramework
from src.database.agent_database import AgentDatabase
from enhanced_data_pipeline import EnhancedDataPipeline
from database_migration import DatabaseMigration

class InitialAgentTrainingPipeline:
    """Pipeline for training initial NEAT agents for top S&P 500 stocks."""
    
    def __init__(self, db_path: str = "data/neat_agents.db"):
        """Initialize training pipeline."""
        self.db_path = db_path
        self.agent_db = AgentDatabase(db_path)
        self.data_pipeline = EnhancedDataPipeline()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Top 20 S&P 500 stocks by market cap (high liquidity, options volume)
        self.top_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'UNH', 'JNJ', 'V', 'PG', 'JPM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV',
            'PFE', 'AVGO'
        ]
        
        # Training configuration
        self.training_config = {
            'generations': 150,  # 100+ generations as required
            'population_size': 150,
            'accuracy_threshold': 0.80,
            'overfitting_threshold': 0.15,
            'early_stopping_patience': 25,
            'validation_split': 0.2,
            'min_data_years': 3,
            'max_parallel_jobs': min(mp.cpu_count(), 4)  # Conservative parallelism
        }
    
    def run_initial_training(self) -> Dict[str, str]:
        """Run initial training for all top stocks."""
        
        self.logger.info("🚀 Starting Initial Agent Training Pipeline")
        self.logger.info(f"Training {len(self.top_stocks)} stocks with {self.training_config['generations']} generations each")
        
        # Ensure database is ready
        migration = DatabaseMigration(self.db_path)
        if not migration.run_migrations():
            raise RuntimeError("Database migration failed")
        
        # Pre-collect data for all stocks
        self.logger.info("📊 Pre-collecting data for all stocks...")
        self._precollect_data()
        
        # Train agents in parallel
        training_results = self._train_agents_parallel()
        
        # Validate and set champions
        champions = self._validate_and_set_champions(training_results)
        
        # Generate training report
        report = self._generate_training_report(champions)
        
        self.logger.info("✅ Initial Agent Training Pipeline Complete!")
        return champions
    
    def _precollect_data(self):
        """Pre-collect data for all stocks to avoid race conditions."""
        
        for symbol in self.top_stocks:
            try:
                self.logger.info(f"Collecting data for {symbol}...")
                
                # Check if data exists
                existing_data = self.data_pipeline.load_stock_data(symbol)
                
                if len(existing_data) < (self.training_config['min_data_years'] * 252):
                    # Collect fresh data
                    self.data_pipeline.collect_extended_data(
                        symbol, 
                        years=self.training_config['min_data_years'] + 2  # Extra buffer
                    )
                    self.logger.info(f"✅ Data collected for {symbol}")
                else:
                    self.logger.info(f"✅ Existing data sufficient for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to collect data for {symbol}: {str(e)}")
    
    def _train_agents_parallel(self) -> Dict[str, Optional[str]]:
        """Train agents in parallel for all stocks."""
        
        training_results = {}
        
        # Use process pool for parallel training
        with ProcessPoolExecutor(max_workers=self.training_config['max_parallel_jobs']) as executor:
            
            # Submit training jobs
            future_to_symbol = {
                executor.submit(self._train_single_agent, symbol): symbol 
                for symbol in self.top_stocks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                
                try:
                    agent_id = future.result(timeout=3600)  # 1 hour timeout per stock
                    training_results[symbol] = agent_id
                    
                    if agent_id:
                        self.logger.info(f"✅ Training completed for {symbol}: {agent_id}")
                    else:
                        self.logger.warning(f"⚠️ Training failed for {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"❌ Training error for {symbol}: {str(e)}")
                    training_results[symbol] = None
        
        return training_results
    
    def _train_single_agent(self, symbol: str) -> Optional[str]:
        """Train a single agent for a stock."""
        
        try:
            # Create framework instance
            framework = PerStockNEATFramework(symbol, self.db_path)
            
            # Start training session
            start_time = datetime.now()
            self.logger.info(f"🎓 Starting training for {symbol} at {start_time}")
            
            # Train agent
            agent_id = framework.train_agent(
                generations=self.training_config['generations']
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            if agent_id:
                self.logger.info(f"✅ {symbol} training completed in {duration}")
                self.logger.info(f"   Agent ID: {agent_id}")
                
                # Log training completion
                self._log_training_completion(symbol, agent_id, duration)
                
                return agent_id
            else:
                self.logger.warning(f"⚠️ {symbol} training failed to produce viable agent")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Training failed for {symbol}: {str(e)}")
            return None
    
    def _log_training_completion(self, symbol: str, agent_id: str, duration: timedelta):
        """Log training completion details."""
        
        try:
            # Load agent performance
            agent_data = self.agent_db.load_agent(symbol)
            
            if agent_data:
                self.logger.info(f"📊 {symbol} Agent Performance:")
                self.logger.info(f"   Accuracy: {agent_data.get('accuracy', 0):.1%}")
                self.logger.info(f"   Total Return: {agent_data.get('total_return', 0):.1%}")
                self.logger.info(f"   Sharpe Ratio: {agent_data.get('sharpe_ratio', 0):.2f}")
                self.logger.info(f"   Training Duration: {duration}")
                
        except Exception as e:
            self.logger.error(f"Failed to log completion for {symbol}: {str(e)}")
    
    def _validate_and_set_champions(self, training_results: Dict[str, Optional[str]]) -> Dict[str, str]:
        """Validate trained agents and set champions."""
        
        champions = {}
        
        for symbol, agent_id in training_results.items():
            if agent_id:
                try:
                    # Load agent data
                    agent_data = self.agent_db.load_agent(symbol)
                    
                    if agent_data and self._validate_agent_performance(agent_data):
                        # Set as champion
                        self.agent_db.set_champion_agent(symbol, agent_id)
                        champions[symbol] = agent_id
                        
                        self.logger.info(f"👑 {symbol} agent set as champion: {agent_id}")
                    else:
                        self.logger.warning(f"⚠️ {symbol} agent failed validation")
                        
                except Exception as e:
                    self.logger.error(f"❌ Validation failed for {symbol}: {str(e)}")
        
        return champions
    
    def _validate_agent_performance(self, agent_data: Dict) -> bool:
        """Validate agent meets minimum performance requirements."""
        
        # Performance thresholds
        min_accuracy = 0.75  # Minimum 75% accuracy
        min_return = 0.10    # Minimum 10% return
        min_sharpe = 0.8     # Minimum Sharpe ratio
        max_drawdown = 0.20  # Maximum 20% drawdown
        
        accuracy = agent_data.get('accuracy', 0)
        total_return = agent_data.get('total_return', 0)
        sharpe_ratio = agent_data.get('sharpe_ratio', 0)
        max_dd = agent_data.get('max_drawdown', 1.0)
        
        # Check all criteria
        criteria_met = (
            accuracy >= min_accuracy and
            total_return >= min_return and
            sharpe_ratio >= min_sharpe and
            max_dd <= max_drawdown
        )
        
        if not criteria_met:
            self.logger.warning(f"Agent validation failed:")
            self.logger.warning(f"  Accuracy: {accuracy:.1%} (min: {min_accuracy:.1%})")
            self.logger.warning(f"  Return: {total_return:.1%} (min: {min_return:.1%})")
            self.logger.warning(f"  Sharpe: {sharpe_ratio:.2f} (min: {min_sharpe:.2f})")
            self.logger.warning(f"  Drawdown: {max_dd:.1%} (max: {max_drawdown:.1%})")
        
        return criteria_met
    
    def _generate_training_report(self, champions: Dict[str, str]) -> Dict:
        """Generate comprehensive training report."""
        
        report = {
            'training_date': datetime.now().isoformat(),
            'total_stocks': len(self.top_stocks),
            'successful_training': len([r for r in champions.values() if r]),
            'champion_agents': len(champions),
            'success_rate': len(champions) / len(self.top_stocks),
            'training_config': self.training_config,
            'champions': champions,
            'performance_summary': {}
        }
        
        # Collect performance summary
        total_accuracy = 0
        total_return = 0
        total_sharpe = 0
        
        for symbol, agent_id in champions.items():
            try:
                agent_data = self.agent_db.load_agent(symbol)
                if agent_data:
                    report['performance_summary'][symbol] = {
                        'accuracy': agent_data.get('accuracy', 0),
                        'total_return': agent_data.get('total_return', 0),
                        'sharpe_ratio': agent_data.get('sharpe_ratio', 0),
                        'agent_id': agent_id
                    }
                    
                    total_accuracy += agent_data.get('accuracy', 0)
                    total_return += agent_data.get('total_return', 0)
                    total_sharpe += agent_data.get('sharpe_ratio', 0)
                    
            except Exception as e:
                self.logger.error(f"Failed to get performance for {symbol}: {str(e)}")
        
        # Calculate averages
        if champions:
            report['average_performance'] = {
                'accuracy': total_accuracy / len(champions),
                'total_return': total_return / len(champions),
                'sharpe_ratio': total_sharpe / len(champions)
            }
        
        # Save report
        report_path = f"data/training_reports/initial_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            import os
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"📄 Training report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {str(e)}")
        
        # Print summary
        self._print_training_summary(report)
        
        return report
    
    def _print_training_summary(self, report: Dict):
        """Print training summary to console."""
        
        print("\n" + "="*60)
        print("🎯 INITIAL AGENT TRAINING SUMMARY")
        print("="*60)
        print(f"Training Date: {report['training_date']}")
        print(f"Stocks Trained: {report['total_stocks']}")
        print(f"Successful Training: {report['successful_training']}")
        print(f"Champion Agents: {report['champion_agents']}")
        print(f"Success Rate: {report['success_rate']:.1%}")
        
        if 'average_performance' in report:
            avg_perf = report['average_performance']
            print(f"\n📊 Average Performance:")
            print(f"  Accuracy: {avg_perf['accuracy']:.1%}")
            print(f"  Total Return: {avg_perf['total_return']:.1%}")
            print(f"  Sharpe Ratio: {avg_perf['sharpe_ratio']:.2f}")
        
        print(f"\n👑 Champion Agents:")
        for symbol, agent_id in report['champions'].items():
            perf = report['performance_summary'].get(symbol, {})
            print(f"  {symbol}: {agent_id[:8]}... "
                  f"(Acc: {perf.get('accuracy', 0):.1%}, "
                  f"Ret: {perf.get('total_return', 0):.1%}, "
                  f"Sharpe: {perf.get('sharpe_ratio', 0):.2f})")
        
        print("="*60)
    
    def get_training_status(self) -> Dict:
        """Get current training status."""
        
        status = {
            'active_sessions': self.agent_db.get_active_training_sessions(),
            'completed_agents': [],
            'champion_count': 0
        }
        
        # Check for completed agents
        for symbol in self.top_stocks:
            agent_data = self.agent_db.load_agent(symbol)
            if agent_data:
                status['completed_agents'].append({
                    'symbol': symbol,
                    'agent_id': agent_data['agent_id'],
                    'is_champion': agent_data.get('is_champion', False),
                    'accuracy': agent_data.get('accuracy', 0),
                    'total_return': agent_data.get('total_return', 0)
                })
                
                if agent_data.get('is_champion', False):
                    status['champion_count'] += 1
        
        return status


def run_initial_training():
    """Run the initial agent training pipeline."""
    
    print("🚀 NEAT Options System - Initial Agent Training")
    print("="*60)
    
    # Create training pipeline
    pipeline = InitialAgentTrainingPipeline()
    
    try:
        # Run training
        champions = pipeline.run_initial_training()
        
        print(f"\n✅ Initial training completed!")
        print(f"Champion agents trained: {len(champions)}")
        
        return champions
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_initial_training()
