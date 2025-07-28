#!/usr/bin/env python3
"""
SwingTradeAI - Professional Strategy Reports Generator
User Acceptance: User gets professional-quality strategy reports
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import json

from .advanced_backtester import BacktestResult, RegimePerformance
from .model_competition import CompetitionResult, StrategyConfig
from .scenario_analyzer import ScenarioResult, PortfolioOptimization
from ..core.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class ReportSection:
    """Individual report section"""
    title: str
    content: str
    charts: List[str]  # Base64 encoded chart images
    tables: List[pd.DataFrame]
    key_metrics: Dict[str, Any]
    recommendations: List[str]

@dataclass
class StrategyReport:
    """Complete professional strategy report"""
    report_id: str
    generation_date: datetime
    strategy_name: str
    symbol: str
    executive_summary: str
    sections: List[ReportSection]
    appendices: Dict[str, Any]
    disclaimer: str
    report_metadata: Dict[str, Any]

class StrategyReporter:
    """Production-ready professional strategy reporting system"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
    async def generate_comprehensive_report(
        self,
        backtest_result: BacktestResult,
        competition_result: Optional[CompetitionResult] = None,
        scenario_results: Optional[Dict[str, ScenarioResult]] = None,
        portfolio_optimization: Optional[PortfolioOptimization] = None
    ) -> StrategyReport:
        """Generate comprehensive professional strategy report"""
        
        try:
            report_id = f"report_{backtest_result.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"📊 Generating comprehensive report: {report_id}")
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(backtest_result)
            
            # Generate report sections
            sections = []
            
            # 1. Performance Analysis
            performance_section = await self._generate_performance_section(backtest_result)
            sections.append(performance_section)
            
            # 2. Risk Analysis
            risk_section = await self._generate_risk_section(backtest_result)
            sections.append(risk_section)
            
            # 3. Trade Analysis
            trade_section = await self._generate_trade_section(backtest_result)
            sections.append(trade_section)
            
            # 4. Competition Analysis (if available)
            if competition_result:
                competition_section = await self._generate_competition_section(competition_result)
                sections.append(competition_section)
            
            # Create final report
            report = StrategyReport(
                report_id=report_id,
                generation_date=datetime.now(),
                strategy_name=backtest_result.strategy_name,
                symbol=backtest_result.symbol,
                executive_summary=executive_summary,
                sections=sections,
                appendices=self._generate_appendices(backtest_result),
                disclaimer=self._generate_disclaimer(),
                report_metadata={
                    'analysis_period_days': (backtest_result.end_date - backtest_result.start_date).days,
                    'total_trades': backtest_result.total_trades,
                    'data_quality_score': self._calculate_data_quality_score(backtest_result),
                    'report_confidence': self._calculate_report_confidence(backtest_result)
                }
            )
            
            # Save report
            await self._save_report(report)
            
            logger.info(f"✅ Comprehensive report generated: {report_id}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate comprehensive report: {str(e)}")
            raise
    
    def _generate_executive_summary(self, backtest_result: BacktestResult) -> str:
        """Generate executive summary"""
        
        assessment = self._assess_overall_performance(backtest_result)
        recommendations = self._generate_executive_recommendations(backtest_result)
        
        return f"""
# Executive Summary

## Strategy Overview
**Strategy Name**: {backtest_result.strategy_name}
**Symbol**: {backtest_result.symbol}
**Analysis Period**: {backtest_result.start_date.strftime('%Y-%m-%d')} to {backtest_result.end_date.strftime('%Y-%m-%d')}
**Report Date**: {datetime.now().strftime('%Y-%m-%d')}

## Key Performance Metrics
- **Total Return**: {backtest_result.total_return:.2f}%
- **Annualized Return**: {backtest_result.annualized_return:.2f}%
- **Sharpe Ratio**: {backtest_result.sharpe_ratio:.2f}
- **Maximum Drawdown**: {backtest_result.max_drawdown:.2f}%
- **Win Rate**: {backtest_result.win_rate:.1f}%

## Overall Assessment
{assessment}

## Key Recommendations
{recommendations}
        """
    
    async def _generate_performance_section(self, result: BacktestResult) -> ReportSection:
        """Generate performance analysis section"""
        
        benchmark_comparison = "outperformed" if result.benchmark_comparison.get('alpha', 0) > 0 else "underperformed"
        sharpe_assessment = self._assess_sharpe_ratio(result.sharpe_ratio)
        
        content = f"""
# Performance Analysis

## Return Characteristics
The strategy generated a total return of {result.total_return:.2f}% over the analysis period, 
corresponding to an annualized return of {result.annualized_return:.2f}%. This performance 
{benchmark_comparison} the benchmark return of {result.benchmark_comparison.get('benchmark_return', 0):.2f}%.

## Risk-Adjusted Performance
With a Sharpe ratio of {result.sharpe_ratio:.2f}, the strategy demonstrates {sharpe_assessment} 
risk-adjusted performance. The maximum drawdown of {result.max_drawdown:.2f}% indicates 
{self._assess_drawdown(result.max_drawdown)} downside risk management.

## Trading Characteristics
The strategy executed {result.total_trades} trades with a win rate of {result.win_rate:.1f}%, 
suggesting {self._assess_consistency(result.win_rate)} execution consistency.
        """
        
        # Create performance metrics table
        performance_metrics = pd.DataFrame({
            'Metric': ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate'],
            'Value': [
                f"{result.total_return:.2f}%",
                f"{result.annualized_return:.2f}%",
                f"{result.volatility:.2f}%",
                f"{result.sharpe_ratio:.2f}",
                f"{result.max_drawdown:.2f}%",
                f"{result.win_rate:.1f}%"
            ],
            'Assessment': [
                self._assess_return(result.total_return),
                self._assess_return(result.annualized_return),
                self._assess_volatility(result.volatility),
                sharpe_assessment,
                self._assess_drawdown(result.max_drawdown),
                self._assess_consistency(result.win_rate)
            ]
        })
        
        return ReportSection(
            title="Performance Analysis",
            content=content,
            charts=[],
            tables=[performance_metrics],
            key_metrics={
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'alpha': result.benchmark_comparison.get('alpha', 0)
            },
            recommendations=self._generate_performance_recommendations(result)
        )
    
    async def _generate_risk_section(self, result: BacktestResult) -> ReportSection:
        """Generate risk analysis section"""
        
        content = f"""
# Risk Analysis

## Drawdown Analysis
The strategy experienced a maximum drawdown of {result.max_drawdown:.2f}%, indicating 
{self._assess_drawdown(result.max_drawdown)} risk management.

## Volatility Analysis
The strategy exhibited an annualized volatility of {result.volatility:.2f}%, which is 
{self._assess_volatility(result.volatility)} compared to typical market volatility.

## Risk Metrics
- **Value at Risk (95%)**: {result.risk_metrics.get('var_95', 0):.2f}%
- **Expected Shortfall**: {result.risk_metrics.get('cvar_95', 0):.2f}%
- **Downside Deviation**: {result.risk_metrics.get('downside_deviation', 0):.2f}%
        """
        
        # Create risk metrics table
        risk_metrics_table = pd.DataFrame({
            'Risk Metric': ['Maximum Drawdown', 'Volatility', 'VaR (95%)', 'Expected Shortfall', 'Downside Deviation'],
            'Value': [
                f"{result.max_drawdown:.2f}%",
                f"{result.volatility:.2f}%",
                f"{result.risk_metrics.get('var_95', 0):.2f}%",
                f"{result.risk_metrics.get('cvar_95', 0):.2f}%",
                f"{result.risk_metrics.get('downside_deviation', 0):.2f}%"
            ],
            'Assessment': [
                self._assess_drawdown(result.max_drawdown),
                self._assess_volatility(result.volatility),
                self._assess_var(result.risk_metrics.get('var_95', 0)),
                "Moderate",
                "Acceptable"
            ]
        })
        
        return ReportSection(
            title="Risk Analysis",
            content=content,
            charts=[],
            tables=[risk_metrics_table],
            key_metrics={
                'max_drawdown': result.max_drawdown,
                'volatility': result.volatility,
                'var_95': result.risk_metrics.get('var_95', 0)
            },
            recommendations=self._generate_risk_recommendations(result)
        )
    
    async def _generate_trade_section(self, result: BacktestResult) -> ReportSection:
        """Generate trade analysis section"""
        
        if result.trade_history.empty:
            return ReportSection(
                title="Trade Analysis",
                content="No trade history available for analysis.",
                charts=[],
                tables=[],
                key_metrics={},
                recommendations=[]
            )
        
        trades_df = result.trade_history
        avg_holding_period = trades_df['holding_days'].mean() if 'holding_days' in trades_df.columns else 0
        
        content = f"""
# Trade Analysis

## Trading Statistics
- **Total Trades**: {result.total_trades}
- **Win Rate**: {result.win_rate:.1f}%
- **Average Holding Period**: {avg_holding_period:.1f} days

## Trade Distribution
The strategy shows {self._assess_consistency(result.win_rate)} consistency in trade execution.
        """
        
        # Create trade summary table
        trade_summary = pd.DataFrame({
            'Metric': ['Total Trades', 'Win Rate', 'Avg Holding Period'],
            'Value': [
                result.total_trades,
                f"{result.win_rate:.1f}%",
                f"{avg_holding_period:.1f} days"
            ]
        })
        
        return ReportSection(
            title="Trade Analysis",
            content=content,
            charts=[],
            tables=[trade_summary],
            key_metrics={
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'avg_holding_period': avg_holding_period
            },
            recommendations=self._generate_trade_recommendations(result)
        )
    
    async def _generate_competition_section(self, competition_result: CompetitionResult) -> ReportSection:
        """Generate competition analysis section"""
        
        content = f"""
# Strategy Competition Analysis

## Competition Overview
- **Strategies Tested**: {len(competition_result.strategies)}
- **Best Strategy**: {competition_result.best_strategy}
- **Analysis Period**: {competition_result.period_start.strftime('%Y-%m-%d')} to {competition_result.period_end.strftime('%Y-%m-%d')}

## Performance Rankings
The competition analyzed multiple strategies with varying performance characteristics.

## Key Insights
The best performing strategy was {competition_result.best_strategy} based on risk-adjusted returns.
        """
        
        return ReportSection(
            title="Strategy Competition Analysis",
            content=content,
            charts=[],
            tables=[competition_result.performance_matrix],
            key_metrics={
                'strategies_tested': len(competition_result.strategies),
                'best_strategy': competition_result.best_strategy
            },
            recommendations=competition_result.recommendations
        )
    
    def _generate_appendices(self, result: BacktestResult) -> Dict[str, Any]:
        """Generate report appendices"""
        
        return {
            'methodology': "Mean reversion and momentum strategies with regime detection",
            'assumptions': "Historical data patterns continue, no structural market changes",
            'data_sources': "Yahoo Finance, real market data only",
            'raw_data': {
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown
            }
        }
    
    def _generate_disclaimer(self) -> str:
        """Generate legal disclaimer"""
        
        return """
## Important Disclaimer

This report is for informational and educational purposes only. Past performance does not guarantee future results. 
All investments carry risk of loss. Please consult with a qualified financial advisor before making investment decisions.
        """
    
    # Assessment helper methods
    def _assess_overall_performance(self, result: BacktestResult) -> str:
        """Assess overall strategy performance"""
        
        if result.sharpe_ratio > 2.0 and result.total_return > 20:
            return "The strategy demonstrates exceptional performance with strong risk-adjusted returns."
        elif result.sharpe_ratio > 1.5 and result.total_return > 10:
            return "The strategy shows strong performance with good risk management."
        elif result.sharpe_ratio > 1.0 and result.total_return > 5:
            return "The strategy exhibits moderate performance with acceptable risk levels."
        else:
            return "The strategy shows below-average performance and may require optimization."
    
    def _assess_sharpe_ratio(self, sharpe: float) -> str:
        """Assess Sharpe ratio"""
        if sharpe > 2.0:
            return "excellent"
        elif sharpe > 1.5:
            return "very good"
        elif sharpe > 1.0:
            return "good"
        else:
            return "moderate"
    
    def _assess_drawdown(self, drawdown: float) -> str:
        """Assess maximum drawdown"""
        if drawdown < 5:
            return "excellent"
        elif drawdown < 10:
            return "good"
        elif drawdown < 15:
            return "moderate"
        else:
            return "concerning"
    
    def _assess_consistency(self, win_rate: float) -> str:
        """Assess trading consistency"""
        if win_rate > 70:
            return "excellent"
        elif win_rate > 60:
            return "good"
        elif win_rate > 50:
            return "moderate"
        else:
            return "poor"
    
    def _assess_return(self, return_pct: float) -> str:
        """Assess return level"""
        if return_pct > 25:
            return "Excellent"
        elif return_pct > 15:
            return "Very Good"
        elif return_pct > 10:
            return "Good"
        else:
            return "Moderate"
    
    def _assess_volatility(self, volatility: float) -> str:
        """Assess volatility level"""
        if volatility < 10:
            return "low"
        elif volatility < 20:
            return "moderate"
        else:
            return "high"
    
    def _assess_var(self, var: float) -> str:
        """Assess Value at Risk"""
        if abs(var) < 2:
            return "Low Risk"
        elif abs(var) < 5:
            return "Moderate Risk"
        else:
            return "High Risk"
    
    # Recommendation generators
    def _generate_executive_recommendations(self, result: BacktestResult) -> str:
        """Generate executive recommendations"""
        
        recommendations = []
        
        if result.sharpe_ratio < 1.0:
            recommendations.append("• Consider optimizing risk management parameters")
        
        if result.max_drawdown > 15:
            recommendations.append("• Implement stricter stop-loss controls")
        
        if result.win_rate < 50:
            recommendations.append("• Review entry criteria and market timing")
        
        if result.total_trades < 10:
            recommendations.append("• Consider increasing trading frequency for better statistics")
        
        return "\n".join(recommendations) if recommendations else "• Strategy performance is within acceptable parameters"
    
    def _generate_performance_recommendations(self, result: BacktestResult) -> List[str]:
        """Generate performance-specific recommendations"""
        
        recommendations = []
        
        if result.total_return < 10:
            recommendations.append("Consider strategies to improve absolute returns")
        
        if result.sharpe_ratio < 1.5:
            recommendations.append("Focus on improving risk-adjusted performance")
        
        alpha = result.benchmark_comparison.get('alpha', 0)
        if alpha < 0:
            recommendations.append("Strategy underperforms benchmark - consider index investing")
        
        return recommendations
    
    def _generate_risk_recommendations(self, result: BacktestResult) -> List[str]:
        """Generate risk-specific recommendations"""
        
        recommendations = []
        
        if result.max_drawdown > 20:
            recommendations.append("Implement position sizing controls to reduce drawdown")
        
        if result.volatility > 25:
            recommendations.append("Consider volatility targeting to smooth returns")
        
        return recommendations
    
    def _generate_trade_recommendations(self, result: BacktestResult) -> List[str]:
        """Generate trade-specific recommendations"""
        
        recommendations = []
        
        if result.total_trades < 20:
            recommendations.append("Increase sample size for more robust statistics")
        
        if result.win_rate < 45:
            recommendations.append("Review and optimize entry criteria")
        
        return recommendations
    
    def _calculate_data_quality_score(self, result: BacktestResult) -> float:
        """Calculate data quality score"""
        
        score = 1.0
        
        if result.total_trades < 10:
            score -= 0.3
        
        if (result.end_date - result.start_date).days < 365:
            score -= 0.2
        
        return max(0.0, score)
    
    def _calculate_report_confidence(self, result: BacktestResult) -> float:
        """Calculate report confidence level"""
        
        confidence = 0.8  # Base confidence
        
        if result.total_trades > 50:
            confidence += 0.1
        
        if result.sharpe_ratio > 1.5:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def _save_report(self, report: StrategyReport):
        """Save report to file"""
        
        try:
            reports_dir = Path("data/reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{report.report_id}.json"
            filepath = reports_dir / filename
            
            # Convert to serializable format
            serializable_report = {
                'report_id': report.report_id,
                'generation_date': report.generation_date.isoformat(),
                'strategy_name': report.strategy_name,
                'symbol': report.symbol,
                'executive_summary': report.executive_summary,
                'sections': [
                    {
                        'title': section.title,
                        'content': section.content,
                        'key_metrics': section.key_metrics,
                        'recommendations': section.recommendations
                    }
                    for section in report.sections
                ],
                'disclaimer': report.disclaimer,
                'report_metadata': report.report_metadata
            }
            
            with open(filepath, 'w') as f:
                json.dump(serializable_report, f, indent=2, default=str)
            
            logger.info(f"💾 Report saved to {filepath}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save report: {str(e)}")
