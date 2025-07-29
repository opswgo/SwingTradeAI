#!/usr/bin/env python3
"""
SwingTradeAI - Comprehensive Risk Management System
User Acceptance: User understands and can modify risk parameters
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import logging
from pathlib import Path

from .trade_logger import TradeLogger, TradeEntry, TradeType, TradeStatus

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class RiskParameters:
    """User-configurable risk parameters"""
    # Position Sizing
    max_position_size_pct: float = 5.0  # Max % of portfolio per position
    max_portfolio_risk_pct: float = 2.0  # Max % portfolio risk per trade
    max_total_exposure_pct: float = 20.0  # Max % total portfolio exposure
    
    # Stop Loss & Take Profit
    default_stop_loss_pct: float = 15.0  # Default stop loss %
    default_take_profit_pct: float = 30.0  # Default take profit %
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 10.0
    
    # Portfolio Limits
    max_open_positions: int = 10
    max_daily_trades: int = 5
    max_weekly_trades: int = 20
    
    # Drawdown Protection
    max_daily_loss_pct: float = 5.0
    max_weekly_loss_pct: float = 10.0
    max_monthly_loss_pct: float = 15.0
    
    # Concentration Limits
    max_sector_exposure_pct: float = 30.0
    max_single_symbol_pct: float = 10.0
    
    # Options Specific
    max_options_allocation_pct: float = 50.0
    min_days_to_expiration: int = 7
    max_days_to_expiration: int = 45
    
    # Volatility Controls
    max_iv_percentile: float = 80.0  # Don't buy high IV
    min_liquidity_volume: int = 1000  # Min daily volume

@dataclass
class RiskAssessment:
    """Risk assessment result"""
    approved: bool
    risk_score: float  # 0-100
    warnings: List[str]
    position_size: int
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    max_loss_amount: float
    risk_reward_ratio: float

class RiskManager:
    """Production-ready risk management system"""
    
    def __init__(self, trade_logger: TradeLogger, initial_capital: float = 100000.0):
        self.trade_logger = trade_logger
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_params = RiskParameters()
        self._load_risk_config()
        
    def _load_risk_config(self):
        """Load risk parameters from config file"""
        config_path = Path("config/risk_parameters.json")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    # Update risk parameters with saved values
                    for key, value in config_data.items():
                        if hasattr(self.risk_params, key):
                            setattr(self.risk_params, key, value)
                logger.info("✅ Risk parameters loaded from config")
            except Exception as e:
                logger.warning(f"⚠️ Could not load risk config: {str(e)}")
    
    def save_risk_config(self):
        """Save current risk parameters to config file"""
        try:
            config_path = Path("config/risk_parameters.json")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                'max_position_size_pct': self.risk_params.max_position_size_pct,
                'max_portfolio_risk_pct': self.risk_params.max_portfolio_risk_pct,
                'max_total_exposure_pct': self.risk_params.max_total_exposure_pct,
                'default_stop_loss_pct': self.risk_params.default_stop_loss_pct,
                'default_take_profit_pct': self.risk_params.default_take_profit_pct,
                'max_open_positions': self.risk_params.max_open_positions,
                'max_daily_trades': self.risk_params.max_daily_trades,
                'max_daily_loss_pct': self.risk_params.max_daily_loss_pct,
                'max_weekly_loss_pct': self.risk_params.max_weekly_loss_pct,
                'max_monthly_loss_pct': self.risk_params.max_monthly_loss_pct,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info("✅ Risk parameters saved to config")
            
        except Exception as e:
            logger.error(f"❌ Failed to save risk config: {str(e)}")
    
    def update_capital(self, new_capital: float):
        """Update current capital (from P&L tracking)"""
        self.current_capital = new_capital
        logger.info(f"💰 Capital updated: ${new_capital:,.2f}")
    
    def assess_trade_risk(self, symbol: str, trade_type: TradeType, 
                         entry_price: float, quantity: int,
                         ai_confidence: float, market_conditions: Dict[str, Any]) -> RiskAssessment:
        """Comprehensive trade risk assessment"""
        
        warnings = []
        risk_score = 0.0
        
        # Calculate position value
        position_value = entry_price * quantity
        position_size_pct = (position_value / self.current_capital) * 100
        
        # Position size check
        if position_size_pct > self.risk_params.max_position_size_pct:
            warnings.append(f"Position size {position_size_pct:.1f}% exceeds max {self.risk_params.max_position_size_pct}%")
            risk_score += 30
        
        # Portfolio exposure check
        current_exposure = self._get_current_exposure()
        total_exposure_pct = ((current_exposure + position_value) / self.current_capital) * 100
        
        if total_exposure_pct > self.risk_params.max_total_exposure_pct:
            warnings.append(f"Total exposure {total_exposure_pct:.1f}% exceeds max {self.risk_params.max_total_exposure_pct}%")
            risk_score += 25
        
        # Open positions check
        open_positions = len(self._get_open_positions())
        if open_positions >= self.risk_params.max_open_positions:
            warnings.append(f"Already at max open positions ({open_positions})")
            risk_score += 20
        
        # Daily trades check
        daily_trades = self._get_daily_trade_count()
        if daily_trades >= self.risk_params.max_daily_trades:
            warnings.append(f"Daily trade limit reached ({daily_trades})")
            risk_score += 15
        
        # Drawdown check
        daily_pnl_pct = self._get_daily_pnl_pct()
        if daily_pnl_pct <= -self.risk_params.max_daily_loss_pct:
            warnings.append(f"Daily loss limit reached ({daily_pnl_pct:.1f}%)")
            risk_score += 40
        
        # Symbol concentration check
        symbol_exposure_pct = self._get_symbol_exposure_pct(symbol)
        if symbol_exposure_pct > self.risk_params.max_single_symbol_pct:
            warnings.append(f"Symbol exposure {symbol_exposure_pct:.1f}% exceeds max {self.risk_params.max_single_symbol_pct}%")
            risk_score += 20
        
        # Options-specific checks
        if trade_type in [TradeType.CALL, TradeType.PUT]:
            # Check options allocation
            options_exposure_pct = self._get_options_exposure_pct()
            if options_exposure_pct > self.risk_params.max_options_allocation_pct:
                warnings.append(f"Options exposure {options_exposure_pct:.1f}% exceeds max {self.risk_params.max_options_allocation_pct}%")
                risk_score += 25
            
            # Check days to expiration
            dte = market_conditions.get('days_to_expiration', 0)
            if dte < self.risk_params.min_days_to_expiration:
                warnings.append(f"Days to expiration ({dte}) below minimum ({self.risk_params.min_days_to_expiration})")
                risk_score += 30
            elif dte > self.risk_params.max_days_to_expiration:
                warnings.append(f"Days to expiration ({dte}) above maximum ({self.risk_params.max_days_to_expiration})")
                risk_score += 15
            
            # Check IV percentile
            iv_percentile = market_conditions.get('iv_percentile', 50)
            if iv_percentile > self.risk_params.max_iv_percentile:
                warnings.append(f"IV percentile ({iv_percentile:.1f}) too high (max {self.risk_params.max_iv_percentile})")
                risk_score += 20
        
        # AI confidence adjustment
        if ai_confidence < 0.6:
            warnings.append(f"Low AI confidence ({ai_confidence:.1%})")
            risk_score += 15
        elif ai_confidence > 0.8:
            risk_score -= 10  # Reduce risk for high confidence
        
        # Calculate position sizing
        optimal_quantity = self._calculate_position_size(
            entry_price, symbol, trade_type, ai_confidence
        )
        
        # Calculate stop loss and take profit
        stop_loss_price = self._calculate_stop_loss(entry_price, trade_type)
        take_profit_price = self._calculate_take_profit(entry_price, trade_type)
        
        # Calculate max loss
        if stop_loss_price:
            if trade_type == TradeType.CALL or trade_type == TradeType.STOCK:
                max_loss_amount = (entry_price - stop_loss_price) * optimal_quantity
            else:  # PUT
                max_loss_amount = (stop_loss_price - entry_price) * optimal_quantity
        else:
            max_loss_amount = position_value  # Worst case: total loss
        
        # Calculate risk-reward ratio
        if stop_loss_price and take_profit_price:
            if trade_type == TradeType.CALL or trade_type == TradeType.STOCK:
                risk = entry_price - stop_loss_price
                reward = take_profit_price - entry_price
            else:  # PUT
                risk = stop_loss_price - entry_price
                reward = entry_price - take_profit_price
            
            risk_reward_ratio = reward / risk if risk > 0 else 0
        else:
            risk_reward_ratio = 0
        
        # Final approval decision
        approved = (
            risk_score < 50 and  # Risk score threshold
            len(warnings) == 0 and  # No critical warnings
            risk_reward_ratio >= 1.5  # Minimum risk-reward ratio
        )
        
        return RiskAssessment(
            approved=approved,
            risk_score=min(100, max(0, risk_score)),
            warnings=warnings,
            position_size=optimal_quantity,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            max_loss_amount=abs(max_loss_amount),
            risk_reward_ratio=risk_reward_ratio
        )
    
    def _calculate_position_size(self, entry_price: float, symbol: str, 
                               trade_type: TradeType, ai_confidence: float) -> int:
        """Calculate optimal position size based on risk parameters"""
        
        # Base position size from portfolio risk
        max_risk_amount = self.current_capital * (self.risk_params.max_portfolio_risk_pct / 100)
        
        # Estimate risk per share (using default stop loss)
        stop_loss_pct = self.risk_params.default_stop_loss_pct / 100
        risk_per_share = entry_price * stop_loss_pct
        
        # Calculate max shares from risk
        max_shares_from_risk = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
        
        # Calculate max shares from position size limit
        max_position_value = self.current_capital * (self.risk_params.max_position_size_pct / 100)
        max_shares_from_size = int(max_position_value / entry_price)
        
        # Take the smaller of the two
        base_quantity = min(max_shares_from_risk, max_shares_from_size)
        
        # Adjust for AI confidence
        confidence_multiplier = 0.5 + (ai_confidence * 0.5)  # 0.5 to 1.0
        adjusted_quantity = int(base_quantity * confidence_multiplier)
        
        return max(1, adjusted_quantity)  # Minimum 1 share
    
    def _calculate_stop_loss(self, entry_price: float, trade_type: TradeType) -> Optional[float]:
        """Calculate stop loss price"""
        stop_loss_pct = self.risk_params.default_stop_loss_pct / 100
        
        if trade_type == TradeType.CALL or trade_type == TradeType.STOCK:
            return entry_price * (1 - stop_loss_pct)
        else:  # PUT
            return entry_price * (1 + stop_loss_pct)
    
    def _calculate_take_profit(self, entry_price: float, trade_type: TradeType) -> Optional[float]:
        """Calculate take profit price"""
        take_profit_pct = self.risk_params.default_take_profit_pct / 100
        
        if trade_type == TradeType.CALL or trade_type == TradeType.STOCK:
            return entry_price * (1 + take_profit_pct)
        else:  # PUT
            return entry_price * (1 - take_profit_pct)
    
    def _get_current_exposure(self) -> float:
        """Get current portfolio exposure from open positions"""
        try:
            open_trades = self.trade_logger.get_trade_history(status=TradeStatus.OPEN)
            if open_trades.empty:
                return 0.0
            
            total_exposure = (open_trades['entry_price'] * open_trades['quantity']).sum()
            return float(total_exposure)
            
        except Exception as e:
            logger.error(f"❌ Failed to get current exposure: {str(e)}")
            return 0.0
    
    def _get_open_positions(self) -> List[Dict]:
        """Get list of open positions"""
        try:
            open_trades = self.trade_logger.get_trade_history(status=TradeStatus.OPEN)
            return open_trades.to_dict('records') if not open_trades.empty else []
        except Exception:
            return []
    
    def _get_daily_trade_count(self) -> int:
        """Get number of trades today"""
        try:
            today = datetime.now().date()
            all_trades = self.trade_logger.get_trade_history(limit=None)
            if all_trades.empty:
                return 0
            
            all_trades['entry_date'] = pd.to_datetime(all_trades['entry_timestamp']).dt.date
            today_trades = all_trades[all_trades['entry_date'] == today]
            return len(today_trades)
            
        except Exception:
            return 0
    
    def _get_daily_pnl_pct(self) -> float:
        """Get today's P&L percentage"""
        try:
            today = datetime.now().date()
            all_trades = self.trade_logger.get_trade_history(limit=None)
            if all_trades.empty:
                return 0.0
            
            # Get trades closed today
            closed_today = all_trades[
                (all_trades['status'] == 'closed') &
                (pd.to_datetime(all_trades['exit_timestamp']).dt.date == today)
            ]
            
            if closed_today.empty:
                return 0.0
            
            daily_pnl = closed_today['realized_pnl'].sum()
            return (daily_pnl / self.current_capital) * 100
            
        except Exception:
            return 0.0
    
    def _get_symbol_exposure_pct(self, symbol: str) -> float:
        """Get current exposure percentage for a specific symbol"""
        try:
            symbol_trades = self.trade_logger.get_trade_history(symbol=symbol, status=TradeStatus.OPEN)
            if symbol_trades.empty:
                return 0.0
            
            symbol_exposure = (symbol_trades['entry_price'] * symbol_trades['quantity']).sum()
            return (symbol_exposure / self.current_capital) * 100
            
        except Exception:
            return 0.0
    
    def _get_options_exposure_pct(self) -> float:
        """Get current options exposure percentage"""
        try:
            open_trades = self.trade_logger.get_trade_history(status=TradeStatus.OPEN)
            if open_trades.empty:
                return 0.0
            
            options_trades = open_trades[open_trades['trade_type'].isin(['call', 'put'])]
            if options_trades.empty:
                return 0.0
            
            options_exposure = (options_trades['entry_price'] * options_trades['quantity']).sum()
            return (options_exposure / self.current_capital) * 100
            
        except Exception:
            return 0.0
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio risk summary"""
        try:
            open_positions = self._get_open_positions()
            current_exposure = self._get_current_exposure()
            
            summary = {
                'current_capital': self.current_capital,
                'total_exposure': current_exposure,
                'exposure_pct': (current_exposure / self.current_capital) * 100,
                'open_positions': len(open_positions),
                'available_buying_power': self.current_capital - current_exposure,
                'daily_trades': self._get_daily_trade_count(),
                'daily_pnl_pct': self._get_daily_pnl_pct(),
                'risk_utilization': {
                    'position_count': f"{len(open_positions)}/{self.risk_params.max_open_positions}",
                    'daily_trades': f"{self._get_daily_trade_count()}/{self.risk_params.max_daily_trades}",
                    'exposure_pct': f"{(current_exposure / self.current_capital) * 100:.1f}/{self.risk_params.max_total_exposure_pct}%"
                },
                'risk_parameters': {
                    'max_position_size_pct': self.risk_params.max_position_size_pct,
                    'max_portfolio_risk_pct': self.risk_params.max_portfolio_risk_pct,
                    'default_stop_loss_pct': self.risk_params.default_stop_loss_pct,
                    'max_daily_loss_pct': self.risk_params.max_daily_loss_pct
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get portfolio summary: {str(e)}")
            return {}
    
    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check if any open positions hit stop losses"""
        triggered_stops = []
        
        try:
            open_trades = self.trade_logger.get_trade_history(status=TradeStatus.OPEN)
            
            for _, trade in open_trades.iterrows():
                symbol = trade['symbol']
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                stop_loss = trade['stop_loss']
                
                if stop_loss is None:
                    continue
                
                # Check if stop loss triggered
                stop_triggered = False
                if trade['trade_type'] in ['call', 'stock']:
                    # Long position
                    stop_triggered = current_price <= stop_loss
                else:
                    # Short position (put)
                    stop_triggered = current_price >= stop_loss
                
                if stop_triggered:
                    triggered_stops.append({
                        'trade_id': trade['trade_id'],
                        'symbol': symbol,
                        'trade_type': trade['trade_type'],
                        'current_price': current_price,
                        'stop_loss': stop_loss,
                        'entry_price': trade['entry_price'],
                        'quantity': trade['quantity']
                    })
            
            return triggered_stops
            
        except Exception as e:
            logger.error(f"❌ Failed to check stop losses: {str(e)}")
            return []
