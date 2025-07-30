import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union

class PaperTradingSystem:
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001, 
                 risk_per_trade: float = 0.01, stop_loss_pct: float = 0.02, 
                 take_profit_pct: float = 0.04):
        """
        Initialize the paper trading system.
        
        Args:
            initial_capital: Starting capital in dollars
            commission: Commission per trade as a percentage of trade value
            risk_per_trade: Percentage of capital to risk per trade
            stop_loss_pct: Stop loss percentage (e.g., 0.02 for 2%)
            take_profit_pct: Take profit percentage (e.g., 0.04 for 4%)
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.positions: Dict = {}
        self.trade_history: List[Dict] = []
        self.portfolio_history = [{'date': None, 'value': initial_capital}]
        self.current_date = None
    
    def update_market_data(self, symbol: str, price_data: pd.Series):
        """
        Update market data and check for stop loss/take profit triggers.
        
        Args:
            symbol: Stock symbol
            price_data: Pandas Series with 'Open', 'High', 'Low', 'Close' prices
        """
        self.current_date = price_data.name if hasattr(price_data, 'name') else datetime.now()
        
        if symbol in self.positions:
            position = self.positions[symbol]
            current_price = price_data['Close']
            
            # Check stop loss and take profit
            if position['position_type'] == 'LONG':
                if current_price <= position['stop_loss']:
                    self.close_position(symbol, position['stop_loss'], 'STOP_LOSS')
                elif current_price >= position['take_profit']:
                    self.close_position(symbol, position['take_profit'], 'TAKE_PROFIT')
            else:  # SHORT
                if current_price >= position['stop_loss']:
                    self.close_position(symbol, position['stop_loss'], 'STOP_LOSS')
                elif current_price <= position['take_profit']:
                    self.close_position(symbol, position['take_profit'], 'TAKE_PROFIT')
    
    def execute_trade(self, symbol: str, price: float, signal: int, 
                     date: Optional[datetime] = None, 
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None):
        """
        Execute a trade based on signal.
        
        Args:
            symbol: Stock symbol
            price: Current price
            signal: 1 for long, -1 for short, 0 for no trade
            date: Trade datetime
            stop_loss: Custom stop loss price (optional)
            take_profit: Custom take profit price (optional)
        """
        if signal == 0 or price <= 0:
            return
            
        date = date or self.current_date or datetime.now()
        
        # Close opposite position if exists
        if symbol in self.positions:
            if (self.positions[symbol]['position_type'] == 'LONG' and signal == -1) or \
               (self.positions[symbol]['position_type'] == 'SHORT' and signal == 1):
                self.close_position(symbol, price, 'REVERSAL')
        
        # Calculate position size
        position_size = self.capital * self.risk_per_trade / self.stop_loss_pct
        position_size = min(position_size, self.capital * 0.95)  # Max 95% of capital
        
        if position_size < 10:  # Minimum position size
            return
            
        shares = position_size / price
        commission = position_size * self.commission
        
        # Set stop loss and take profit
        if stop_loss is None:
            stop_loss = price * (1 - self.stop_loss_pct * signal)
        if take_profit is None:
            take_profit = price * (1 + self.take_profit_pct * signal)
        
        # Record position
        self.positions[symbol] = {
            'entry_price': price,
            'shares': shares,
            'position_type': 'LONG' if signal > 0 else 'SHORT',
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_date': date,
            'exit_date': None,
            'exit_price': None,
            'pnl': 0,
            'pnl_pct': 0,
            'status': 'OPEN'
        }
        
        self.capital -= commission
        self._record_trade(symbol, 'OPEN', price, shares, commission, date)
    
    def close_position(self, symbol: str, price: float, exit_reason: str = 'MANUAL'):
        """Close an existing position"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        entry_value = position['entry_price'] * position['shares']
        exit_value = price * position['shares']
        
        if position['position_type'] == 'LONG':
            pnl = exit_value - entry_value
        else:  # SHORT
            pnl = entry_value - exit_value
            
        pnl_pct = (pnl / entry_value) * 100
        commission = exit_value * self.commission
        pnl_after_commission = pnl - commission
        
        # Update capital
        self.capital += exit_value - commission
        
        # Update position
        position.update({
            'exit_price': price,
            'exit_date': self.current_date or datetime.now(),
            'pnl': pnl_after_commission,
            'pnl_pct': pnl_pct,
            'status': 'CLOSED',
            'exit_reason': exit_reason
        })
        
        # Record trade
        self._record_trade(
            symbol, 'CLOSE', price, position['shares'], 
            commission, position['exit_date'], pnl_after_commission, pnl_pct, exit_reason
        )
        
        # Remove from active positions
        del self.positions[symbol]
    
    def _record_trade(self, symbol: str, action: str, price: float, shares: float, 
                     commission: float, date: datetime, pnl: float = 0, 
                     pnl_pct: float = 0, exit_reason: str = None):
        """Record trade details"""
        trade = {
            'date': date,
            'symbol': symbol,
            'action': action,
            'price': price,
            'shares': shares,
            'value': price * shares,
            'commission': commission,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'portfolio_value': self.capital,
            'exit_reason': exit_reason
        }
        
        self.trade_history.append(trade)
        self.portfolio_history.append({
            'date': date,
            'value': self.capital,
            'cash': self.cash_value(),
            'positions_value': self.positions_value(price if symbol in self.positions else None)
        })
    
    def cash_value(self) -> float:
        """Get current cash balance"""
        return self.capital - sum(
            pos['entry_price'] * pos['shares'] 
            for pos in self.positions.values()
        )
    
    def positions_value(self, current_price: float = None) -> Dict[str, float]:
        """Get current value of all positions"""
        if not self.positions:
            return {}
            
        if current_price is None:
            return {
                symbol: pos['entry_price'] * pos['shares']
                for symbol, pos in self.positions.items()
            }
        else:
            return {
                symbol: current_price * pos['shares']
                for symbol, pos in self.positions.items()
            }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate and return performance metrics"""
        if not self.trade_history:
            return {}
            
        df = pd.DataFrame(self.portfolio_history).dropna()
        
        if len(df) < 2:
            return {}
        
        # Calculate returns
        df['returns'] = df['value'].pct_change().fillna(0)
        
        # Basic metrics
        total_return = (df['value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Annualized return
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        annualized_return = (1 + total_return/100) ** (365/max(1, days)) - 1 if days > 0 else 0
        
        # Volatility
        volatility = df['returns'].std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe_ratio = (df['returns'].mean() / df['returns'].std()) * np.sqrt(252) if df['returns'].std() > 0 else 0
        
        # Max drawdown
        cumulative_returns = (1 + df['returns']).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min() * 100  # as percentage
        
        # Win rate
        closed_trades = [t for t in self.trade_history if t['action'] == 'CLOSE']
        win_rate = (len([t for t in closed_trades if t['pnl'] > 0]) / len(closed_trades)) * 100 if closed_trades else 0
        
        # Average win/loss
        winning_trades = [t['pnl_pct'] for t in closed_trades if t['pnl'] > 0]
        losing_trades = [t['pnl_pct'] for t in closed_trades if t['pnl'] <= 0]
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in closed_trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in closed_trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': df['value'].iloc[-1],
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return * 100,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Get trade history as a pandas DataFrame"""
        if not self.trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_history)
    
    def get_portfolio_history_df(self) -> pd.DataFrame:
        """Get portfolio history as a pandas DataFrame"""
        if not self.portfolio_history:
            return pd.DataFrame()
        return pd.DataFrame(self.portfolio_history).dropna()
