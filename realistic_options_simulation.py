#!/usr/bin/env python3
"""
REALISTIC OPTIONS TRADING SIMULATION
Fixes critical issues identified by user:
1. Proper options expiration cycles (weekly/monthly)
2. Realistic holding periods (7-45 days)
3. Roll costs and transaction fees
4. Proper strike price selection
5. Real options mechanics validation
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealisticOptionsSimulation:
    """Realistic cash-secured put simulation with proper options mechanics."""
    
    def __init__(self):
        # Realistic transaction costs
        self.option_commission = 0.65  # Per contract
        self.roll_cost = 1.30  # Close + open new position
        
        # Options expiration cycles (Fridays)
        self.weekly_options = True
        self.monthly_options = True
        
        # Realistic parameters
        self.min_days_to_expiration = 7   # Minimum 1 week
        self.max_days_to_expiration = 45  # Maximum ~6 weeks
        self.min_premium_yield = 0.005    # 0.5% minimum premium
        
    def get_next_expiration_date(self, current_date, days_out=14):
        """Get next realistic options expiration date (Friday)."""
        target_date = current_date + timedelta(days=days_out)
        
        # Find next Friday
        days_ahead = 4 - target_date.weekday()  # Friday is 4
        if days_ahead <= 0:
            days_ahead += 7
        
        expiration = target_date + timedelta(days=days_ahead)
        return expiration
    
    def calculate_realistic_strike(self, current_price, confidence, days_to_exp):
        """Calculate realistic strike price for cash-secured puts."""
        # Strike selection based on delta (typically 15-30 delta)
        # Higher confidence = closer to money (higher premium)
        
        if days_to_exp <= 7:
            # Weekly options - more conservative
            strike_discount = 0.02 + (1 - confidence) * 0.03  # 2-5% OTM
        else:
            # Monthly options - can be more aggressive
            strike_discount = 0.03 + (1 - confidence) * 0.05  # 3-8% OTM
        
        strike = current_price * (1 - strike_discount)
        
        # Round to nearest $0.50 or $1.00 (realistic strike increments)
        if strike > 100:
            strike = round(strike)
        else:
            strike = round(strike * 2) / 2
            
        return strike
    
    def calculate_realistic_premium(self, current_price, strike, days_to_exp, confidence):
        """Calculate realistic option premium based on market conditions."""
        # Simplified Black-Scholes approximation for premium
        moneyness = strike / current_price
        time_value = np.sqrt(days_to_exp / 365)
        
        # Base premium calculation
        intrinsic = max(0, strike - current_price)
        
        # Time value based on implied volatility (simplified)
        implied_vol = 0.20 + confidence * 0.15  # 20-35% IV range
        time_premium = current_price * implied_vol * time_value * 0.4
        
        total_premium = intrinsic + time_premium
        
        # Premium as percentage of strike price
        premium_yield = total_premium / strike
        
        # Ensure minimum premium yield
        if premium_yield < self.min_premium_yield:
            premium_yield = self.min_premium_yield
            
        return premium_yield
    
    def simulate_realistic_trading(self, prices, dates, predictions):
        """Simulate realistic options trading with proper mechanics."""
        logger.info("🎯 Starting REALISTIC options simulation")
        logger.info(f"   Data points: {len(prices)}")
        logger.info(f"   Date range: {dates[0]} to {dates[-1]}")
        
        trades = []
        position = None
        total_return = 0
        wins = 0
        assignments = 0
        
        i = 0
        while i < len(prices) - 1:
            current_date = dates[i]
            current_price = prices[i]
            prediction = predictions[i]
            
            # Entry logic - only enter if no position and high confidence
            if position is None and prediction > 0.6:
                # Calculate realistic expiration (2-4 weeks out)
                days_to_exp = int(14 + prediction * 21)  # 14-35 days
                expiration_date = self.get_next_expiration_date(current_date, days_to_exp)
                
                # Calculate realistic strike
                strike = self.calculate_realistic_strike(current_price, prediction, days_to_exp)
                
                # Calculate realistic premium
                premium_yield = self.calculate_realistic_premium(
                    current_price, strike, days_to_exp, prediction
                )
                
                # Enter position
                position = {
                    'entry_date': current_date,
                    'entry_price': current_price,
                    'strike': strike,
                    'expiration': expiration_date,
                    'premium_yield': premium_yield,
                    'confidence': prediction,
                    'days_to_exp': days_to_exp
                }
                
                logger.info(f"📈 ENTERED: {current_date.date()} | Price: ${current_price:.2f} | Strike: ${strike:.2f} | Premium: {premium_yield:.3f} ({premium_yield*100:.1f}%)")
                
            # Exit logic - check if we have a position
            elif position is not None:
                days_in_trade = (current_date - position['entry_date']).days
                days_to_exp = (position['expiration'] - current_date).days
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                # 1. Expiration reached
                if days_to_exp <= 0:
                    should_exit = True
                    exit_reason = "EXPIRATION"
                    
                    # Check assignment
                    if current_price < position['strike']:
                        # Assigned - lose money
                        pnl = position['premium_yield'] - (position['strike'] - current_price) / position['strike']
                        assignments += 1
                    else:
                        # Expired worthless - keep premium
                        pnl = position['premium_yield']
                        
                # 2. Early profit taking (50% of premium captured)
                elif days_in_trade >= 7 and position['premium_yield'] > 0.02:
                    profit_target = position['premium_yield'] * 0.5
                    if current_price > position['strike'] * 1.02:  # Well above strike
                        should_exit = True
                        exit_reason = "PROFIT_TARGET"
                        pnl = profit_target - self.roll_cost / (position['strike'] * 100)  # Roll cost
                        
                # 3. Loss management
                elif current_price < position['strike'] * 0.95:  # 5% below strike
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                    loss = (position['strike'] - current_price) / position['strike']
                    pnl = position['premium_yield'] - loss - self.roll_cost / (position['strike'] * 100)
                
                if should_exit:
                    # Account for transaction costs
                    net_pnl = pnl - (self.option_commission / (position['strike'] * 100))
                    
                    trade_record = {
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'strike': position['strike'],
                        'premium_yield': position['premium_yield'],
                        'days_held': days_in_trade,
                        'pnl': net_pnl,
                        'exit_reason': exit_reason,
                        'assigned': exit_reason == "EXPIRATION" and current_price < position['strike']
                    }
                    
                    trades.append(trade_record)
                    total_return += net_pnl
                    
                    if net_pnl > 0:
                        wins += 1
                        
                    logger.info(f"📉 EXITED: {current_date.date()} | {exit_reason} | Days: {days_in_trade} | P&L: {net_pnl:.3f} ({net_pnl*100:.1f}%)")
                    
                    position = None
                    
                    # Skip ahead to avoid immediate re-entry
                    i += max(1, days_in_trade // 3)
                    continue
            
            i += 1
        
        # Calculate final metrics
        if trades:
            win_rate = wins / len(trades)
            avg_return_per_trade = np.mean([t['pnl'] for t in trades])
            annual_return = avg_return_per_trade * (252 / np.mean([t['days_held'] for t in trades]))
            assignment_rate = assignments / len(trades)
            
            logger.info("📊 REALISTIC SIMULATION RESULTS:")
            logger.info(f"   Total Trades: {len(trades)}")
            logger.info(f"   Win Rate: {win_rate:.1%}")
            logger.info(f"   Assignment Rate: {assignment_rate:.1%}")
            logger.info(f"   Avg Return/Trade: {avg_return_per_trade:.3f} ({avg_return_per_trade*100:.1f}%)")
            logger.info(f"   Annual Return: {annual_return:.1%}")
            logger.info(f"   Total Return: {total_return:.1%}")
            
            return {
                'trades': trades,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'assignment_rate': assignment_rate,
                'annual_return': annual_return,
                'total_return': total_return
            }
        else:
            logger.warning("⚠️ No trades generated in simulation")
            return {
                'trades': [],
                'total_trades': 0,
                'win_rate': 0,
                'assignment_rate': 0,
                'annual_return': 0,
                'total_return': 0
            }

def test_realistic_simulation():
    """Test the realistic options simulation."""
    logger.info("🚀 Testing Realistic Options Simulation")
    
    # Get real AAPL data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1y")
    
    prices = data['Close'].values
    dates = data.index
    
    # Generate sample predictions (replace with real NEAT predictions)
    np.random.seed(42)
    predictions = np.random.uniform(0.3, 0.8, len(prices))
    
    # Run realistic simulation
    sim = RealisticOptionsSimulation()
    results = sim.simulate_realistic_trading(prices, dates, predictions)
    
    return results

if __name__ == "__main__":
    results = test_realistic_simulation()
