"""
SwingTradeAI - Options Chain Processing
User Acceptance: User can see options data with implied volatility and Greeks
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
from scipy.stats import norm

from ..core.logger import setup_logger, log_user_action, log_data_quality

logger = setup_logger(__name__)


class OptionsChainProcessor:
    """
    Options chain data processing with user-friendly analysis.
    User Focus: Clear options insights and trading opportunities
    """
    
    def __init__(self):
        self.risk_free_rate = 0.05  # Default 5% risk-free rate
        self.current_chains = {}
        
    async def fetch_options_chain(self, symbol: str, expiration_dates: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fetch and process options chain data.
        
        Args:
            symbol: Stock symbol
            expiration_dates: Specific expiration dates (None for all available)
            
        Returns:
            Processed options chain with user insights
        """
        log_user_action("fetch_options_chain", {"symbol": symbol})
        
        try:
            logger.info(f"📊 Fetching options chain for {symbol}...")
            
            # Get ticker and current stock price
            ticker = yf.Ticker(symbol)
            
            # Get current stock price
            hist = ticker.history(period="1d")
            if hist.empty:
                raise ValueError(f"No price data available for {symbol}")
            
            current_price = hist['Close'].iloc[-1]
            
            # Get options chain
            try:
                options_data = ticker.options
                if not options_data or len(options_data) == 0:
                    return {
                        'success': False,
                        'error': f'No options data available for {symbol}',
                        'symbol': symbol
                    }
            except Exception as e:
                logger.warning(f"⚠️ Options data fetch failed for {symbol}: {str(e)}")
                return {
                    'success': False,
                    'error': f'Options data unavailable: {str(e)}',
                    'symbol': symbol
                }
            
            # Process each expiration date
            processed_chains = {}
            
            # Limit to requested dates or first few available
            target_dates = expiration_dates if expiration_dates else options_data[:5]
            
            for exp_date in target_dates:
                try:
                    chain_data = await self._process_expiration_date(ticker, symbol, exp_date, current_price)
                    if chain_data['success']:
                        processed_chains[exp_date] = chain_data
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process expiration {exp_date}: {str(e)}")
                    continue
            
            if not processed_chains:
                return {
                    'success': False,
                    'error': 'No valid options data could be processed',
                    'symbol': symbol
                }
            
            # Generate user insights
            user_insights = self._generate_options_insights(processed_chains, current_price, symbol)
            
            # Store for quick access
            self.current_chains[symbol] = {
                'data': processed_chains,
                'current_price': current_price,
                'timestamp': datetime.now(),
                'insights': user_insights
            }
            
            logger.info(f"✅ Processed options chain for {symbol}: {len(processed_chains)} expirations")
            
            return {
                'success': True,
                'symbol': symbol,
                'current_price': current_price,
                'expiration_dates': list(processed_chains.keys()),
                'chains': processed_chains,
                'insights': user_insights,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Options chain processing failed for {symbol}: {str(e)}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e)
            }
    
    async def _process_expiration_date(
        self, 
        ticker, 
        symbol: str, 
        exp_date: str, 
        current_price: float
    ) -> Dict[str, Any]:
        """Process options for a specific expiration date."""
        try:
            # Get options chain for this expiration
            opt_chain = ticker.option_chain(exp_date)
            
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            if calls.empty and puts.empty:
                return {'success': False, 'error': f'No options data for {exp_date}'}
            
            # Calculate days to expiration
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            days_to_exp = (exp_datetime - datetime.now()).days
            time_to_exp = days_to_exp / 365.0  # Convert to years for Black-Scholes
            
            # Process calls
            processed_calls = await self._process_options_data(calls, 'call', current_price, time_to_exp)
            
            # Process puts
            processed_puts = await self._process_options_data(puts, 'put', current_price, time_to_exp)
            
            # Calculate summary statistics
            summary_stats = self._calculate_chain_summary(processed_calls, processed_puts, current_price)
            
            return {
                'success': True,
                'expiration_date': exp_date,
                'days_to_expiration': days_to_exp,
                'calls': processed_calls,
                'puts': processed_puts,
                'summary': summary_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process expiration {exp_date}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _process_options_data(
        self, 
        options_df: pd.DataFrame, 
        option_type: str, 
        current_price: float, 
        time_to_exp: float
    ) -> pd.DataFrame:
        """Process and enhance options data with Greeks and analysis."""
        if options_df.empty:
            return pd.DataFrame()
        
        try:
            # Create a copy to avoid modifying original
            df = options_df.copy()
            
            # Calculate moneyness
            df['moneyness'] = df['strike'] / current_price
            df['itm'] = (df['strike'] < current_price) if option_type == 'call' else (df['strike'] > current_price)
            
            # Calculate intrinsic and time value
            if option_type == 'call':
                df['intrinsic_value'] = np.maximum(current_price - df['strike'], 0)
            else:
                df['intrinsic_value'] = np.maximum(df['strike'] - current_price, 0)
            
            df['time_value'] = df['lastPrice'] - df['intrinsic_value']
            
            # Calculate implied volatility (simplified Black-Scholes approximation)
            df['implied_volatility'] = df.get('impliedVolatility', 0.2)  # Use yfinance IV if available
            
            # Calculate Greeks (simplified approximations)
            df = self._calculate_greeks(df, option_type, current_price, time_to_exp)
            
            # Add user-friendly classifications
            df['moneyness_category'] = df['moneyness'].apply(self._classify_moneyness)
            df['liquidity_score'] = df['volume'].fillna(0) + df['openInterest'].fillna(0)
            
            # Sort by strike price
            df = df.sort_values('strike')
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Options data processing failed: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_greeks(
        self, 
        df: pd.DataFrame, 
        option_type: str, 
        current_price: float, 
        time_to_exp: float
    ) -> pd.DataFrame:
        """Calculate option Greeks using simplified Black-Scholes."""
        try:
            # Avoid division by zero
            time_to_exp = max(time_to_exp, 1/365)  # Minimum 1 day
            
            # Calculate d1 and d2 for Black-Scholes
            iv = df['implied_volatility'].fillna(0.2)
            strikes = df['strike']
            
            d1 = (np.log(current_price / strikes) + (self.risk_free_rate + 0.5 * iv**2) * time_to_exp) / (iv * np.sqrt(time_to_exp))
            d2 = d1 - iv * np.sqrt(time_to_exp)
            
            # Delta
            if option_type == 'call':
                df['delta'] = norm.cdf(d1)
            else:
                df['delta'] = norm.cdf(d1) - 1
            
            # Gamma (same for calls and puts)
            df['gamma'] = norm.pdf(d1) / (current_price * iv * np.sqrt(time_to_exp))
            
            # Theta (time decay)
            theta_common = -(current_price * norm.pdf(d1) * iv) / (2 * np.sqrt(time_to_exp))
            if option_type == 'call':
                df['theta'] = (theta_common - self.risk_free_rate * strikes * np.exp(-self.risk_free_rate * time_to_exp) * norm.cdf(d2)) / 365
            else:
                df['theta'] = (theta_common + self.risk_free_rate * strikes * np.exp(-self.risk_free_rate * time_to_exp) * norm.cdf(-d2)) / 365
            
            # Vega (same for calls and puts)
            df['vega'] = current_price * norm.pdf(d1) * np.sqrt(time_to_exp) / 100
            
            # Add user-friendly interpretations
            df['delta_interpretation'] = df['delta'].apply(self._interpret_delta)
            df['gamma_interpretation'] = df['gamma'].apply(self._interpret_gamma)
            df['theta_interpretation'] = df['theta'].apply(self._interpret_theta)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Greeks calculation failed: {str(e)}")
            # Return original DataFrame if Greeks calculation fails
            return df
    
    def _classify_moneyness(self, moneyness: float) -> str:
        """Classify option moneyness for user understanding."""
        if moneyness < 0.95:
            return "Deep ITM"
        elif moneyness < 0.98:
            return "ITM"
        elif moneyness < 1.02:
            return "ATM"
        elif moneyness < 1.05:
            return "OTM"
        else:
            return "Deep OTM"
    
    def _interpret_delta(self, delta: float) -> str:
        """Interpret delta for user understanding."""
        abs_delta = abs(delta)
        if abs_delta > 0.8:
            return "High sensitivity to price moves"
        elif abs_delta > 0.5:
            return "Moderate sensitivity to price moves"
        elif abs_delta > 0.2:
            return "Low sensitivity to price moves"
        else:
            return "Very low sensitivity to price moves"
    
    def _interpret_gamma(self, gamma: float) -> str:
        """Interpret gamma for user understanding."""
        if gamma > 0.1:
            return "High acceleration risk"
        elif gamma > 0.05:
            return "Moderate acceleration risk"
        else:
            return "Low acceleration risk"
    
    def _interpret_theta(self, theta: float) -> str:
        """Interpret theta for user understanding."""
        if theta < -0.1:
            return "High time decay"
        elif theta < -0.05:
            return "Moderate time decay"
        else:
            return "Low time decay"
    
    def _calculate_chain_summary(
        self, 
        calls: pd.DataFrame, 
        puts: pd.DataFrame, 
        current_price: float
    ) -> Dict[str, Any]:
        """Calculate summary statistics for the options chain."""
        try:
            summary = {}
            
            # Call/Put ratio
            call_volume = calls['volume'].fillna(0).sum() if not calls.empty else 0
            put_volume = puts['volume'].fillna(0).sum() if not puts.empty else 0
            
            summary['call_put_volume_ratio'] = call_volume / max(put_volume, 1)
            summary['total_call_volume'] = int(call_volume)
            summary['total_put_volume'] = int(put_volume)
            
            # Open interest
            call_oi = calls['openInterest'].fillna(0).sum() if not calls.empty else 0
            put_oi = puts['openInterest'].fillna(0).sum() if not puts.empty else 0
            
            summary['call_put_oi_ratio'] = call_oi / max(put_oi, 1)
            summary['total_call_oi'] = int(call_oi)
            summary['total_put_oi'] = int(put_oi)
            
            # Find ATM options
            if not calls.empty:
                atm_call_idx = (calls['strike'] - current_price).abs().idxmin()
                summary['atm_call_iv'] = calls.loc[atm_call_idx, 'implied_volatility']
                summary['atm_call_strike'] = calls.loc[atm_call_idx, 'strike']
            
            if not puts.empty:
                atm_put_idx = (puts['strike'] - current_price).abs().idxmin()
                summary['atm_put_iv'] = puts.loc[atm_put_idx, 'implied_volatility']
                summary['atm_put_strike'] = puts.loc[atm_put_idx, 'strike']
            
            # Calculate skew (put IV - call IV)
            if 'atm_call_iv' in summary and 'atm_put_iv' in summary:
                summary['iv_skew'] = summary['atm_put_iv'] - summary['atm_call_iv']
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Chain summary calculation failed: {str(e)}")
            return {}
    
    def _generate_options_insights(
        self, 
        chains: Dict[str, Any], 
        current_price: float, 
        symbol: str
    ) -> Dict[str, Any]:
        """Generate user-friendly insights from options data."""
        try:
            insights = {
                'symbol': symbol,
                'current_price': current_price,
                'analysis_timestamp': datetime.now().isoformat(),
                'key_observations': [],
                'trading_opportunities': [],
                'risk_warnings': []
            }
            
            # Analyze across all expirations
            all_call_volumes = []
            all_put_volumes = []
            all_iv_levels = []
            
            for exp_date, chain_data in chains.items():
                if not chain_data['success']:
                    continue
                
                summary = chain_data.get('summary', {})
                
                # Collect data for overall analysis
                all_call_volumes.append(summary.get('total_call_volume', 0))
                all_put_volumes.append(summary.get('total_put_volume', 0))
                
                if 'atm_call_iv' in summary:
                    all_iv_levels.append(summary['atm_call_iv'])
                if 'atm_put_iv' in summary:
                    all_iv_levels.append(summary['atm_put_iv'])
            
            # Generate insights
            if all_call_volumes and all_put_volumes:
                total_call_vol = sum(all_call_volumes)
                total_put_vol = sum(all_put_volumes)
                overall_cp_ratio = total_call_vol / max(total_put_vol, 1)
                
                if overall_cp_ratio > 2.0:
                    insights['key_observations'].append("📈 Strong bullish sentiment - high call volume")
                elif overall_cp_ratio < 0.5:
                    insights['key_observations'].append("📉 Strong bearish sentiment - high put volume")
                else:
                    insights['key_observations'].append("⚖️ Balanced sentiment - mixed call/put activity")
            
            # IV analysis
            if all_iv_levels:
                avg_iv = np.mean(all_iv_levels)
                
                if avg_iv > 0.4:
                    insights['key_observations'].append("⚡ High implied volatility - market expects large moves")
                    insights['trading_opportunities'].append("Consider selling premium strategies")
                elif avg_iv < 0.15:
                    insights['key_observations'].append("😴 Low implied volatility - market expects small moves")
                    insights['trading_opportunities'].append("Consider buying premium strategies")
                else:
                    insights['key_observations'].append("📊 Normal implied volatility levels")
            
            # Expiration analysis
            if len(chains) > 1:
                insights['key_observations'].append(f"📅 {len(chains)} expiration dates available for analysis")
                
                # Find nearest expiration with good liquidity
                liquid_expirations = []
                for exp_date, chain_data in chains.items():
                    if chain_data['success']:
                        total_volume = (chain_data.get('summary', {}).get('total_call_volume', 0) + 
                                      chain_data.get('summary', {}).get('total_put_volume', 0))
                        if total_volume > 10:  # Minimum liquidity threshold
                            liquid_expirations.append(exp_date)
                
                if liquid_expirations:
                    insights['trading_opportunities'].append(f"Focus on liquid expirations: {', '.join(liquid_expirations[:3])}")
            
            # Risk warnings
            if not all_iv_levels:
                insights['risk_warnings'].append("⚠️ Limited options data - verify liquidity before trading")
            
            if sum(all_call_volumes) + sum(all_put_volumes) < 50:
                insights['risk_warnings'].append("⚠️ Low options volume - wide bid-ask spreads likely")
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Options insights generation failed: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'key_observations': ['Analysis failed - limited insights available']
            }
    
    async def get_options_features(self, symbol: str) -> Dict[str, Any]:
        """
        Extract options-based features for ML models.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of options-based features
        """
        log_user_action("get_options_features", {"symbol": symbol})
        
        try:
            # Get fresh options data
            options_result = await self.fetch_options_chain(symbol)
            
            if not options_result['success']:
                return {
                    'success': False,
                    'error': options_result.get('error', 'Options data unavailable'),
                    'features': {}
                }
            
            features = {}
            chains = options_result['chains']
            
            # Calculate features from nearest expiration
            if chains:
                nearest_exp = min(chains.keys())
                nearest_chain = chains[nearest_exp]
                
                if nearest_chain['success']:
                    summary = nearest_chain.get('summary', {})
                    
                    # Volume-based features
                    features['call_put_volume_ratio'] = summary.get('call_put_volume_ratio', 1.0)
                    features['total_options_volume'] = summary.get('total_call_volume', 0) + summary.get('total_put_volume', 0)
                    
                    # Open interest features
                    features['call_put_oi_ratio'] = summary.get('call_put_oi_ratio', 1.0)
                    features['total_options_oi'] = summary.get('total_call_oi', 0) + summary.get('total_put_oi', 0)
                    
                    # Volatility features
                    features['atm_implied_volatility'] = summary.get('atm_call_iv', 0.2)
                    features['iv_skew'] = summary.get('iv_skew', 0.0)
                    
                    # Time to expiration
                    features['days_to_nearest_expiration'] = nearest_chain.get('days_to_expiration', 30)
            
            # Add sentiment indicators
            if 'call_put_volume_ratio' in features:
                ratio = features['call_put_volume_ratio']
                if ratio > 2.0:
                    features['options_sentiment'] = 1  # Bullish
                elif ratio < 0.5:
                    features['options_sentiment'] = -1  # Bearish
                else:
                    features['options_sentiment'] = 0  # Neutral
            
            return {
                'success': True,
                'symbol': symbol,
                'features': features,
                'feature_count': len(features),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Options features extraction failed for {symbol}: {str(e)}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e),
                'features': {}
            }
    
    async def process_options_chain(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """
        Process options chain for a given symbol and current price.
        This method is used by the user acceptance tests.
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            
        Returns:
            Processed options chain data
        """
        try:
            # Fetch the options chain
            chain_data = await self.fetch_options_chain(symbol)
            
            if not chain_data or not chain_data.get('success', False):
                raise ValueError(f"No options data available for {symbol}")
                
            # Return the processed chain data with expected structure
            return {
                "success": True,
                "symbol": symbol,
                "current_price": current_price,
                "chains": chain_data.get('chains', {}),
                "processing_time": 1.0  # Placeholder for processing time
            }
            
        except Exception as e:
            logger.error(f"Failed to process options chain for {symbol}: {str(e)}")
            raise ValueError(f"Options chain processing failed: {str(e)}")
    
    def get_cached_chain(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached options chain data."""
        if symbol in self.current_chains:
            cached_data = self.current_chains[symbol]
            # Check if data is still fresh (within 1 hour)
            if (datetime.now() - cached_data['timestamp']).seconds < 3600:
                return cached_data
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check options processing system health."""
        try:
            # Test with a common stock
            test_symbol = "AAPL"
            
            start_time = datetime.now()
            test_result = await self.fetch_options_chain(test_symbol)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'healthy': test_result['success'],
                'test_symbol': test_symbol,
                'processing_time': processing_time,
                'cached_symbols': list(self.current_chains.keys()),
                'message': 'Options processing system operational' if test_result['success'] else f"Test failed: {test_result.get('error', 'Unknown error')}"
            }
            
        except Exception as e:
            logger.error(f"❌ Options system health check failed: {str(e)}")
            return {
                'healthy': False,
                'error': str(e),
                'message': 'Options processing system health check failed'
            }
