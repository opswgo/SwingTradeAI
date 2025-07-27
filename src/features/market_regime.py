"""
SwingTradeAI - Market Regime Detection
User Acceptance: User can see and understand market regime explanations
Includes: Trend indicators (200-day SMA), VIX volatility, breadth indicators
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

from ..core.logger import setup_logger, log_user_action, log_data_quality

logger = setup_logger(__name__)


class MarketRegime(Enum):
    """Market regime classifications with user-friendly descriptions."""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    BULL_VOLATILE = "bull_volatile"
    BEAR_VOLATILE = "bear_volatile"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"


class MarketRegimeDetector:
    """
    Market regime detection with user-focused explanations.
    User Focus: Clear regime identification and trading implications
    """
    
    def __init__(self):
        self.regime_descriptions = self._get_regime_descriptions()
        self.indicator_thresholds = self._get_indicator_thresholds()
        self.current_regime = None
        self.regime_confidence = 0.0
        
    def _get_regime_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Get user-friendly regime descriptions and trading implications."""
        return {
            MarketRegime.BULL_TRENDING.value: {
                'name': 'Bull Trending Market',
                'description': 'Strong upward trend with low volatility',
                'characteristics': 'Price above 200-day SMA, low VIX, strong breadth',
                'trading_implications': 'Favor long positions, momentum strategies',
                'risk_level': 'Low to Medium',
                'emoji': '📈'
            },
            MarketRegime.BEAR_TRENDING.value: {
                'name': 'Bear Trending Market',
                'description': 'Strong downward trend with controlled volatility',
                'characteristics': 'Price below 200-day SMA, elevated VIX, weak breadth',
                'trading_implications': 'Favor short positions, defensive strategies',
                'risk_level': 'Medium to High',
                'emoji': '📉'
            },
            MarketRegime.BULL_VOLATILE.value: {
                'name': 'Bull Volatile Market',
                'description': 'Upward bias with high volatility swings',
                'characteristics': 'Price above 200-day SMA, high VIX, mixed breadth',
                'trading_implications': 'Swing trading, volatility strategies',
                'risk_level': 'Medium to High',
                'emoji': '🎢'
            },
            MarketRegime.BEAR_VOLATILE.value: {
                'name': 'Bear Volatile Market',
                'description': 'Downward bias with extreme volatility',
                'characteristics': 'Price below 200-day SMA, very high VIX, poor breadth',
                'trading_implications': 'Defensive positions, volatility trading',
                'risk_level': 'High',
                'emoji': '⚡'
            },
            MarketRegime.SIDEWAYS_LOW_VOL.value: {
                'name': 'Sideways Low Volatility',
                'description': 'Range-bound market with low volatility',
                'characteristics': 'Price near 200-day SMA, low VIX, neutral breadth',
                'trading_implications': 'Range trading, mean reversion',
                'risk_level': 'Low',
                'emoji': '➡️'
            },
            MarketRegime.SIDEWAYS_HIGH_VOL.value: {
                'name': 'Sideways High Volatility',
                'description': 'Range-bound market with high volatility',
                'characteristics': 'Price near 200-day SMA, high VIX, mixed breadth',
                'trading_implications': 'Volatility strategies, short-term trades',
                'risk_level': 'Medium',
                'emoji': '🌊'
            },
            MarketRegime.CRISIS.value: {
                'name': 'Crisis Market',
                'description': 'Extreme market stress and panic selling',
                'characteristics': 'Sharp decline, very high VIX (>40), terrible breadth',
                'trading_implications': 'Capital preservation, wait for stabilization',
                'risk_level': 'Very High',
                'emoji': '🚨'
            },
            MarketRegime.RECOVERY.value: {
                'name': 'Recovery Market',
                'description': 'Emerging from crisis with improving conditions',
                'characteristics': 'Price recovering, declining VIX, improving breadth',
                'trading_implications': 'Gradual re-entry, quality stocks first',
                'risk_level': 'Medium',
                'emoji': '🌱'
            }
        }
    
    def _get_indicator_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Define thresholds for regime classification."""
        return {
            'vix': {
                'low': 15.0,
                'medium': 25.0,
                'high': 35.0,
                'crisis': 40.0
            },
            'trend': {
                'strong_bull': 0.05,  # 5% above 200-day SMA
                'weak_bull': 0.02,   # 2% above 200-day SMA
                'neutral': 0.02,     # Within 2% of 200-day SMA
                'weak_bear': -0.02,  # 2% below 200-day SMA
                'strong_bear': -0.05 # 5% below 200-day SMA
            },
            'breadth': {
                'strong': 0.7,       # 70% of stocks above their 50-day MA
                'moderate': 0.5,     # 50% of stocks above their 50-day MA
                'weak': 0.3          # 30% of stocks above their 50-day MA
            }
        }
    
    async def detect_current_regime(
        self, 
        market_data: pd.DataFrame, 
        vix_data: Optional[pd.DataFrame] = None,
        breadth_data: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Detect current market regime with user-friendly analysis.
        
        Args:
            market_data: OHLCV data for market index (e.g., SPY)
            vix_data: VIX volatility data (optional, will fetch if not provided)
            breadth_data: Market breadth indicators (optional)
            
        Returns:
            Dictionary with regime analysis and user explanations
        """
        log_user_action("detect_market_regime", {"data_points": len(market_data)})
        
        try:
            # Calculate trend indicators
            trend_analysis = await self._analyze_trend(market_data)
            
            # Get volatility analysis
            volatility_analysis = await self._analyze_volatility(vix_data)
            
            # Analyze market breadth
            breadth_analysis = await self._analyze_breadth(breadth_data)
            
            # Determine regime based on combined analysis
            regime_result = self._classify_regime(trend_analysis, volatility_analysis, breadth_analysis)
            
            # Generate user-friendly report
            user_report = self._generate_user_report(regime_result, trend_analysis, volatility_analysis, breadth_analysis)
            
            # Store current regime
            self.current_regime = regime_result['regime']
            self.regime_confidence = regime_result['confidence']
            
            logger.info(f"✅ Market regime detected: {regime_result['regime'].value} (confidence: {regime_result['confidence']:.1%})")
            
            return user_report
            
        except Exception as e:
            logger.error(f"❌ Market regime detection failed: {str(e)}")
            return {
                'error': str(e),
                'regime': MarketRegime.SIDEWAYS_LOW_VOL,  # Default fallback
                'confidence': 0.0,
                'user_message': f"Unable to detect market regime: {str(e)}"
            }
    
    async def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend using 200-day SMA and price action."""
        if len(data) < 200:
            logger.warning("⚠️ Insufficient data for 200-day SMA, using available data")
            sma_period = min(len(data) - 1, 50)  # Use 50-day if not enough data
        else:
            sma_period = 200
        
        # Calculate 200-day (or available) SMA
        data['sma_200'] = data['Close'].rolling(window=sma_period).mean()
        
        latest_price = data['Close'].iloc[-1]
        latest_sma = data['sma_200'].iloc[-1]
        
        # Calculate trend strength
        trend_ratio = (latest_price - latest_sma) / latest_sma
        
        # Determine trend classification
        thresholds = self.indicator_thresholds['trend']
        if trend_ratio > thresholds['strong_bull']:
            trend_strength = 'strong_bull'
        elif trend_ratio > thresholds['weak_bull']:
            trend_strength = 'weak_bull'
        elif trend_ratio > thresholds['neutral']:
            trend_strength = 'neutral_bull'
        elif trend_ratio > thresholds['weak_bear']:
            trend_strength = 'neutral_bear'
        elif trend_ratio > thresholds['strong_bear']:
            trend_strength = 'weak_bear'
        else:
            trend_strength = 'strong_bear'
        
        # Calculate recent momentum
        recent_return = data['Close'].pct_change(20).iloc[-1] * 100  # 20-day return
        
        return {
            'trend_ratio': trend_ratio,
            'trend_strength': trend_strength,
            'sma_period': sma_period,
            'current_price': latest_price,
            'sma_value': latest_sma,
            'recent_momentum': recent_return,
            'user_message': f"Price is {abs(trend_ratio)*100:.1f}% {'above' if trend_ratio > 0 else 'below'} {sma_period}-day average"
        }
    
    async def _analyze_volatility(self, vix_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Analyze market volatility using VIX or calculated volatility."""
        try:
            if vix_data is not None and not vix_data.empty:
                current_vix = vix_data['Close'].iloc[-1]
                vix_source = 'provided'
            else:
                # Fetch VIX data
                logger.info("📊 Fetching VIX data for volatility analysis...")
                try:
                    vix_ticker = yf.Ticker("^VIX")
                    vix_hist = vix_ticker.history(period="1mo")
                    if not vix_hist.empty:
                        current_vix = vix_hist['Close'].iloc[-1]
                        vix_source = 'yfinance'
                    else:
                        # No fallback - fail if we don't have real VIX data
                        raise ValueError("No real VIX data available - cannot make reliable volatility assessment")
                except Exception as e:
                    # No fallback - fail if we don't have real VIX data
                    raise ValueError(f"VIX fetch failed: {str(e)} - cannot make reliable volatility assessment")
            
            # Classify volatility level
            thresholds = self.indicator_thresholds['vix']
            if current_vix > thresholds['crisis']:
                vol_level = 'crisis'
            elif current_vix > thresholds['high']:
                vol_level = 'high'
            elif current_vix > thresholds['medium']:
                vol_level = 'medium'
            elif current_vix > thresholds['low']:
                vol_level = 'low'
            else:
                vol_level = 'very_low'
            
            return {
                'vix_value': current_vix,
                'volatility_level': vol_level,
                'vix_source': vix_source,
                'user_message': f"Market volatility (VIX): {current_vix:.1f} - {vol_level.replace('_', ' ').title()}"
            }
            
        except Exception as e:
            logger.error(f"❌ Volatility analysis failed: {str(e)}")
            # Don't return fake data - re-raise the error
            raise ValueError(f"Cannot perform volatility analysis without real VIX data: {str(e)}")
    
    async def _analyze_breadth(self, breadth_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Analyze market breadth indicators."""
        try:
            if breadth_data:
                # Use provided breadth data
                breadth_ratio = breadth_data.get('stocks_above_50ma', 0.5)
                breadth_source = 'provided'
            else:
                # No estimation - fail if we don't have real breadth data
                raise ValueError("No real market breadth data available - cannot make reliable breadth assessment")
            
            # Classify breadth strength
            thresholds = self.indicator_thresholds['breadth']
            if breadth_ratio > thresholds['strong']:
                breadth_strength = 'strong'
            elif breadth_ratio > thresholds['moderate']:
                breadth_strength = 'moderate'
            elif breadth_ratio > thresholds['weak']:
                breadth_strength = 'weak'
            else:
                breadth_strength = 'very_weak'
            
            return {
                'breadth_ratio': breadth_ratio,
                'breadth_strength': breadth_strength,
                'breadth_source': breadth_source,
                'user_message': f"Market breadth: {breadth_ratio:.1%} of stocks above 50-day MA - {breadth_strength.replace('_', ' ').title()}"
            }
            
        except Exception as e:
            logger.error(f"❌ Breadth analysis failed: {str(e)}")
            return {
                'breadth_ratio': 0.5,
                'breadth_strength': 'moderate',
                'breadth_source': 'error_fallback',
                'user_message': f"Breadth analysis error: {str(e)}"
            }
    
    def _classify_regime(
        self, 
        trend_analysis: Dict[str, Any], 
        volatility_analysis: Dict[str, Any], 
{{ ... }}
        breadth_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify market regime based on combined analysis."""
        
        trend_strength = trend_analysis['trend_strength']
        vol_level = volatility_analysis['volatility_level']
        breadth_strength = breadth_analysis['breadth_strength']
        
        # Regime classification logic
        regime = MarketRegime.SIDEWAYS_LOW_VOL  # Default
        confidence = 0.5
        
        # Crisis detection (highest priority)
        if vol_level == 'crisis':
            regime = MarketRegime.CRISIS
            confidence = 0.9
        
        # Recovery detection
        elif (vol_level in ['high', 'medium'] and 
              trend_strength in ['weak_bull', 'neutral_bull'] and 
              breadth_strength in ['moderate', 'strong']):
            regime = MarketRegime.RECOVERY
            confidence = 0.7
        
        # Bull markets
        elif trend_strength in ['strong_bull', 'weak_bull']:
            if vol_level in ['very_low', 'low']:
                regime = MarketRegime.BULL_TRENDING
                confidence = 0.8
            else:
                regime = MarketRegime.BULL_VOLATILE
                confidence = 0.7
        
        # Bear markets
        elif trend_strength in ['strong_bear', 'weak_bear']:
            if vol_level in ['high', 'crisis']:
                regime = MarketRegime.BEAR_VOLATILE
                confidence = 0.8
            else:
                regime = MarketRegime.BEAR_TRENDING
                confidence = 0.7
        
        # Sideways markets
        else:
            if vol_level in ['very_low', 'low']:
                regime = MarketRegime.SIDEWAYS_LOW_VOL
                confidence = 0.6
            else:
                regime = MarketRegime.SIDEWAYS_HIGH_VOL
                confidence = 0.6
        
        # Adjust confidence based on breadth confirmation
        if breadth_strength == 'strong' and regime in [MarketRegime.BULL_TRENDING, MarketRegime.BULL_VOLATILE]:
            confidence = min(0.95, confidence + 0.1)
        elif breadth_strength == 'very_weak' and regime in [MarketRegime.BEAR_TRENDING, MarketRegime.BEAR_VOLATILE]:
            confidence = min(0.95, confidence + 0.1)
        
        return {
            'regime': regime,
            'confidence': confidence,
            'contributing_factors': {
                'trend': trend_strength,
                'volatility': vol_level,
                'breadth': breadth_strength
            }
        }
    
    def _generate_user_report(
        self, 
        regime_result: Dict[str, Any], 
        trend_analysis: Dict[str, Any], 
        volatility_analysis: Dict[str, Any], 
        breadth_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive user-friendly report."""
        
        regime = regime_result['regime']
        regime_info = self.regime_descriptions[regime.value]
        
        return {
            'regime': regime,
            'regime_name': regime_info['name'],
            'confidence': regime_result['confidence'],
            'emoji': regime_info['emoji'],
            
            # User-friendly summary
            'summary': {
                'current_regime': f"{regime_info['emoji']} {regime_info['name']}",
                'confidence_level': f"{regime_result['confidence']:.1%}",
                'description': regime_info['description'],
                'risk_level': regime_info['risk_level']
            },
            
            # Detailed analysis
            'analysis': {
                'trend': trend_analysis,
                'volatility': volatility_analysis,
                'breadth': breadth_analysis
            },
            
            # Trading implications
            'trading_implications': {
                'strategy_recommendation': regime_info['trading_implications'],
                'risk_level': regime_info['risk_level'],
                'key_characteristics': regime_info['characteristics']
            },
            
            # User messages
            'user_messages': [
                f"{regime_info['emoji']} Current Market Regime: {regime_info['name']}",
                f"🎯 Confidence Level: {regime_result['confidence']:.1%}",
                f"📊 {trend_analysis['user_message']}",
                f"📈 {volatility_analysis['user_message']}",
                f"🏢 {breadth_analysis['user_message']}",
                f"💡 Strategy: {regime_info['trading_implications']}"
            ],
            
            # Timestamp
            'timestamp': datetime.now().isoformat(),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_regime_history(self, data: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
        """
        Calculate historical regime classifications.
        
        Args:
            data: Historical market data
            window_size: Rolling window for regime detection
            
        Returns:
            DataFrame with regime classifications over time
        """
        logger.info(f"📊 Calculating regime history with {window_size}-day windows")
        
        regime_history = []
        
        # Calculate for each window
        for i in range(window_size, len(data)):
            window_data = data.iloc[i-window_size:i+1]
            
            try:
                # Simplified regime detection for historical analysis
                sma_200 = window_data['Close'].rolling(window=min(200, len(window_data))).mean().iloc[-1]
                current_price = window_data['Close'].iloc[-1]
                
                # Simple trend classification
                trend_ratio = (current_price - sma_200) / sma_200
                
                # Estimate volatility from price data
                returns = window_data['Close'].pct_change().dropna()
                estimated_vol = returns.std() * np.sqrt(252) * 100  # Annualized volatility
                
                # Simple regime classification
                if estimated_vol > 35:
                    if trend_ratio > 0.02:
                        regime = MarketRegime.BULL_VOLATILE
                    elif trend_ratio < -0.02:
                        regime = MarketRegime.BEAR_VOLATILE
                    else:
                        regime = MarketRegime.SIDEWAYS_HIGH_VOL
                elif estimated_vol > 25:
                    if trend_ratio > 0.05:
                        regime = MarketRegime.BULL_TRENDING
                    elif trend_ratio < -0.05:
                        regime = MarketRegime.BEAR_TRENDING
                    else:
                        regime = MarketRegime.SIDEWAYS_HIGH_VOL
                else:
                    if trend_ratio > 0.05:
                        regime = MarketRegime.BULL_TRENDING
                    elif trend_ratio < -0.05:
                        regime = MarketRegime.BEAR_TRENDING
                    else:
                        regime = MarketRegime.SIDEWAYS_LOW_VOL
                
                regime_history.append({
                    'date': window_data.index[-1],
                    'regime': regime.value,
                    'regime_name': self.regime_descriptions[regime.value]['name'],
                    'trend_ratio': trend_ratio,
                    'estimated_volatility': estimated_vol,
                    'price': current_price,
                    'sma_200': sma_200
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Regime calculation failed for window ending {window_data.index[-1]}: {str(e)}")
                continue
        
        return pd.DataFrame(regime_history).set_index('date')
    
    def get_regime_transitions(self, regime_history: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify regime transitions for user analysis.
        
        Args:
            regime_history: DataFrame from get_regime_history()
            
        Returns:
            List of regime transitions with user explanations
        """
        transitions = []
        
        if len(regime_history) < 2:
            return transitions
        
        current_regime = regime_history['regime'].iloc[0]
        transition_start = regime_history.index[0]
        
        for i in range(1, len(regime_history)):
            new_regime = regime_history['regime'].iloc[i]
            
            if new_regime != current_regime:
                # Regime transition detected
                transition_end = regime_history.index[i]
                duration = (transition_end - transition_start).days
                
                transitions.append({
                    'from_regime': current_regime,
                    'to_regime': new_regime,
                    'from_name': self.regime_descriptions[current_regime]['name'],
                    'to_name': self.regime_descriptions[new_regime]['name'],
                    'transition_date': transition_end,
                    'duration_days': duration,
                    'user_message': f"Regime changed from {self.regime_descriptions[current_regime]['name']} to {self.regime_descriptions[new_regime]['name']} (lasted {duration} days)"
                })
                
                current_regime = new_regime
                transition_start = transition_end
        
        return transitions
