#!/usr/bin/env python3
"""
COMPREHENSIVE FEATURE ENGINEERING - Sprint 1
50+ features including options-specific indicators for profitable trading.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveFeatureEngine:
    """Comprehensive feature engineering for options trading profitability."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.feature_cache = {}
        
    def _setup_logging(self):
        """Setup comprehensive logging."""
        logger = logging.getLogger('FeatureEngine')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def engineer_all_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Engineer all 50+ comprehensive features for options trading.
        
        Args:
            data: OHLCV DataFrame
            symbol: Stock symbol for logging
            
        Returns:
            DataFrame with 50+ engineered features
        """
        self.logger.info(f"🔧 Engineering comprehensive features for {symbol}")
        
        df = data.copy()
        features = pd.DataFrame(index=df.index)
        
        try:
            # 1. PRICE MOMENTUM FEATURES (12 features)
            momentum_features = self._price_momentum_features(df)
            for col, values in momentum_features.items():
                features[col] = values
            
            # 2. VOLATILITY FEATURES (10 features)
            vol_features = self._volatility_features(df)
            for col, values in vol_features.items():
                features[col] = values
            
            # 3. VOLUME FEATURES (8 features)
            volume_features = self._volume_features(df)
            for col, values in volume_features.items():
                features[col] = values
            
            # 4. TECHNICAL INDICATORS (15 features)
            tech_features = self._technical_indicators(df)
            for col, values in tech_features.items():
                features[col] = values
            
            # 5. OPTIONS-SPECIFIC FEATURES (10 features)
            options_features = self._options_specific_features(df)
            for col, values in options_features.items():
                features[col] = values
            
            # 6. REGIME DETECTION FEATURES (8 features)
            regime_features = self._regime_detection_features(df, features)
            for col, values in regime_features.items():
                features[col] = values
            
            # 7. RISK FEATURES (5 features)
            risk_features = self._risk_features(df)
            for col, values in risk_features.items():
                features[col] = values
            
            # Clean and normalize all features
            features = self._clean_and_normalize_features(features)
            
            self.logger.info(f"✅ Engineered {len(features.columns)} comprehensive features for {symbol}")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed for {symbol}: {e}")
            raise
    
    def _price_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price momentum features (12 features)."""
        features = pd.DataFrame(index=df.index)
        
        # Returns over multiple periods
        for period in [1, 5, 10, 20, 50]:
            features[f'returns_{period}d'] = df['Close'].pct_change(period)
        
        # Price relative to moving averages
        for period in [10, 20, 50, 200]:
            sma = df['Close'].rolling(period).mean()
            features[f'price_vs_sma{period}'] = (df['Close'] / sma) - 1
        
        # Momentum ratios
        features['momentum_5_20'] = (df['Close'].rolling(5).mean() / df['Close'].rolling(20).mean()) - 1
        features['momentum_20_50'] = (df['Close'].rolling(20).mean() / df['Close'].rolling(50).mean()) - 1
        features['momentum_50_200'] = (df['Close'].rolling(50).mean() / df['Close'].rolling(200).mean()) - 1
        
        return features
    
    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility features (10 features)."""
        features = pd.DataFrame(index=df.index)
        
        # Realized volatility over multiple periods
        for period in [5, 10, 20, 50]:
            features[f'volatility_{period}d'] = df['Close'].pct_change().rolling(period).std() * np.sqrt(252)
        
        # Volatility ratios
        features['vol_ratio_5_20'] = features['volatility_5d'] / features['volatility_20d']
        features['vol_ratio_10_50'] = features['volatility_10d'] / features['volatility_50d']
        
        # Volatility trends
        features['vol_trend_short'] = features['volatility_5d'] - features['volatility_20d']
        features['vol_trend_long'] = features['volatility_20d'] - features['volatility_50d']
        
        # Volatility percentiles (regime detection)
        features['vol_percentile_252d'] = df['Close'].pct_change().rolling(252).rank(pct=True)
        features['vol_spike'] = (features['volatility_5d'] > features['volatility_20d'] * 1.5).astype(float)
        
        return features
    
    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume features (8 features)."""
        features = pd.DataFrame(index=df.index)
        
        # Volume ratios
        for period in [5, 20, 50]:
            features[f'volume_ratio_{period}d'] = df['Volume'] / df['Volume'].rolling(period).mean()
        
        # Volume trends
        features['volume_trend'] = (df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()) - 1
        features['volume_volatility'] = (df['Volume'] / df['Volume'].rolling(20).mean()).rolling(20).std()
        
        # Price-volume correlation
        features['price_volume_corr_20d'] = df['Close'].pct_change().rolling(20).corr(df['Volume'].pct_change())
        
        # Volume spikes
        features['volume_spike'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 2).astype(float)
        features['volume_drought'] = (df['Volume'] < df['Volume'].rolling(20).mean() * 0.5).astype(float)
        
        return features
    
    def _technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Technical indicators (15 features)."""
        features = pd.DataFrame(index=df.index)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_oversold'] = (features['rsi'] < 30).astype(float)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(float)
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = df['Close'].rolling(bb_period).mean()
        bb_upper = bb_middle + (df['Close'].rolling(bb_period).std() * bb_std)
        bb_lower = bb_middle - (df['Close'].rolling(bb_period).std() * bb_std)
        features['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(20).quantile(0.1)).astype(float)
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        features['macd_bullish'] = (features['macd'] > features['macd_signal']).astype(float)
        
        # Support/Resistance
        features['high_20d'] = df['High'].rolling(20).max()
        features['low_20d'] = df['Low'].rolling(20).min()
        features['price_position'] = (df['Close'] - features['low_20d']) / (features['high_20d'] - features['low_20d'])
        
        # Stochastic
        features['stoch_k'] = ((df['Close'] - features['low_20d']) / (features['high_20d'] - features['low_20d'])) * 100
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        return features
    
    def _options_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Options-specific features (10 features)."""
        features = pd.DataFrame(index=df.index)
        
        # Implied volatility proxies
        features['realized_vol_5d'] = df['Close'].pct_change().rolling(5).std() * np.sqrt(252)
        features['realized_vol_20d'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        features['vol_term_structure'] = features['realized_vol_5d'] / features['realized_vol_20d']
        
        # Options expiration timing
        features['days_to_friday'] = np.clip(4 - df.index.dayofweek, 0, 4)  # Days until Friday
        features['monthly_opex'] = ((df.index.day >= 15) & (df.index.day <= 21) & (df.index.dayofweek == 4)).astype(float)
        features['weekly_opex'] = (df.index.dayofweek == 4).astype(float)  # Every Friday
        
        # Put/call sentiment proxies
        features['put_call_ratio_proxy'] = features['realized_vol_20d'] / 0.20  # Normalized vol as fear proxy
        
        # Options strategy favorability (need to calculate after other features)
        features['theta_favorable'] = 0.5  # Placeholder, will be updated later
        features['assignment_risk'] = (df['Close'].pct_change(20) < -0.05).astype(float)  # Recent decline
        
        return features
    
    def _regime_detection_features(self, df: pd.DataFrame, existing_features: pd.DataFrame = None) -> pd.DataFrame:
        """Market regime detection features (8 features)."""
        features = pd.DataFrame(index=df.index)
        
        # Trend strength and direction
        price_vs_sma20 = (df['Close'] / df['Close'].rolling(20).mean()) - 1
        price_vs_sma50 = (df['Close'] / df['Close'].rolling(50).mean()) - 1
        
        features['trend_strength'] = abs(price_vs_sma20)
        features['trend_direction'] = np.sign(price_vs_sma20)
        
        # Market regime classification
        features['bull_market'] = (price_vs_sma50 > 0.05).astype(float)
        features['bear_market'] = (price_vs_sma50 < -0.05).astype(float)
        features['sideways_market'] = ((price_vs_sma50 >= -0.05) & (price_vs_sma50 <= 0.05)).astype(float)
        
        # Volatility regime
        vol_20d = df['Close'].pct_change().rolling(20).std()
        features['high_vol_regime'] = (vol_20d > vol_20d.rolling(252).quantile(0.8)).astype(float)
        features['low_vol_regime'] = (vol_20d < vol_20d.rolling(252).quantile(0.2)).astype(float)
        
        # Momentum regime
        features['momentum_regime'] = np.sign(df['Close'].pct_change(20))
        
        return features
    
    def _risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Risk assessment features (5 features)."""
        features = pd.DataFrame(index=df.index)
        
        # Drawdown features
        rolling_max = df['Close'].rolling(252).max()
        features['drawdown'] = (df['Close'] / rolling_max) - 1
        features['max_drawdown_252d'] = features['drawdown'].rolling(252).min()
        
        # Risk-adjusted returns
        returns = df['Close'].pct_change()
        features['sharpe_proxy_20d'] = (returns.rolling(20).mean() / returns.rolling(20).std()) * np.sqrt(252)
        
        # Tail risk
        features['var_5pct_20d'] = returns.rolling(20).quantile(0.05)  # 5% VaR
        features['skewness_20d'] = returns.rolling(20).skew()
        
        return features
    
    def _clean_and_normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize all features."""
        # Forward fill then zero fill
        features = features.fillna(method='ffill').fillna(0)
        
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        # Robust normalization (handle outliers)
        for col in features.columns:
            if features[col].std() > 0:
                # Use robust scaling (median and IQR)
                median = features[col].median()
                q75 = features[col].quantile(0.75)
                q25 = features[col].quantile(0.25)
                iqr = q75 - q25
                
                if iqr > 0:
                    features[col] = (features[col] - median) / iqr
                    # Clip extreme outliers
                    features[col] = np.clip(features[col], -3, 3)
                else:
                    features[col] = features[col] - median
        
        return features
    
    def get_feature_importance_report(self, features: pd.DataFrame) -> Dict:
        """Generate feature importance and quality report."""
        return {
            'total_features': len(features.columns),
            'feature_categories': {
                'price_momentum': len([c for c in features.columns if 'returns' in c or 'momentum' in c or 'sma' in c]),
                'volatility': len([c for c in features.columns if 'vol' in c]),
                'volume': len([c for c in features.columns if 'volume' in c]),
                'technical': len([c for c in features.columns if any(x in c for x in ['rsi', 'bb', 'macd', 'stoch'])]),
                'options_specific': len([c for c in features.columns if any(x in c for x in ['opex', 'theta', 'assignment'])]),
                'regime': len([c for c in features.columns if 'regime' in c or 'market' in c]),
                'risk': len([c for c in features.columns if any(x in c for x in ['drawdown', 'sharpe', 'var', 'skew'])])
            },
            'data_quality': {
                'null_values': features.isnull().sum().sum(),
                'infinite_values': np.isinf(features.values).sum(),
                'feature_variance': features.var().mean(),
                'correlation_matrix_rank': np.linalg.matrix_rank(features.corr().fillna(0))
            }
        }


if __name__ == "__main__":
    # Test comprehensive feature engineering
    from bulletproof_data_fetcher import BulletproofDataFetcher
    
    print("🔥 TESTING COMPREHENSIVE FEATURE ENGINEERING")
    print("=" * 60)
    
    # Initialize systems
    data_fetcher = BulletproofDataFetcher()
    feature_engine = ComprehensiveFeatureEngine()
    
    # Test with all required tickers
    test_tickers = ['AAPL', 'AVGO', 'HOOD', 'TSLA', 'TSLL']
    
    for ticker in test_tickers:
        try:
            # Fetch data
            data = data_fetcher.fetch_data(ticker, "2y")
            
            # Engineer features
            features = feature_engine.engineer_all_features(data, ticker)
            
            # Generate report
            report = feature_engine.get_feature_importance_report(features)
            
            print(f"\n✅ {ticker}:")
            print(f"   Total Features: {report['total_features']}")
            print(f"   Categories: {report['feature_categories']}")
            print(f"   Data Quality: {report['data_quality']['null_values']} nulls, {report['data_quality']['infinite_values']} infs")
            print(f"   Feature Variance: {report['data_quality']['feature_variance']:.4f}")
            
        except Exception as e:
            print(f"\n❌ {ticker}: {e}")
    
    print(f"\n🏆 COMPREHENSIVE FEATURE ENGINEERING TEST COMPLETE")
