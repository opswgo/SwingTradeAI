"""
Enhanced Data Pipeline - NEAT Options System
Extends data collection to 5-7 years with quality validation and regime detection.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import sqlite3
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataPipeline:
    """Enhanced data pipeline for per-stock evolutionary agents."""
    
    def __init__(self, db_path: str = "data/market_data.db"):
        """Initialize enhanced data pipeline."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._init_data_database()
    
    def _init_data_database(self):
        """Initialize market data database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    data_quality_score REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_indicators (
                    date DATE PRIMARY KEY,
                    vix REAL,
                    spy_close REAL,
                    qqq_close REAL,
                    dxy_close REAL,
                    tnx_close REAL,
                    market_regime TEXT,
                    regime_confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    date DATE,
                    issue_type TEXT,
                    issue_description TEXT,
                    severity TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_date ON stock_data(symbol, date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_market_indicators_date ON market_indicators(date)")
            
            conn.commit()
    
    def collect_extended_data(self, symbol: str, years: int = 7) -> pd.DataFrame:
        """Collect 5-7 years of data for a symbol with quality validation."""
        
        end_date = datetime.now().date()
        start_date = (datetime.now() - timedelta(days=years * 365)).date()
        
        self.logger.info(f"Collecting {years} years of data for {symbol}")
        
        try:
            # Download data from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if hist.empty:
                self.logger.error(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Data quality validation
            quality_score = self._validate_data_quality(hist, symbol)
            
            # Add quality score to dataframe
            hist['data_quality_score'] = quality_score
            hist['symbol'] = symbol
            
            # Store in database
            self._store_stock_data(symbol, hist)
            
            self.logger.info(f"Collected {len(hist)} days of data for {symbol} (Quality: {quality_score:.3f})")
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _validate_data_quality(self, data: pd.DataFrame, symbol: str) -> float:
        """Validate data quality and return quality score (0-1)."""
        
        quality_issues = []
        quality_score = 1.0
        
        # Check for missing data
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > 0.01:  # More than 1% missing
            quality_issues.append(f"Missing data: {missing_pct:.2%}")
            quality_score -= missing_pct * 0.5
        
        # Check for zero volume days
        zero_volume_pct = (data['Volume'] == 0).sum() / len(data)
        if zero_volume_pct > 0.05:  # More than 5% zero volume
            quality_issues.append(f"Zero volume days: {zero_volume_pct:.2%}")
            quality_score -= zero_volume_pct * 0.3
        
        # Check for price anomalies (gaps > 20%)
        price_changes = data['Close'].pct_change().abs()
        large_gaps = (price_changes > 0.20).sum()
        if large_gaps > len(data) * 0.02:  # More than 2% of days with >20% moves
            quality_issues.append(f"Large price gaps: {large_gaps} days")
            quality_score -= 0.1
        
        # Check for duplicate dates
        if data.index.duplicated().any():
            quality_issues.append("Duplicate dates found")
            quality_score -= 0.2
        
        # Check data recency
        last_date = data.index.max()
        if hasattr(last_date, 'tz_localize'):
            last_date = last_date.tz_localize(None) if last_date.tz is not None else last_date
        days_old = (datetime.now() - last_date.to_pydatetime()).days
        if days_old > 7:  # Data more than a week old
            quality_issues.append(f"Data {days_old} days old")
            quality_score -= min(days_old / 30, 0.3)  # Cap penalty at 0.3
        
        # Log quality issues
        for issue in quality_issues:
            self._log_data_quality_issue(symbol, datetime.now().date(), 
                                       "QUALITY_CHECK", issue, "MEDIUM")
        
        return max(quality_score, 0.0)  # Ensure non-negative
    
    def _store_stock_data(self, symbol: str, data: pd.DataFrame):
        """Store stock data in database."""
        
        with sqlite3.connect(self.db_path) as conn:
            for date, row in data.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO stock_data (
                        symbol, date, open, high, low, close, volume, adj_close, data_quality_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, date.date(), row['Open'], row['High'], row['Low'],
                    row['Close'], row['Volume'], row['Close'], row['data_quality_score']
                ))
            
            conn.commit()
    
    def _log_data_quality_issue(self, symbol: str, date, issue_type: str, 
                               description: str, severity: str):
        """Log data quality issue."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO data_quality_log (symbol, date, issue_type, issue_description, severity)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol, date, issue_type, description, severity))
            conn.commit()
    
    def collect_market_indicators(self) -> pd.DataFrame:
        """Collect market-wide indicators for regime detection."""
        
        indicators = ['VIX', 'SPY', 'QQQ', 'DXY', '^TNX']
        market_data = {}
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7 * 365)  # 7 years
        
        for indicator in indicators:
            try:
                ticker = yf.Ticker(indicator)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    market_data[indicator] = hist['Close']
                    self.logger.info(f"Collected {len(hist)} days of {indicator} data")
                
            except Exception as e:
                self.logger.error(f"Error collecting {indicator}: {str(e)}")
        
        if market_data:
            # Combine into single dataframe
            df = pd.DataFrame(market_data)
            df = df.dropna()
            
            # Add regime detection
            df['market_regime'] = self._detect_market_regime(df)
            df['regime_confidence'] = self._calculate_regime_confidence(df)
            
            # Store in database
            self._store_market_indicators(df)
            
            return df
        
        return pd.DataFrame()
    
    def _detect_market_regime(self, market_data: pd.DataFrame) -> pd.Series:
        """Detect market regime based on indicators."""
        
        regimes = []
        
        for date, row in market_data.iterrows():
            vix = row.get('VIX', 20)
            spy = row.get('SPY', 400)
            
            # Simple regime detection logic
            if vix > 30:
                regime = 'HIGH_VOL'
            elif vix < 15:
                regime = 'LOW_VOL'
            else:
                # Look at trend
                if len(market_data) > 20:
                    spy_20d = market_data['SPY'].rolling(20).mean()
                    current_spy_ma = spy_20d.loc[date] if date in spy_20d.index else spy
                    
                    if spy > current_spy_ma * 1.02:
                        regime = 'BULL'
                    elif spy < current_spy_ma * 0.98:
                        regime = 'BEAR'
                    else:
                        regime = 'SIDEWAYS'
                else:
                    regime = 'SIDEWAYS'
            
            regimes.append(regime)
        
        return pd.Series(regimes, index=market_data.index)
    
    def _calculate_regime_confidence(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate confidence in regime detection."""
        
        # Simple confidence based on VIX level and trend consistency
        confidences = []
        
        for i, (date, row) in enumerate(market_data.iterrows()):
            vix = row.get('VIX', 20)
            
            # Base confidence on VIX extremes
            if vix > 35 or vix < 12:
                confidence = 0.9  # High confidence in extreme regimes
            elif vix > 25 or vix < 18:
                confidence = 0.7  # Medium confidence
            else:
                confidence = 0.5  # Low confidence in neutral regimes
            
            # Adjust for trend consistency (look at last 5 days)
            if i >= 5:
                recent_regimes = market_data['market_regime'].iloc[i-4:i+1]
                consistency = (recent_regimes == recent_regimes.iloc[-1]).mean()
                confidence *= consistency
            
            confidences.append(confidence)
        
        return pd.Series(confidences, index=market_data.index)
    
    def _store_market_indicators(self, data: pd.DataFrame):
        """Store market indicators in database."""
        
        with sqlite3.connect(self.db_path) as conn:
            for date, row in data.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO market_indicators (
                        date, vix, spy_close, qqq_close, dxy_close, tnx_close,
                        market_regime, regime_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date.date(),
                    row.get('VIX'),
                    row.get('SPY'),
                    row.get('QQQ'),
                    row.get('DXY'),
                    row.get('^TNX'),
                    row.get('market_regime'),
                    row.get('regime_confidence')
                ))
            
            conn.commit()
    
    def create_walk_forward_splits(self, symbol: str, 
                                  train_years: int = 4, 
                                  test_months: int = 6) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create walk-forward validation splits."""
        
        # Load data from database
        data = self.load_stock_data(symbol)
        
        if data.empty or len(data) < train_years * 252:  # Not enough data
            return []
        
        splits = []
        
        # Start with minimum training period
        start_idx = train_years * 252  # Approximate trading days
        test_days = test_months * 21  # Approximate trading days per month
        
        while start_idx + test_days < len(data):
            # Training set: from beginning to start_idx
            train_data = data.iloc[:start_idx].copy()
            
            # Test set: next test_days
            test_data = data.iloc[start_idx:start_idx + test_days].copy()
            
            splits.append((train_data, test_data))
            
            # Move forward by test period
            start_idx += test_days
        
        self.logger.info(f"Created {len(splits)} walk-forward splits for {symbol}")
        return splits
    
    def load_stock_data(self, symbol: str, start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Load stock data from database."""
        
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM stock_data WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date.date())
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.date())
            
            query += " ORDER BY date"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            return df
    
    def get_data_quality_report(self, symbol: Optional[str] = None) -> Dict:
        """Get data quality report."""
        
        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                # Symbol-specific report
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_days,
                        AVG(data_quality_score) as avg_quality,
                        MIN(data_quality_score) as min_quality,
                        MIN(date) as start_date,
                        MAX(date) as end_date
                    FROM stock_data WHERE symbol = ?
                """, (symbol,))
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        'symbol': symbol,
                        'total_days': result[0],
                        'avg_quality': result[1],
                        'min_quality': result[2],
                        'start_date': result[3],
                        'end_date': result[4]
                    }
            else:
                # Overall report
                cursor = conn.execute("""
                    SELECT 
                        COUNT(DISTINCT symbol) as total_symbols,
                        COUNT(*) as total_records,
                        AVG(data_quality_score) as avg_quality,
                        MIN(data_quality_score) as min_quality
                    FROM stock_data
                """)
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        'total_symbols': result[0],
                        'total_records': result[1],
                        'avg_quality': result[2],
                        'min_quality': result[3]
                    }
        
        return {}
    
    def optimize_storage_performance(self):
        """Optimize database for better performance."""
        
        with sqlite3.connect(self.db_path) as conn:
            # Analyze tables
            conn.execute("ANALYZE")
            
            # Vacuum to reclaim space
            conn.execute("VACUUM")
            
            # Update statistics
            conn.execute("PRAGMA optimize")
            
            conn.commit()
        
        self.logger.info("Database optimization complete")
    
    def batch_collect_data(self, symbols: List[str], years: int = 7) -> Dict[str, pd.DataFrame]:
        """Batch collect data for multiple symbols."""
        
        results = {}
        
        for i, symbol in enumerate(symbols):
            self.logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
            
            try:
                data = self.collect_extended_data(symbol, years)
                if not data.empty:
                    results[symbol] = data
                
                # Small delay to avoid rate limiting
                import time
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Failed to collect data for {symbol}: {str(e)}")
        
        self.logger.info(f"Successfully collected data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw OHLCV data for NEAT training."""
        
        if data.empty:
            return pd.DataFrame()
        
        df = data.copy()
        
        try:
            # Basic price features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['high_low_ratio'] = df['High'] / df['Low']
            df['close_open_ratio'] = df['Close'] / df['Open']
            
            # Moving averages
            for period in [5, 10, 20, 50, 200]:
                df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
                df[f'price_vs_sma_{period}'] = df['Close'] / df[f'sma_{period}'] - 1
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma_20 = df['Close'].rolling(window=bb_period).mean()
            std_20 = df['Close'].rolling(window=bb_period).std()
            df['bb_upper'] = sma_20 + (std_20 * bb_std)
            df['bb_lower'] = sma_20 - (std_20 * bb_std)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volatility features
            df['volatility_5'] = df['returns'].rolling(window=5).std()
            df['volatility_20'] = df['returns'].rolling(window=20).std()
            df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
            
            # ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
            low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Volume features
            df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
            df['price_volume'] = df['Close'] * df['Volume']
            
            # Price action features
            df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            df['intraday_return'] = (df['Close'] - df['Open']) / df['Open']
            df['overnight_return'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            
            # Momentum features
            for period in [5, 10, 20]:
                df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            
            # Support/Resistance levels
            df['high_20'] = df['High'].rolling(window=20).max()
            df['low_20'] = df['Low'].rolling(window=20).min()
            df['price_vs_high_20'] = df['Close'] / df['high_20'] - 1
            df['price_vs_low_20'] = df['Close'] / df['low_20'] - 1
            
            # Market microstructure
            df['spread'] = (df['High'] - df['Low']) / df['Close']
            df['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
            df['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
            
            # Remove rows with NaN values (due to rolling calculations)
            df = df.dropna()
            
            # Select only feature columns (exclude OHLCV and metadata)
            feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'symbol', 'data_quality_score']]
            features_df = df[feature_cols].copy()
            
            self.logger.info(f"Engineered {len(feature_cols)} features from {len(df)} data points")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {str(e)}")
            return pd.DataFrame()
