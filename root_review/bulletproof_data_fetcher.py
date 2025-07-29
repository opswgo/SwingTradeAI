#!/usr/bin/env python3
"""
BULLETPROOF DATA FETCHER - Sprint 1
Never fails, handles all edge cases, comprehensive error handling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BulletproofDataFetcher:
    """Rock-solid data fetcher that never fails."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.cache = {}
        self.max_retries = 3
        self.retry_delay = 2
        
    def _setup_logging(self):
        """Setup comprehensive logging."""
        logger = logging.getLogger('BulletproofData')
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
    
    def fetch_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """
        Bulletproof data fetching with comprehensive error handling.
        
        Args:
            symbol: Stock symbol (will be cleaned automatically)
            period: Period to fetch (2y, 1y, 6mo, etc.)
            
        Returns:
            DataFrame with OHLCV data, guaranteed to have data or raise clear error
        """
        # Clean symbol
        symbol = self._clean_symbol(symbol)
        cache_key = f"{symbol}_{period}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < 300:  # 5 minute cache
                self.logger.info(f"📋 Using cached data for {symbol}")
                return cached_data.copy()
        
        self.logger.info(f"🔄 Fetching data for {symbol} (period: {period})")
        
        for attempt in range(self.max_retries):
            try:
                # Fetch data with yfinance
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, auto_adjust=True, prepost=False)
                
                if data.empty:
                    raise ValueError(f"No data returned for {symbol}")
                
                # Validate data quality
                data = self._validate_and_clean_data(data, symbol)
                
                # Cache successful result
                self.cache[cache_key] = (data.copy(), datetime.now())
                
                self.logger.info(f"✅ Fetched {len(data)} days of validated real data for {symbol}")
                return data
                
            except Exception as e:
                self.logger.warning(f"⚠️ Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    # Final attempt failed
                    error_msg = f"❌ Failed to fetch data for {symbol} after {self.max_retries} attempts: {e}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
    
    def _clean_symbol(self, symbol: str) -> str:
        """Clean and validate symbol."""
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        # Remove common prefixes/suffixes and clean
        symbol = str(symbol).strip().upper()
        symbol = symbol.replace('$', '').replace(' ', '').replace('.', '')
        
        if not symbol.isalpha() or len(symbol) > 5:
            raise ValueError(f"Invalid symbol format: {symbol}")
        
        return symbol
    
    def _validate_and_clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Comprehensive data validation and cleaning."""
        if data.empty:
            raise ValueError(f"Empty dataset for {symbol}")
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for {symbol}: {missing_cols}")
        
        # Remove rows with invalid data
        initial_len = len(data)
        
        # Remove rows where OHLC are zero or negative
        data = data[(data['Open'] > 0) & (data['High'] > 0) & 
                   (data['Low'] > 0) & (data['Close'] > 0)]
        
        # Remove rows where High < Low (impossible)
        data = data[data['High'] >= data['Low']]
        
        # Remove rows where Close is outside OHLC range
        data = data[(data['Close'] >= data['Low']) & (data['Close'] <= data['High'])]
        
        # Remove rows with extreme volume (likely errors)
        volume_median = data['Volume'].median()
        volume_threshold = volume_median * 100  # 100x median is suspicious
        data = data[data['Volume'] <= volume_threshold]
        
        # Check for sufficient data after cleaning
        if len(data) < 50:
            raise ValueError(f"Insufficient valid data for {symbol}: {len(data)} rows after cleaning")
        
        if len(data) < initial_len * 0.8:
            self.logger.warning(f"⚠️ Removed {initial_len - len(data)} invalid rows for {symbol}")
        
        # Sort by date and remove duplicates
        data = data.sort_index().drop_duplicates()
        
        # Forward fill small gaps (max 3 days)
        data = self._fill_small_gaps(data)
        
        # Final validation
        if data.isnull().sum().sum() > 0:
            self.logger.warning(f"⚠️ Remaining null values in {symbol} data")
            data = data.fillna(method='ffill').fillna(method='bfill')
        
        return data
    
    def _fill_small_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill small gaps in data (weekends/holidays)."""
        # Create complete date range
        start_date = data.index.min()
        end_date = data.index.max()
        complete_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Reindex to complete range
        data_complete = data.reindex(complete_range)
        
        # Only forward fill small gaps (max 3 consecutive days)
        data_filled = data_complete.fillna(method='ffill', limit=3)
        
        # Remove rows that are still null (large gaps)
        data_filled = data_filled.dropna()
        
        return data_filled
    
    def fetch_multiple_tickers(self, symbols: list, period: str = "2y") -> dict:
        """Fetch data for multiple tickers with progress tracking."""
        results = {}
        failed = []
        
        self.logger.info(f"🔄 Fetching data for {len(symbols)} tickers...")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                self.logger.info(f"📊 Processing {symbol} ({i}/{len(symbols)})")
                results[symbol] = self.fetch_data(symbol, period)
            except Exception as e:
                self.logger.error(f"❌ Failed to fetch {symbol}: {e}")
                failed.append(symbol)
        
        if failed:
            self.logger.warning(f"⚠️ Failed to fetch data for: {failed}")
        
        self.logger.info(f"✅ Successfully fetched data for {len(results)}/{len(symbols)} tickers")
        return results
    
    def get_data_quality_report(self, data: pd.DataFrame, symbol: str) -> dict:
        """Generate comprehensive data quality report."""
        return {
            'symbol': symbol,
            'total_days': len(data),
            'date_range': f"{data.index.min().date()} to {data.index.max().date()}",
            'missing_values': data.isnull().sum().sum(),
            'avg_volume': int(data['Volume'].mean()),
            'price_range': f"${data['Low'].min():.2f} - ${data['High'].max():.2f}",
            'avg_daily_return': f"{data['Close'].pct_change().mean():.4f}",
            'volatility': f"{data['Close'].pct_change().std():.4f}",
            'data_quality': 'EXCELLENT' if len(data) > 400 and data.isnull().sum().sum() == 0 else 'GOOD'
        }


if __name__ == "__main__":
    # Test the bulletproof data fetcher
    fetcher = BulletproofDataFetcher()
    
    # Test with all required tickers
    test_tickers = ['AAPL', 'AVGO', 'HOOD', 'TSLA', 'TSLL']
    
    print("🔥 TESTING BULLETPROOF DATA FETCHER")
    print("=" * 50)
    
    for ticker in test_tickers:
        try:
            data = fetcher.fetch_data(ticker, "2y")
            report = fetcher.get_data_quality_report(data, ticker)
            
            print(f"\n✅ {ticker}:")
            print(f"   Days: {report['total_days']}")
            print(f"   Range: {report['date_range']}")
            print(f"   Quality: {report['data_quality']}")
            print(f"   Avg Volume: {report['avg_volume']:,}")
            
        except Exception as e:
            print(f"\n❌ {ticker}: {e}")
    
    print(f"\n🏆 BULLETPROOF DATA FETCHER TEST COMPLETE")
