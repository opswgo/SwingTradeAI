"""
Alpha Vantage data fetcher - free alternative to yfinance
Get free API key at: https://www.alphavantage.co/support/#api-key
"""

import pandas as pd
import requests
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
import time


class AlphaVantageDataFetcher:
    """Free alternative to yfinance using Alpha Vantage API."""
    
    def __init__(self, api_key: str = "demo"):
        """
        Initialize with Alpha Vantage API key.
        Use 'demo' for testing (limited data) or get free key at alphavantage.co
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # 5 calls/minute = 12 seconds between calls
        self.last_call_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting for free tier."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            print(f"⏱️ Rate limiting: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    async def fetch_historical_data(self, symbol: str, period: str = "2y") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch historical data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period (not used for Alpha Vantage, always returns full history)
        
        Returns:
            Tuple of (DataFrame with OHLCV data, validation report)
        """
        print(f"📊 Fetching {symbol} data from Alpha Vantage...")
        
        # Rate limiting
        self._rate_limit()
        
        # Alpha Vantage parameters
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': 'full',  # Get full historical data
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
            
            if 'Note' in data:
                raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")
            
            # Extract time series data
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                available_keys = list(data.keys())
                raise ValueError(f"No time series data found. Available keys: {available_keys}")
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Adj Close': float(values['5. adjusted close']),
                    'Volume': int(values['6. volume'])
                })
            
            # Create DataFrame
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)  # Sort by date ascending
            
            # Filter by period if needed
            if period != "max":
                days_map = {
                    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
                    "6mo": 180, "1y": 365, "2y": 730, "5y": 1825
                }
                if period in days_map:
                    cutoff_date = datetime.now() - timedelta(days=days_map[period])
                    df = df[df.index >= cutoff_date]
            
            print(f"✅ Fetched {len(df)} days of data for {symbol}")
            print(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
            
            # Validation report
            validation_report = {
                'symbol': symbol,
                'records': len(df),
                'date_range': f"{df.index[0].date()} to {df.index[-1].date()}",
                'data_quality': 'good' if len(df) > 100 else 'limited',
                'source': 'Alpha Vantage'
            }
            
            return df, validation_report
            
        except Exception as e:
            print(f"❌ Alpha Vantage fetch failed: {str(e)}")
            raise ValueError(f"Failed to fetch data from Alpha Vantage: {str(e)}")


class StooqDataFetcher:
    """Free alternative using Stooq direct CSV downloads."""
    
    @staticmethod
    async def fetch_historical_data(symbol: str, period: str = "2y") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch historical data from Stooq.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period
        
        Returns:
            Tuple of (DataFrame with OHLCV data, validation report)
        """
        print(f"📊 Fetching {symbol} data from Stooq...")
        
        # Stooq URL format
        stooq_symbol = f"{symbol}.US"  # Add .US for US stocks
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
        
        try:
            # Download CSV data with SSL verification disabled
            import ssl
            import urllib.request
            
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            with urllib.request.urlopen(url, context=ssl_context) as response:
                df = pd.read_csv(response)
            
            if df.empty:
                raise ValueError(f"No data available for {symbol} from Stooq")
            
            # Standardize column names
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Add Adj Close (same as Close for simplicity)
            df['Adj Close'] = df['Close']
            
            # Filter by period
            if period != "max":
                days_map = {
                    "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
                    "6mo": 180, "1y": 365, "2y": 730, "5y": 1825
                }
                if period in days_map:
                    cutoff_date = datetime.now() - timedelta(days=days_map[period])
                    df = df[df.index >= cutoff_date]
            
            print(f"✅ Fetched {len(df)} days of data for {symbol}")
            print(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
            
            # Validation report
            validation_report = {
                'symbol': symbol,
                'records': len(df),
                'date_range': f"{df.index[0].date()} to {df.index[-1].date()}",
                'data_quality': 'good' if len(df) > 100 else 'limited',
                'source': 'Stooq'
            }
            
            return df, validation_report
            
        except Exception as e:
            print(f"❌ Stooq fetch failed: {str(e)}")
            raise ValueError(f"Failed to fetch data from Stooq: {str(e)}")


class YahooDirectFetcher:
    """Direct Yahoo Finance CSV fetcher (bypasses yfinance library)."""
    
    @staticmethod
    async def fetch_historical_data(symbol: str, period: str = "2y") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch historical data directly from Yahoo Finance CSV endpoint.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period
        
        Returns:
            Tuple of (DataFrame with OHLCV data, validation report)
        """
        print(f"📊 Fetching {symbol} data from Yahoo Finance direct CSV...")
        
        # Calculate date range
        end_date = int(datetime.now().timestamp())
        
        days_map = {
            "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
            "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "max": 7300
        }
        days = days_map.get(period, 730)  # Default to 2 years
        start_date = int((datetime.now() - timedelta(days=days)).timestamp())
        
        # Yahoo Finance direct CSV URL
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
        params = {
            'period1': start_date,
            'period2': end_date,
            'interval': '1d',
            'events': 'history'
        }
        
        try:
            # Build URL with parameters
            import urllib.parse
            query_string = urllib.parse.urlencode(params)
            full_url = f"{url}?{query_string}"
            
            # Download CSV data
            df = pd.read_csv(full_url)
            
            if df.empty:
                raise ValueError(f"No data available for {symbol} from Yahoo Finance")
            
            # Set date index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Rename columns to match expected format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            
            print(f"✅ Fetched {len(df)} days of data for {symbol}")
            print(f"📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
            
            # Validation report
            validation_report = {
                'symbol': symbol,
                'records': len(df),
                'date_range': f"{df.index[0].date()} to {df.index[-1].date()}",
                'data_quality': 'good' if len(df) > 100 else 'limited',
                'source': 'Yahoo Finance Direct'
            }
            
            return df, validation_report
            
        except Exception as e:
            print(f"❌ Yahoo Direct fetch failed: {str(e)}")
            raise ValueError(f"Failed to fetch data from Yahoo Finance direct: {str(e)}")


async def test_all_fetchers():
    """Test all data fetchers to see which ones work."""
    symbol = "AAPL"
    fetchers = [
        ("Stooq", StooqDataFetcher()),
        ("Yahoo Direct", YahooDirectFetcher()),
        ("Alpha Vantage (demo)", AlphaVantageDataFetcher("demo"))
    ]
    
    for name, fetcher in fetchers:
        try:
            print(f"\n🧪 Testing {name}...")
            data, report = await fetcher.fetch_historical_data(symbol, "1y")
            print(f"✅ {name} SUCCESS: {len(data)} records")
            print(f"   Sample data: Close prices from {data['Close'].iloc[0]:.2f} to {data['Close'].iloc[-1]:.2f}")
            return fetcher, name  # Return first working fetcher
        except Exception as e:
            print(f"❌ {name} FAILED: {str(e)}")
    
    raise ValueError("All data fetchers failed")


if __name__ == "__main__":
    asyncio.run(test_all_fetchers())
