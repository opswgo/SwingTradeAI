"""
Production-ready yfinance data ingestion pipeline for NEAT-based options system.
Handles real-time stock and options data with comprehensive validation and error handling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from database import db_manager

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """Production-ready data ingestion pipeline with real data validation."""
    
    def __init__(self, max_workers: int = 5, retry_attempts: int = 3, retry_delay: float = 1.0):
        """
        Initialize data ingestion pipeline.
        
        Args:
            max_workers: Maximum concurrent threads for data fetching
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retry attempts in seconds
        """
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.session = requests.Session()
        
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if a stock symbol exists and has data."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if we got valid info
            if not info or 'symbol' not in info:
                logger.warning(f"Symbol {symbol} not found or invalid")
                return False
                
            # Check if it's a valid equity
            if info.get('quoteType') != 'EQUITY':
                logger.warning(f"Symbol {symbol} is not an equity")
                return False
                
            # Check market cap requirement (≥$5B)
            market_cap = info.get('marketCap', 0)
            if market_cap < 5_000_000_000:  # $5B minimum
                logger.warning(f"Symbol {symbol} market cap ${market_cap:,.0f} below $5B minimum")
                return False
                
            logger.info(f"Symbol {symbol} validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    def fetch_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data with validation and error handling.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        for attempt in range(self.retry_attempts):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    auto_adjust=False,
                    prepost=False,
                    threads=True
                )
                
                if data.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return None
                
                # Validate data quality
                if not self._validate_stock_data(data, symbol):
                    logger.error(f"Data validation failed for {symbol}")
                    return None
                
                # Clean and prepare data
                data = self._clean_stock_data(data)
                
                logger.info(f"Successfully fetched {len(data)} days of data for {symbol}")
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All attempts failed for {symbol}")
                    return None
    
    def fetch_options_chain(self, symbol: str, expiration_dates: List[str] = None) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Fetch options chain data with Greeks calculation.
        
        Args:
            symbol: Stock symbol
            expiration_dates: List of expiration dates to fetch (YYYY-MM-DD format)
            
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames or None if failed
        """
        for attempt in range(self.retry_attempts):
            try:
                ticker = yf.Ticker(symbol)
                
                # Get available expiration dates if not provided
                if expiration_dates is None:
                    expiration_dates = ticker.options
                    if not expiration_dates:
                        logger.warning(f"No options available for {symbol}")
                        return None
                
                # Filter to 1-8 week expirations only
                today = datetime.now().date()
                valid_expirations = []
                
                for exp_date in expiration_dates:
                    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d').date()
                    days_to_exp = (exp_datetime - today).days
                    weeks_to_exp = days_to_exp / 7
                    
                    # Only include 1-8 week expirations
                    if 7 <= days_to_exp <= 56:  # 1-8 weeks
                        valid_expirations.append(exp_date)
                
                if not valid_expirations:
                    logger.warning(f"No valid 1-8 week expirations found for {symbol}")
                    return None
                
                # Fetch options data for valid expirations
                all_calls = []
                all_puts = []
                
                for exp_date in valid_expirations[:4]:  # Limit to 4 expirations for performance
                    try:
                        option_chain = ticker.option_chain(exp_date)
                        
                        if not option_chain.calls.empty:
                            calls_df = option_chain.calls.copy()
                            calls_df['expiration'] = exp_date
                            calls_df['option_type'] = 'call'
                            all_calls.append(calls_df)
                        
                        if not option_chain.puts.empty:
                            puts_df = option_chain.puts.copy()
                            puts_df['expiration'] = exp_date
                            puts_df['option_type'] = 'put'
                            all_puts.append(puts_df)
                            
                        # Small delay to avoid rate limiting
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.warning(f"Failed to fetch options for {symbol} expiration {exp_date}: {e}")
                        continue
                
                if not all_calls and not all_puts:
                    logger.warning(f"No options data retrieved for {symbol}")
                    return None
                
                # Combine all data
                result = {}
                if all_calls:
                    result['calls'] = pd.concat(all_calls, ignore_index=True)
                    result['calls'] = self._clean_options_data(result['calls'])
                
                if all_puts:
                    result['puts'] = pd.concat(all_puts, ignore_index=True)
                    result['puts'] = self._clean_options_data(result['puts'])
                
                logger.info(f"Successfully fetched options data for {symbol}")
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol} options: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All attempts failed for {symbol} options")
                    return None
    
    def _validate_stock_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate stock data quality and completeness."""
        
        # Check if data is empty
        if data.empty:
            logger.error(f"Empty data for {symbol}")
            return False
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing columns for {symbol}: {missing_columns}")
            return False
        
        # Check for excessive NaN values
        nan_percentage = data.isnull().sum().max() / len(data)
        if nan_percentage > 0.1:  # More than 10% NaN values
            logger.error(f"Excessive NaN values for {symbol}: {nan_percentage:.2%}")
            return False
        
        # Check for unrealistic price movements (>50% daily change)
        daily_returns = data['Close'].pct_change().abs()
        extreme_moves = (daily_returns > 0.5).sum()
        if extreme_moves > len(data) * 0.01:  # More than 1% of days with >50% moves
            logger.warning(f"Unusual price volatility detected for {symbol}")
        
        # Check for zero or negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if (data[col] <= 0).any():
                logger.error(f"Zero or negative prices found in {col} for {symbol}")
                return False
        
        # Check High >= Low logic
        if (data['High'] < data['Low']).any():
            logger.error(f"High < Low violation found for {symbol}")
            return False
        
        logger.info(f"Data validation passed for {symbol}")
        return True
    
    def _clean_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare stock data for storage."""
        
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Ensure proper data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        if 'Volume' in data.columns:
            data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce').astype('int64')
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    def _clean_options_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare options data for storage."""
        
        # Remove rows with missing critical data
        data = data.dropna(subset=['strike', 'lastPrice'])
        
        # Ensure proper data types
        numeric_columns = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Filter out options with unrealistic data
        data = data[data['strike'] > 0]
        data = data[data['lastPrice'] > 0]
        
        # Sort by strike price
        data = data.sort_values('strike')
        
        return data
    
    def calculate_greeks(self, options_data: pd.DataFrame, stock_price: float, risk_free_rate: float = 0.05) -> pd.DataFrame:
        """
        Calculate options Greeks using Black-Scholes approximation.
        
        Args:
            options_data: Options DataFrame
            stock_price: Current stock price
            risk_free_rate: Risk-free interest rate
            
        Returns:
            DataFrame with Greeks added
        """
        try:
            from scipy.stats import norm
            import math
            
            def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
                """Calculate Black-Scholes Greeks."""
                if T <= 0 or sigma <= 0:
                    return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
                
                d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
                d2 = d1 - sigma * math.sqrt(T)
                
                if option_type == 'call':
                    delta = norm.cdf(d1)
                    rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
                else:  # put
                    delta = -norm.cdf(-d1)
                    rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
                
                gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) 
                        - r * K * math.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)) / 365
                vega = S * norm.pdf(d1) * math.sqrt(T) / 100
                
                return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}
            
            # Calculate Greeks for each option
            greeks_data = []
            today = datetime.now().date()
            
            for _, row in options_data.iterrows():
                try:
                    # Calculate time to expiration
                    exp_date = datetime.strptime(row['expiration'], '%Y-%m-%d').date()
                    T = (exp_date - today).days / 365.0
                    
                    # Use implied volatility if available, otherwise estimate
                    sigma = row.get('impliedVolatility', 0.3)
                    if pd.isna(sigma) or sigma <= 0:
                        sigma = 0.3  # Default 30% volatility
                    
                    # Calculate Greeks
                    greeks = black_scholes_greeks(
                        S=stock_price,
                        K=row['strike'],
                        T=T,
                        r=risk_free_rate,
                        sigma=sigma,
                        option_type=row['option_type']
                    )
                    
                    greeks_data.append(greeks)
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate Greeks for option: {e}")
                    greeks_data.append({'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0})
            
            # Add Greeks to DataFrame
            greeks_df = pd.DataFrame(greeks_data)
            for col in greeks_df.columns:
                options_data[col] = greeks_df[col]
            
            return options_data
            
        except ImportError:
            logger.warning("scipy not available, Greeks calculation skipped")
            # Add zero Greeks if scipy not available
            options_data['delta'] = 0
            options_data['gamma'] = 0
            options_data['theta'] = 0
            options_data['vega'] = 0
            options_data['rho'] = 0
            return options_data
    
    def ingest_symbol_data(self, symbol: str, years_back: int = 15) -> bool:
        """
        Ingest complete historical data for a symbol.
        
        Args:
            symbol: Stock symbol to ingest
            years_back: Number of years of historical data to fetch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate symbol first
            if not self.validate_symbol(symbol):
                return False
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365)
            
            logger.info(f"Starting data ingestion for {symbol} from {start_date.date()} to {end_date.date()}")
            
            # Fetch stock data
            stock_data = self.fetch_stock_data(symbol, start_date, end_date)
            if stock_data is None:
                logger.error(f"Failed to fetch stock data for {symbol}")
                return False
            
            # Store stock data
            rows_inserted = db_manager.insert_stock_data(symbol, stock_data)
            logger.info(f"Inserted {rows_inserted} stock data rows for {symbol}")
            
            # Fetch current options data
            current_price = stock_data['Close'].iloc[-1]
            options_data = self.fetch_options_chain(symbol)
            
            if options_data:
                # Process puts (focus on cash-secured puts)
                if 'puts' in options_data:
                    puts_df = options_data['puts']
                    
                    # Calculate Greeks
                    puts_df = self.calculate_greeks(puts_df, current_price)
                    
                    # Convert to database format
                    options_records = []
                    today = datetime.now().date()
                    
                    for _, row in puts_df.iterrows():
                        options_records.append({
                            'expiration_date': datetime.strptime(row['expiration'], '%Y-%m-%d').date(),
                            'strike': row['strike'],
                            'option_type': 'put',
                            'last_price': row.get('lastPrice'),
                            'bid': row.get('bid'),
                            'ask': row.get('ask'),
                            'volume': row.get('volume'),
                            'open_interest': row.get('openInterest'),
                            'implied_volatility': row.get('impliedVolatility'),
                            'delta': row.get('delta'),
                            'gamma': row.get('gamma'),
                            'theta': row.get('theta'),
                            'vega': row.get('vega'),
                            'rho': row.get('rho'),
                            'data_date': today
                        })
                    
                    if options_records:
                        rows_inserted = db_manager.insert_options_data(symbol, options_records)
                        logger.info(f"Inserted {rows_inserted} options data rows for {symbol}")
            
            logger.info(f"Successfully completed data ingestion for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Data ingestion failed for {symbol}: {e}")
            return False
    
    def ingest_multiple_symbols(self, symbols: List[str], years_back: int = 15) -> Dict[str, bool]:
        """
        Ingest data for multiple symbols concurrently.
        
        Args:
            symbols: List of stock symbols
            years_back: Number of years of historical data
            
        Returns:
            Dictionary mapping symbols to success status
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.ingest_symbol_data, symbol, years_back): symbol
                for symbol in symbols
            }
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Exception during ingestion of {symbol}: {e}")
                    results[symbol] = False
        
        # Log summary
        successful = sum(results.values())
        total = len(symbols)
        logger.info(f"Data ingestion completed: {successful}/{total} symbols successful")
        
        return results

# Global data ingestion instance
data_pipeline = DataIngestionPipeline()

def test_data_ingestion():
    """Test the data ingestion pipeline with real data."""
    
    # Test symbols (high-volume, liquid stocks)
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    logger.info("Starting data ingestion test with real symbols")
    
    # Test single symbol ingestion
    success = data_pipeline.ingest_symbol_data('AAPL', years_back=2)
    if success:
        print("✅ Single symbol ingestion test passed")
    else:
        print("❌ Single symbol ingestion test failed")
        return False
    
    # Test multiple symbol ingestion
    results = data_pipeline.ingest_multiple_symbols(test_symbols[:3], years_back=1)
    successful_count = sum(results.values())
    
    if successful_count >= 2:  # At least 2 out of 3 should succeed
        print(f"✅ Multiple symbol ingestion test passed ({successful_count}/3 successful)")
        return True
    else:
        print(f"❌ Multiple symbol ingestion test failed ({successful_count}/3 successful)")
        return False

if __name__ == "__main__":
    # Run data ingestion test
    if test_data_ingestion():
        print("✅ Data ingestion pipeline ready for production")
    else:
        print("❌ Data ingestion pipeline needs fixes")
