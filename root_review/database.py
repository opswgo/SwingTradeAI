"""
Production-ready PostgreSQL database configuration and connection management.
Handles historical data storage for NEAT-based options premium harvesting system.
"""

import os
import logging
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Production-ready PostgreSQL database manager with connection pooling."""
    
    def __init__(self, host: str = 'localhost', port: int = 5432, 
                 database: str = 'swing_trade_ai', user: str = 'postgres', 
                 password: str = None, min_connections: int = 1, max_connections: int = 10):
        """
        Initialize database manager with connection pooling.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password (from env if not provided)
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password or os.getenv('POSTGRES_PASSWORD', 'password')
        self.connection_pool = None
        self.fallback_mode = False
        
        # Create connection pool
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                min_connections, max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            logger.info(f"Database connection pool created successfully")
            
            # Initialize database schema
            self._initialize_schema()
            
        except Exception as e:
            logger.warning(f"PostgreSQL not available, using fallback mode: {e}")
            self.fallback_mode = True
            self.connection_pool = None
            # Initialize in-memory fallback storage
            self._init_fallback_storage()
    
    def _init_fallback_storage(self):
        """Initialize in-memory storage for fallback mode."""
        self.stock_data_cache = {}
        self.options_data_cache = {}
        logger.info("Initialized fallback storage mode")
    
    def _initialize_schema(self):
        """Initialize database schema."""
        if self.fallback_mode:
            return  # Skip schema initialization in fallback mode
            
        try:
            with self.get_connection() as conn:
                self.create_tables()
                logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with automatic cleanup."""
        if self.fallback_mode:
            yield None  # Return None for fallback mode
            return
            
        connection = None
        try:
            connection = self.connection_pool.getconn()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if connection:
                self.connection_pool.putconn(connection)
    
    def execute_query(self, query: str, params: tuple = None) -> List[tuple]:
        """Execute a SELECT query and return results."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute multiple queries with different parameters."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount
    
    def create_tables(self):
        """Create all necessary tables for the NEAT options system."""
        
        # Historical stock data table
        stock_data_table = """
        CREATE TABLE IF NOT EXISTS stock_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            open DECIMAL(10,4) NOT NULL,
            high DECIMAL(10,4) NOT NULL,
            low DECIMAL(10,4) NOT NULL,
            close DECIMAL(10,4) NOT NULL,
            volume BIGINT NOT NULL,
            adjusted_close DECIMAL(10,4) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, date)
        );
        """
        
        # Options chain data table
        options_data_table = """
        CREATE TABLE IF NOT EXISTS options_data (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            expiration_date DATE NOT NULL,
            strike DECIMAL(10,4) NOT NULL,
            option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('call', 'put')),
            last_price DECIMAL(10,4),
            bid DECIMAL(10,4),
            ask DECIMAL(10,4),
            volume INTEGER,
            open_interest INTEGER,
            implied_volatility DECIMAL(8,6),
            delta DECIMAL(8,6),
            gamma DECIMAL(8,6),
            theta DECIMAL(8,6),
            vega DECIMAL(8,6),
            rho DECIMAL(8,6),
            data_date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, expiration_date, strike, option_type, data_date)
        );
        """
        
        # NEAT population data table
        neat_population_table = """
        CREATE TABLE IF NOT EXISTS neat_population (
            id SERIAL PRIMARY KEY,
            generation INTEGER NOT NULL,
            genome_id INTEGER NOT NULL,
            genome_data JSONB NOT NULL,
            fitness_score DECIMAL(10,6),
            species_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Trading signals table
        trading_signals_table = """
        CREATE TABLE IF NOT EXISTS trading_signals (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            signal_date DATE NOT NULL,
            signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('buy_put', 'sell_put', 'hold')),
            strike_price DECIMAL(10,4) NOT NULL,
            expiration_date DATE NOT NULL,
            confidence_score DECIMAL(6,4) NOT NULL,
            premium_yield DECIMAL(6,4),
            assignment_probability DECIMAL(6,4),
            model_version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Performance tracking table
        performance_table = """
        CREATE TABLE IF NOT EXISTS performance_tracking (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            trade_date DATE NOT NULL,
            entry_price DECIMAL(10,4) NOT NULL,
            exit_price DECIMAL(10,4),
            strike_price DECIMAL(10,4) NOT NULL,
            expiration_date DATE NOT NULL,
            premium_collected DECIMAL(10,4) NOT NULL,
            assignment_occurred BOOLEAN DEFAULT FALSE,
            profit_loss DECIMAL(10,4),
            trade_duration INTEGER,
            model_version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        # Create indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_date ON stock_data(symbol, date);",
            "CREATE INDEX IF NOT EXISTS idx_options_data_symbol_exp ON options_data(symbol, expiration_date);",
            "CREATE INDEX IF NOT EXISTS idx_options_data_date ON options_data(data_date);",
            "CREATE INDEX IF NOT EXISTS idx_neat_population_gen ON neat_population(generation);",
            "CREATE INDEX IF NOT EXISTS idx_trading_signals_date ON trading_signals(signal_date);",
            "CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_tracking(trade_date);"
        ]
        
        tables = [
            stock_data_table,
            options_data_table,
            neat_population_table,
            trading_signals_table,
            performance_table
        ]
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Create tables
                    for table_sql in tables:
                        cursor.execute(table_sql)
                        logger.info(f"Table created/verified successfully")
                    
                    # Create indexes
                    for index_sql in indexes:
                        cursor.execute(index_sql)
                        logger.info(f"Index created/verified successfully")
                    
                    conn.commit()
                    logger.info("All database tables and indexes created successfully")
                    
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def insert_stock_data(self, symbol: str, data: pd.DataFrame) -> int:
        """Insert stock data into the database."""
        
        insert_query = """
        INSERT INTO stock_data (symbol, date, open, high, low, close, volume, adjusted_close)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            adjusted_close = EXCLUDED.adjusted_close;
        """
        
        # Prepare data for insertion
        records = []
        for date, row in data.iterrows():
            records.append((
                symbol,
                date.date(),
                float(row['Open']),
                float(row['High']),
                float(row['Low']),
                float(row['Close']),
                int(row['Volume']),
                float(row['Adj Close'])
            ))
        
        return self.execute_many(insert_query, records)
    
    def insert_options_data(self, symbol: str, options_data: List[Dict[str, Any]]) -> int:
        """Insert options data into the database."""
        
        insert_query = """
        INSERT INTO options_data (
            symbol, expiration_date, strike, option_type, last_price, bid, ask,
            volume, open_interest, implied_volatility, delta, gamma, theta, vega, rho, data_date
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, expiration_date, strike, option_type, data_date) DO UPDATE SET
            last_price = EXCLUDED.last_price,
            bid = EXCLUDED.bid,
            ask = EXCLUDED.ask,
            volume = EXCLUDED.volume,
            open_interest = EXCLUDED.open_interest,
            implied_volatility = EXCLUDED.implied_volatility,
            delta = EXCLUDED.delta,
            gamma = EXCLUDED.gamma,
            theta = EXCLUDED.theta,
            vega = EXCLUDED.vega,
            rho = EXCLUDED.rho;
        """
        
        records = []
        for option in options_data:
            records.append((
                symbol,
                option['expiration_date'],
                option['strike'],
                option['option_type'],
                option.get('last_price'),
                option.get('bid'),
                option.get('ask'),
                option.get('volume'),
                option.get('open_interest'),
                option.get('implied_volatility'),
                option.get('delta'),
                option.get('gamma'),
                option.get('theta'),
                option.get('vega'),
                option.get('rho'),
                option['data_date']
            ))
        
        return self.execute_many(insert_query, records)
    
    def get_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Retrieve stock data for a symbol within date range."""
        
        if self.fallback_mode:
            # Use yfinance as fallback data source
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date.date(), end=end_date.date())
                if not data.empty:
                    # Rename columns to match expected format
                    data = data.rename(columns={'Adj Close': 'Adj Close'})
                    return data
                else:
                    return pd.DataFrame()
            except Exception as e:
                logger.warning(f"Fallback data fetch failed for {symbol}: {e}")
                return pd.DataFrame()
        
        query = """
        SELECT date, open, high, low, close, volume, adjusted_close
        FROM stock_data
        WHERE symbol = %s AND date >= %s AND date <= %s
        ORDER BY date;
        """
        
        results = self.execute_query(query, (symbol, start_date.date(), end_date.date()))
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        return df
    
    def get_options_data(self, symbol: str, data_date: datetime) -> pd.DataFrame:
        """Retrieve options data for a symbol on a specific date."""
        
        if self.fallback_mode:
            # Return empty DataFrame in fallback mode (options data requires real-time API)
            logger.warning(f"Options data not available in fallback mode for {symbol}")
            return pd.DataFrame()
        
        query = """
        SELECT expiration_date, strike, option_type, last_price, bid, ask,
               volume, open_interest, implied_volatility, delta, gamma, theta, vega, rho
        FROM options_data
        WHERE symbol = %s AND data_date = %s
        ORDER BY expiration_date, strike, option_type;
        """
        
        results = self.execute_query(query, (symbol, data_date.date()))
        
        if not results:
            return pd.DataFrame()
        
        columns = ['expiration_date', 'strike', 'option_type', 'last_price', 'bid', 'ask',
                  'volume', 'open_interest', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho']
        
        return pd.DataFrame(results, columns=columns)
    
    def close_connections(self):
        """Close all database connections."""
        if hasattr(self, 'connection_pool'):
            self.connection_pool.closeall()
            logger.info("Database connections closed")

# Global database instance
db_manager = DatabaseManager()

def initialize_database():
    """Initialize the database with all required tables."""
    try:
        db_manager.create_tables()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    # Test database initialization
    if initialize_database():
        print("✅ Database initialized successfully")
    else:
        print("❌ Database initialization failed")
