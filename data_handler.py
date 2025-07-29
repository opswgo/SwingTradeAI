#!/usr/bin/env python3
"""
Data handler for stock price prediction system.
Fetches real market data and prepares features for NEAT algorithm.
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta

class DataHandler:
    """
    Handles data fetching, feature engineering, and preprocessing for stock price prediction.
    Only uses real market data, no synthetic data generation or estimation.
    """
    
    def __init__(self):
        """Initialize the data handler with logging."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def fetch_data(self, ticker, period="10y"):
        """
        Fetch historical market data for the given ticker.
        Only uses real data, no synthetic data or fallbacks.
        
        Args:
            ticker (str): The stock ticker symbol
            period (str): Time period to fetch (default: "10y")
            
        Returns:
            pd.DataFrame: DataFrame with historical price data
            
        Raises:
            ValueError: If insufficient data is available
        """
        self.logger.info(f"Fetching {period} of historical data for {ticker}")
        
        try:
            # Fetch data from Yahoo Finance (real market data)
            data = yf.download(ticker, period=period)
            
            # Verify we have sufficient data (at least 252 trading days)
            if len(data) < 252:
                raise ValueError(f"Insufficient data for {ticker}: found {len(data)} days, need at least 252 trading days (1 year)")
                
            self.logger.info(f"Successfully fetched {len(data)} days of data for {ticker}")
            
            # Save raw data
            os.makedirs(f"{ticker}", exist_ok=True)
            data.to_csv(f"{ticker}/{ticker}_data.csv")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
            raise
    
    def engineer_features(self, data):
        """
        Generate technical indicators as features for the NEAT algorithm.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            tuple: (features DataFrame, normalized features numpy array)
        """
        self.logger.info("Engineering technical features")
        
        try:
            # Calculate technical indicators
            features = pd.DataFrame(index=data.index)
            
            # 1. RSI (14-day)
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            features['RSI'] = 100 - (100 / (1 + rs))
            
            # 2. MACD
            ema12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema26 = data['Close'].ewm(span=26, adjust=False).mean()
            features['MACD'] = ema12 - ema26
            
            # 3. ATR (Average True Range)
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features['ATR'] = true_range.rolling(window=14).mean()
            
            # 4. Volume (normalized)
            features['Volume'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
            
            # 5. Rate of Change (ROC)
            features['ROC'] = data['Close'].pct_change(10) * 100
            
            # 6. Bollinger Bands Width
            rolling_mean = data['Close'].rolling(window=20).mean()
            rolling_std = data['Close'].rolling(window=20).std()
            features['BB_Width'] = (rolling_std * 2) / rolling_mean
            
            # 7. 10-day Moving Average
            features['MA10'] = data['Close'] / data['Close'].rolling(window=10).mean() - 1
            
            # Drop NaN values (from the rolling calculations)
            features = features.dropna()
            
            # Z-score normalization
            normalized_features = (features - features.mean()) / features.std()
            
            # Replace any remaining infinities or NaN
            normalized_features = normalized_features.replace([np.inf, -np.inf], np.nan)
            normalized_features = normalized_features.fillna(0)
            
            self.logger.info(f"Successfully engineered {len(features.columns)} features, resulting in {len(normalized_features)} valid data points")
            
            return features, normalized_features.values
            
        except Exception as e:
            self.logger.error(f"Failed to engineer features: {str(e)}")
            raise
    
    def create_targets(self, data, prediction_days=56):
        """
        Create target values for prediction (both actual prices and binary up/down).
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            prediction_days (int): Number of days to predict ahead (default: 56 for 8 weeks)
            
        Returns:
            tuple: (price_targets, direction_targets)
        """
        self.logger.info(f"Creating targets for {prediction_days} days ahead")
        
        try:
            # Get closing prices
            closes = data['Close'].values
            
            # Create arrays for price targets and direction targets
            price_targets = np.zeros((len(closes) - prediction_days, prediction_days))
            direction_targets = np.zeros((len(closes) - prediction_days, prediction_days))
            
            # For each day in our dataset (except the last 'prediction_days' days)
            for i in range(len(closes) - prediction_days):
                # Store the actual future prices
                future_prices = closes[i+1:i+prediction_days+1]
                current_price = closes[i]
                
                # Store the price targets (normalized to current price)
                price_targets[i] = future_prices / current_price
                
                # Store direction targets (1 for up, 0 for down)
                direction_targets[i] = (future_prices > current_price).astype(int)
            
            self.logger.info(f"Successfully created targets with shape: {price_targets.shape}")
            
            return price_targets, direction_targets
            
        except Exception as e:
            self.logger.error(f"Failed to create targets: {str(e)}")
            raise
    
    def prepare_data(self, ticker, period="10y"):
        """
        Prepare data for NEAT training: fetch, engineer features, create targets.
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period to fetch
            
        Returns:
            dict: Dictionary with all data needed for training
        """
        self.logger.info(f"Preparing complete dataset for {ticker}")
        
        try:
            # Fetch data
            data = self.fetch_data(ticker, period)
            
            # Engineer features
            features_df, normalized_features = self.engineer_features(data)
            
            # Create targets
            price_targets, direction_targets = self.create_targets(data)
            
            # Align data (ensure same length for features and targets)
            min_len = min(len(normalized_features), len(price_targets))
            aligned_features = normalized_features[:min_len]
            aligned_price_targets = price_targets[:min_len]
            aligned_direction_targets = direction_targets[:min_len]
            aligned_dates = features_df.index[:min_len]
            
            # Create dates for future predictions (for the final prediction)
            last_date = data.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, 57)]  # 56 days (8 weeks)
            
            # Save aligned data
            os.makedirs(f"{ticker}", exist_ok=True)
            np.save(f"{ticker}/features.npy", aligned_features)
            np.save(f"{ticker}/price_targets.npy", aligned_price_targets)
            np.save(f"{ticker}/direction_targets.npy", aligned_direction_targets)
            
            # Prepare data dictionary
            result = {
                'features': aligned_features,
                'price_targets': aligned_price_targets,
                'direction_targets': aligned_direction_targets,
                'dates': aligned_dates,
                'future_dates': future_dates,
                'last_price': data['Close'].iloc[-1],
                'last_date': last_date,
                'data': data
            }
            
            self.logger.info(f"Successfully prepared dataset with {min_len} samples")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to prepare data: {str(e)}")
            raise
