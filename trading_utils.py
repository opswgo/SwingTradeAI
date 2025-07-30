import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class TradingMetrics:
    @staticmethod
    def calculate_metrics(actual, predicted):
        """Calculate and return key performance metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        
        # Direction accuracy
        direction_accuracy = np.mean((np.diff(actual) * np.diff(predicted)) > 0) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }
    
    @staticmethod
    def calculate_drawdown(returns):
        """Calculate the maximum drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

class DataProcessor:
    @staticmethod
    def prepare_features(df, sequence_length=12, predict_steps=1, target_col='Price'):
        """Prepare features for time series prediction"""
        # Ensure we have a datetime index
        if 'Date' in df.columns:
            df = df.set_index('Date')
        
        # Create sequences
        X, y = [], []
        for i in range(len(df) - sequence_length - predict_steps + 1):
            X.append(df.iloc[i:(i + sequence_length)].values)
            y.append(df[target_col].iloc[i + sequence_length:i + sequence_length + predict_steps].values)
            
        return np.array(X), np.array(y), df.index[sequence_length:len(df)-predict_steps+1]
    
    @staticmethod
    def add_technical_indicators(df):
        """Add common technical indicators to the dataframe"""
        # Simple Moving Averages
        df['SMA_20'] = df['Price'].rolling(window=20).mean()
        df['SMA_50'] = df['Price'].rolling(window=50).mean()
        
        # RSI
        delta = df['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Price'].ewm(span=12, adjust=False).mean()
        exp2 = df['Price'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['Middle_Band'] = df['Price'].rolling(window=20).mean()
        df['Upper_Band'] = df['Middle_Band'] + (df['Price'].rolling(window=20).std() * 2)
        df['Lower_Band'] = df['Middle_Band'] - (df['Price'].rolling(window=20).std() * 2)
        
        return df.dropna()
