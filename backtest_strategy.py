import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# Import our modules
from trading_utils import TradingMetrics, DataProcessor
from paper_trader import PaperTradingSystem

class Backtester:
    def __init__(self, model_path: str, data_path: str, sequence_length: int = 12, 
                 predict_steps: int = 1, initial_capital: float = 10000):
        """
        Initialize the backtester.
        
        Args:
            model_path: Path to the trained model
            data_path: Path to the market data CSV
            sequence_length: Number of time steps to use for prediction
            predict_steps: Number of steps ahead to predict
            initial_capital: Starting capital in dollars
        """
        self.model_path = model_path
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.predict_steps = predict_steps
        self.initial_capital = initial_capital
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.data = None
        self.results = None
        
        # Load data and model
        self._load_data()
        self._load_model()
    
    def _load_data(self):
        """Load and preprocess the market data"""
        print(f"Loading data from {self.data_path}...")
        self.data = pd.read_csv(self.data_path)
        
        # Ensure we have a datetime index
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.set_index('Date')
        
        # Add technical indicators
        self.data = DataProcessor.add_technical_indicators(self.data)
        
        # Ensure we have all required columns
        required_cols = ['Price', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def _load_model(self):
        """Load the trained PyTorch model"""
        print(f"Loading model from {self.model_path}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        try:
            # Import the module using importlib to handle spaces in filename
            import importlib.util
            import sys
            
            # Get the absolute path to the model file
            model_file = os.path.abspath('One File Code.py')
            
            # Create a module spec and load it
            spec = importlib.util.spec_from_file_location('samba_model', model_file)
            samba_module = importlib.util.module_from_spec(spec)
            sys.modules['samba_model'] = samba_module
            spec.loader.exec_module(samba_module)
            
            # Get the SAMBA class
            SAMBA = getattr(samba_module, 'SAMBA')
            
            # Get the model configuration from the saved state dict
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model with dummy parameters (will be overridden by state dict)
            model_args = samba_module.ModelArgs(
                d_model=checkpoint.get('d_model', 64),
                n_layer=checkpoint.get('n_layer', 4),
                vocab_size=len(self.data.columns),
                seq_in=self.sequence_length,
                d_inner=checkpoint.get('d_inner', 256),
                d_conv=checkpoint.get('d_conv', 4),
                dt_rank=checkpoint.get('dt_rank', 16),
                d_state=checkpoint.get('d_state', 16),
                bias=checkpoint.get('bias', True),
                conv_bias=checkpoint.get('conv_bias', True)
            )
            
            # Initialize model with the correct architecture
            self.model = SAMBA(
                model_args, 
                hidden=checkpoint.get('hidden', 64),
                inp=len(self.data.columns),
                out=self.predict_steps,
                embed=checkpoint.get('embed', 32),
                cheb_k=checkpoint.get('cheb_k', 3)
            )
            
            # Load model weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully in evaluation mode")
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to load model: {str(e)}\n\n{traceback.format_exc()}"
            raise RuntimeError(error_msg)
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """Prepare features for prediction"""
        # Select features
        feature_cols = ['Price', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']
        features = data[feature_cols].copy()
        
        # Normalize features
        if not hasattr(self, 'feature_scaler'):
            from sklearn.preprocessing import MinMaxScaler
            self.feature_scaler = MinMaxScaler()
            scaled_features = self.feature_scaler.fit_transform(features)
        else:
            scaled_features = self.feature_scaler.transform(features)
        
        # Create sequences
        X, y, dates = DataProcessor.prepare_features(
            pd.DataFrame(scaled_features, index=features.index, columns=features.columns),
            sequence_length=self.sequence_length,
            predict_steps=self.predict_steps,
            target_col='Price'
        )
        
        return X, y, dates
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the loaded model"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
    
    def generate_signals(self, predictions: np.ndarray, actual: np.ndarray, 
                        threshold: float = 0.005) -> np.ndarray:
        """
        Generate trading signals based on predictions
        
        Args:
            predictions: Predicted price changes
            actual: Actual price values
            threshold: Minimum percentage change to trigger a trade
            
        Returns:
            Array of signals: 1 for long, -1 for short, 0 for hold
        """
        signals = np.zeros_like(predictions)
        
        # Simple strategy: go long if predicted price increase > threshold
        # go short if predicted price decrease > threshold
        for i in range(len(predictions)):
            pred_return = (predictions[i] - actual[i]) / actual[i]
            
            if pred_return > threshold:
                signals[i] = 1  # Long
            elif pred_return < -threshold:
                signals[i] = -1  # Short
            else:
                signals[i] = 0  # Hold
                
        return signals
    
    def run_backtest(self, start_date: str = None, end_date: str = None, 
                    commission: float = 0.001, risk_per_trade: float = 0.01,
                    stop_loss_pct: float = 0.02, take_profit_pct: float = 0.04) -> Dict:
        """
        Run the backtest
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            commission: Commission per trade as a percentage
            risk_per_trade: Percentage of capital to risk per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            
        Returns:
            Dictionary containing backtest results
        """
        print("Starting backtest...")
        
        # Filter data by date range if specified
        if start_date:
            start_date = pd.to_datetime(start_date)
            self.data = self.data[self.data.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            self.data = self.data[self.data.index <= end_date]
        
        # Prepare features
        X, y, dates = self.prepare_features(self.data)
        
        # Make predictions
        print("Making predictions...")
        predictions = self.predict(X)
        
        # Generate signals
        print("Generating trading signals...")
        signals = self.generate_signals(predictions, y)
        
        # Initialize paper trading system
        trader = PaperTradingSystem(
            initial_capital=self.initial_capital,
            commission=commission,
            risk_per_trade=risk_per_trade,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct
        )
        
        # Run backtest
        print("Running backtest...")
        for i in range(len(dates)):
            if i < self.sequence_length:
                continue
                
            date = dates[i]
            price = self.data.loc[date, 'Price']
            signal = signals[i]
            
            # Update market data and check for stop loss/take profit
            trader.update_market_data('STOCK', self.data.loc[date])
            
            # Execute trade based on signal
            trader.execute_trade('STOCK', price, signal, date=date)
        
        # Close all open positions at the end
        for symbol in list(trader.positions.keys()):
            last_price = self.data.iloc[-1]['Close']
            trader.close_position(symbol, last_price, 'END_OF_BACKTEST')
        
        # Get performance metrics
        metrics = trader.get_performance_metrics()
        
        # Save results
        self.results = {
            'metrics': metrics,
            'trades': trader.get_trade_history_df().to_dict('records'),
            'portfolio_history': trader.get_portfolio_history_df().to_dict('records'),
            'predictions': predictions.tolist(),
            'actual': y.tolist(),
            'signals': signals.tolist(),
            'dates': [str(d) for d in dates]
        }
        
        return self.results
    
    def plot_results(self, save_path: str = None):
        """Plot backtest results"""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and predictions
        dates = pd.to_datetime(self.results['dates'])
        actual = np.array(self.results['actual']).flatten()
        predictions = np.array(self.results['predictions']).flatten()
        
        ax1.plot(dates, actual, label='Actual Price', color='blue', alpha=0.7)
        ax1.plot(dates, predictions, label='Predicted Price', color='red', alpha=0.7)
        
        # Mark buy/sell signals
        signals = np.array(self.results['signals']).flatten()
        buy_signals = dates[signals > 0]
        sell_signals = dates[signals < 0]
        
        ax1.scatter(buy_signals, actual[signals > 0], 
                   marker='^', color='green', s=100, label='Buy Signal')
        ax1.scatter(sell_signals, actual[signals < 0], 
                   marker='v', color='red', s=100, label='Sell Signal')
        
        ax1.set_title('Price and Predictions')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot portfolio value
        portfolio_values = [p['value'] for p in self.results['portfolio_history']]
        portfolio_dates = pd.to_datetime([p['date'] for p in self.results['portfolio_history']])
        
        ax2.plot(portfolio_dates, portfolio_values, label='Portfolio Value', color='green')
        ax2.axhline(self.initial_capital, color='red', linestyle='--', label='Initial Capital')
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_ylabel('Value ($)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, output_dir: str = 'backtest_results'):
        """Save backtest results to files"""
        if self.results is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to JSON
        results_path = os.path.join(output_dir, 'backtest_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save metrics to CSV
        metrics_path = os.path.join(output_dir, 'performance_metrics.csv')
        pd.DataFrame([self.results['metrics']]).to_csv(metrics_path, index=False)
        
        # Save trades to CSV
        trades_path = os.path.join(output_dir, 'trades.csv')
        pd.DataFrame(self.results['trades']).to_csv(trades_path, index=False)
        
        # Save portfolio history to CSV
        portfolio_path = os.path.join(output_dir, 'portfolio_history.csv')
        pd.DataFrame(self.results['portfolio_history']).to_csv(portfolio_path, index=False)
        
        # Plot results
        plot_path = os.path.join(output_dir, 'backtest_results.png')
        self.plot_results(plot_path)
        
        print(f"Backtest results saved to {output_dir}/")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Run backtest with SAMBA model')
    parser.add_argument('--model-path', type=str, default="AGCRN_Model/milan/call/1_stemgnn.pt",
                      help='Path to the trained model weights')
    parser.add_argument('--data-path', type=str, default="Dataset/combined_dataframe_IXIC.csv",
                      help='Path to the data CSV file')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip training and use pre-trained model')
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = args.model_path
    data_path = args.data_path
    skip_training = args.skip_training
    
    # Initialize backtester
    backtester = Backtester(
        model_path=model_path,
        data_path=data_path,
        sequence_length=12,
        predict_steps=1,
        initial_capital=10000
    )
    
    # Run backtest
    results = backtester.run_backtest(
        start_date="2020-01-01",
        end_date="2023-01-01",
        commission=0.001,
        risk_per_trade=0.01,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    # Save results
    backtester.save_results()
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    for k, v in results['metrics'].items():
        print(f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}")


if __name__ == "__main__":
    main()
