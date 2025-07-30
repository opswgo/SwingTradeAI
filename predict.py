#!/usr/bin/env python3
"""
Prediction Script for SAMBA Model

This script loads a pre-trained SAMBA model and makes predictions on new data.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def load_model(model_path, input_dim, predict_steps=1, device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'):
    """
    Load a pre-trained SAMBA model.
    
    Args:
        model_path: Path to the saved model file
        input_dim: Number of input features
        predict_steps: Number of steps to predict
        device: Device to run the model on ('cuda', 'mps', or 'cpu')
        
    Returns:
        Loaded model ready for inference
    """
    print(f"Loading model from {model_path}...")
    
    # Import the module using importlib to handle spaces in filename
    import importlib.util
    import types
    
    # Get the absolute path to the model file
    model_file = os.path.abspath('One File Code.py')
    
    # Create a custom module to prevent execution of module-level code
    module_name = 'samba_model'
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    module = types.ModuleType(module_name)
    
    # Only execute the code we need
    with open(model_file, 'r') as f:
        code = f.read()
    
    # Execute only the class definitions, not the training code
    # This is a simple way to avoid executing the training code at module level
    code_compiled = compile(code, model_file, 'exec')
    
    # Create a safe globals dict with only what we need
    safe_globals = {'__name__': module_name, '__file__': model_file}
    
    # Execute the code in our safe environment
    exec(code_compiled, safe_globals)
    
    # Get the SAMBA class
    SAMBA = safe_globals.get('SAMBA')
    if SAMBA is None:
        raise RuntimeError("Could not find SAMBA class in the model file")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get ModelArgs class from safe_globals
    ModelArgs = safe_globals.get('ModelArgs')
    if ModelArgs is None:
        raise RuntimeError("Could not find ModelArgs class in the model file")
    
    # Get model configuration from checkpoint or use defaults
    model_config = {
        'd_model': checkpoint.get('d_model', 64),
        'n_layer': checkpoint.get('n_layer', 4),
        'vocab_size': input_dim,
        'seq_in': checkpoint.get('seq_in', 12),  # Default sequence length
        'd_inner': checkpoint.get('d_inner', 256),
        'd_conv': checkpoint.get('d_conv', 4),
        'dt_rank': checkpoint.get('dt_rank', 16),
        'd_state': checkpoint.get('d_state', 16),
        'bias': checkpoint.get('bias', True),
        'conv_bias': checkpoint.get('conv_bias', True)
    }
    
    # Initialize model args
    model_args = ModelArgs(**model_config)
    
    # Get model initialization parameters
    model_params = {
        'model_args': model_args,
        'hidden': checkpoint.get('hidden', 64),
        'inp': input_dim,
        'out': predict_steps,
        'embed': checkpoint.get('embed', 32),
        'cheb_k': checkpoint.get('cheb_k', 3)
    }
    
    # Initialize model with the correct architecture
    model = SAMBA(**model_params)
    
    # Load weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on device: {device}")
    return model

def prepare_data(data_path, sequence_length=12, target_col='Price'):
    """
    Prepare data for prediction.
    
    Args:
        data_path: Path to CSV file with data
        sequence_length: Number of time steps to use for prediction
        target_col: Name of the target column
        
    Returns:
        X: Input sequences
        y: Target values
        dates: Corresponding dates
        scaler: Fitted scaler for inverse transform
    """
    print(f"Loading data from {data_path}...")
    
    # Read the data, ensuring we only keep numeric columns
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Ensure we have the target column in our numeric columns
    if target_col not in numeric_cols:
        raise ValueError(f"Target column '{target_col}' not found in numeric data. "
                        f"Available numeric columns: {numeric_cols}")
    
    # Keep only numeric columns
    df = df[numeric_cols].copy()
    
    # Check for and handle any remaining non-numeric values
    if df.isna().any().any():
        print("Warning: Found NaN values. Filling with forward fill then backward fill.")
        df = df.ffill().bfill()
        
        # If there are still NaNs, fill with 0 as last resort
        if df.isna().any().any():
            print("Warning: Still found NaN values after filling. Filling with 0.")
            df = df.fillna(0)
    
    # Ensure all data is numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    print(f"Using {len(df)} samples with {len(df.columns)} features")
    print(f"Features: {', '.join(df.columns)}")
    
    # Scale the data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length, df.columns.get_loc(target_col)])
    
    X = np.array(X)
    y = np.array(y)
    dates = df.index[sequence_length:]
    
    print(f"Prepared {len(X)} sequences of length {sequence_length}")
    return X, y, dates, scaler

def predict(model, X, device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'):
    """
    Make predictions using the model.
    
    Args:
        model: Loaded PyTorch model
        X: Input data
        device: Device to run inference on
        
    Returns:
        Predictions as numpy array
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor)
        return predictions.cpu().numpy()

def plot_predictions(dates, y_true, y_pred, title='Stock Price Prediction'):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_true, label='Actual', color='blue')
    plt.plot(dates, y_pred, label='Predicted', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plot_path = os.path.join('results', 'predictions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Make predictions with pre-trained SAMBA model')
    parser.add_argument('--model-path', type=str, default='AGCRN_Model/milan/call/1_stemgnn.pt',
                      help='Path to the trained model file')
    parser.add_argument('--data-path', type=str, default='Dataset/combined_dataframe_IXIC.csv',
                      help='Path to the data CSV file')
    parser.add_argument('--sequence-length', type=int, default=12,
                      help='Number of time steps to use for prediction')
    parser.add_argument('--target-col', type=str, default='Price',
                      help='Name of the target column')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Prepare data
        X, y, dates, scaler = prepare_data(
            args.data_path, 
            sequence_length=args.sequence_length,
            target_col=args.target_col
        )
        
        # Load model
        model = load_model(
            args.model_path,
            input_dim=X.shape[2],  # Number of features
            predict_steps=1,       # Predict one step ahead
            device=device
        )
        
        # Make predictions
        print("Making predictions...")
        predictions = predict(model, X, device)
        
        # Reshape predictions if needed
        if predictions.ndim > 2:
            predictions = predictions.reshape(-1, predictions.shape[-1])
        
        # Inverse transform predictions
        # Create a dummy array with the same shape as the input data
        dummy_array = np.zeros((len(predictions), X.shape[2]))
        # Put predictions in the right column
        dummy_array[:, 0] = predictions.reshape(-1)
        # Inverse transform
        dummy_array = scaler.inverse_transform(dummy_array)
        # Extract the predictions
        pred_prices = dummy_array[:, 0]
        
        # Inverse transform actual values
        dummy_array = np.zeros((len(y), X.shape[2]))
        dummy_array[:, 0] = y.reshape(-1)
        dummy_array = scaler.inverse_transform(dummy_array)
        actual_prices = dummy_array[:, 0]
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(actual_prices, pred_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, pred_prices))
        r2 = r2_score(actual_prices, pred_prices)
        
        print(f"\nModel Performance:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Plot results
        plot_predictions(dates, actual_prices, pred_prices, 
                        title=f'Stock Price Prediction\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}')
        
        # Save predictions to CSV
        results_df = pd.DataFrame({
            'Date': dates,
            'Actual': actual_prices,
            'Predicted': pred_prices,
            'Error': actual_prices - pred_prices
        })
        
        os.makedirs('results', exist_ok=True)
        results_path = os.path.join('results', 'predictions.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nPredictions saved to {results_path}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
