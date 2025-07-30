import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the trained model
def load_model(model_path):
    """Load the trained SAMBA model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = torch.load(model_path, map_location=device)
    model.eval()  # Set to evaluation mode
    return model

# Prepare data for prediction
def prepare_data(data_path, sequence_length=12, predict=1):
    """
    Prepare data for prediction in the same format as training.
    
    Args:
        data_path: Path to the CSV file with stock data
        sequence_length: Number of time steps to use for prediction
        predict: Number of steps to predict ahead
        
    Returns:
        X: Input sequences
        y: Target values
        dates: Corresponding dates
        scaler: Fitted scaler for inverse transform
    """
    # Load and preprocess data (adjust according to your data format)
    df = pd.read_csv(data_path)
    
    # Ensure the 'Date' column is in datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Use only numeric columns for prediction
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Date' in df.columns:
        dates = df['Date'].values
        df_numeric = df[numeric_cols]
    else:
        dates = np.arange(len(df))
        df_numeric = df[numeric_cols]
    
    # Normalize the data (use the same scaler as in training)
    # For demonstration, we'll use MinMax scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length - predict + 1):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length:i + sequence_length + predict, 0])  # Assuming first column is target
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, dates[sequence_length:len(dates)-predict+1], scaler

def predict(model, X, batch_size=32):
    """Make predictions using the trained model."""
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Create DataLoader
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)

def plot_predictions(dates, actual, predicted, title='Stock Price Prediction'):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(15, 6))
    plt.plot(dates, actual, label='Actual', color='blue')
    plt.plot(dates, predicted, label='Predicted', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Paths
    model_path = "AGCRN_Model/milan/call/1_stemgnn.pt"  # Using the first fold model
    data_path = "Dataset/combined_dataframe_IXIC.csv"  # Update this to your data file
    
    # Parameters (should match training parameters)
    sequence_length = 12  # Number of time steps to use for prediction
    predict_steps = 1     # Number of steps to predict ahead
    
    try:
        # Load the model
        print("Loading model...")
        model = load_model(model_path)
        model.to(device)
        
        # Prepare data
        print("Preparing data...")
        X, y, dates, scaler = prepare_data(
            data_path, 
            sequence_length=sequence_length, 
            predict=predict_steps
        )
        
        # Make predictions
        print("Making predictions...")
        predictions = predict(model, X)
        
        # Reshape for inverse transform
        # Create a dummy array with the same shape as the original data
        dummy = np.zeros((len(predictions), len(scaler.scale_)))
        dummy[:, 0] = predictions.flatten()  # Assuming first column is the target
        
        # Inverse transform the predictions
        predictions_inv = scaler.inverse_transform(dummy)[:, 0]
        
        # Inverse transform the actual values
        actual = np.zeros((len(y), len(scaler.scale_)))
        actual[:, 0] = y.flatten()
        actual_inv = scaler.inverse_transform(actual)[:, 0]
        
        # Plot results
        plot_predictions(dates, actual_inv, predictions_inv, 'Stock Price Prediction')
        
        # Print some metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(actual_inv, predictions_inv)
        rmse = np.sqrt(mean_squared_error(actual_inv, predictions_inv))
        
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"Root Mean Squared Error: ${rmse:.2f}")
        
        # Create a DataFrame with the results
        results = pd.DataFrame({
            'Date': dates,
            'Actual': actual_inv,
            'Predicted': predictions_inv
        })
        
        # Save results to CSV
        results.to_csv('stock_predictions.csv', index=False)
        print("Predictions saved to 'stock_predictions.csv'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
