## per Grok, these are the correct formulas for Moms


import pandas as pd

def calculate_momentum(df):
    """
    Calculate momentum columns for the SAMBA AI dataset.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with 'Price' column containing daily closing prices.
    
    Returns:
    pandas.DataFrame: DataFrame with added 'mom', 'mom1', 'mom2', 'mom3' columns.
    Note: 'mom' is computed as the raw 1-day return ratio; actual dataset values may include
    additional preprocessing (e.g., normalization, scaling) per Hoseinzade and Haratizadeh (2019).
    """
    # Ensure 'Price' column exists (dataset uses 'Price', not 'Close')
    if 'Price' not in df.columns:
        raise ValueError("DataFrame must contain a 'Price' column with closing prices.")
    
    # Calculate 1-day return ratio: (Price_t - Price_{t-1}) / Price_{t-1}
    df['mom'] = (df['Price'] - df['Price'].shift(1)) / df['Price'].shift(1)
    
    # Calculate lagged momentum columns
    df['mom1'] = df['mom'].shift(1)
    df['mom2'] = df['mom'].shift(2)
    df['mom3'] = df['mom'].shift(3)
    
    return df

# Example usage with sample DJIA data
data = {
    'Date': ['2010-01-04', '2010-01-05', '2010-01-06', '2010-01-07', '2010-01-08'],
    'Price': [10583.96, 10572.02, 10573.68, 10606.86, 10618.19]
}
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

# Calculate momentum columns
df = calculate_momentum(df)

# Display results
print(df)