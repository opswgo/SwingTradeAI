import yfinance as yf

# Test AAPL options chain directly
ticker = yf.Ticker("AAPL")
print("Testing AAPL options chain...")

try:
    options_data = ticker.options
    print(f"Options data type: {type(options_data)}")
    print(f"Options data length: {len(options_data) if options_data else 'None/Empty'}")
    print(f"Options data: {options_data}")
    
    if options_data:
        print(f"First expiration: {options_data[0]}")
        
        # Try to get options chain for first expiration
        chain = ticker.option_chain(options_data[0])
        print(f"Calls shape: {chain.calls.shape}")
        print(f"Puts shape: {chain.puts.shape}")
        print("Options chain fetch successful!")
    else:
        print("No options data returned")
        
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
