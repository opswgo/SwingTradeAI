import yfinance as yf
from datetime import datetime
import asyncio

async def test_options_processing():
    ticker = yf.Ticker("AAPL")
    print("Testing detailed AAPL options processing...")
    
    try:
        # Get options data
        options_data = ticker.options
        print(f"Available expirations: {len(options_data)}")
        
        # Test first expiration
        exp_date = options_data[0]
        print(f"Testing expiration: {exp_date}")
        
        # Get options chain
        opt_chain = ticker.option_chain(exp_date)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        print(f"Calls shape: {calls.shape}")
        print(f"Puts shape: {puts.shape}")
        print(f"Calls empty: {calls.empty}")
        print(f"Puts empty: {puts.empty}")
        
        if not calls.empty:
            print("Sample call data:")
            print(calls.head(2))
            
        if not puts.empty:
            print("Sample put data:")
            print(puts.head(2))
            
        # Test date parsing
        try:
            exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
            days_to_exp = (exp_datetime - datetime.now()).days
            print(f"Days to expiration: {days_to_exp}")
        except Exception as e:
            print(f"Date parsing error: {e}")
            
        print("Options processing test successful!")
        
    except Exception as e:
        print(f"Error in options processing: {str(e)}")
        import traceback
        traceback.print_exc()

# Run the test
asyncio.run(test_options_processing())
