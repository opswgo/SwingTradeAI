import yfinance as yf
import pandas as pd

data = yf.download("AAPL", start="2025-07-21", end="2025-07-25")
print(data)

print ('------ now stooq ------')

url = "https://stooq.com/q/d/l/?s=AAPL.US&i=d"
data = pd.read_csv(url)
print(data.tail())