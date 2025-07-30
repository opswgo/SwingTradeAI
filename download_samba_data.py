# FRED Key: 4cbc5a9898053659bc066e551e6b9515

import os
from datetime import datetime
import pandas as pd
import yfinance as yf
from fredapi import Fred

# -----------------------------
# YOUR FREE FRED API KEY
# -----------------------------
FRED_API_KEY = "4cbc5a9898053659bc066e551e6b9515"
fred = Fred(api_key=FRED_API_KEY)

output_dir = "Dataset"
os.makedirs(output_dir, exist_ok=True)

start = "2010-01-01"
end = datetime.today().strftime("%Y-%m-%d")

# SAMBA Label -> (source, ticker)
mappings = {
    # Equities
    "AAPL": ("yahoo", "AAPL"),
    "MSFT": ("yahoo", "MSFT"),
    "AMZN": ("yahoo", "AMZN"),
    "GE":   ("yahoo", "GE"),
    "WFC":  ("yahoo", "WFC"),
    "XOM":  ("yahoo", "XOM"),
    "JNJ":  ("yahoo", "JNJ"),
    "JPM":  ("yahoo", "JPM"),

    # Futures (Yahoo Futures tickers)
    "S&P-F":     ("yahoo", "ES=F"),
    "NASDAQ-F":  ("yahoo", "NQ=F"),
    "RUSSELL-F": ("yahoo", "RTY=F"),
    "WTI-oil":   ("yahoo", "CL=F"),
    "Brent":     ("yahoo", "BZ=F"),
    "Gold-F":    ("yahoo", "GC=F"),
    "silver-F":  ("yahoo", "SI=F"),
    "Copper-F":  ("yahoo", "HG=F"),
    "Wheat-F":   ("yahoo", "ZW=F"),
    "Gas-F":     ("yahoo", "NG=F"),
    "DJI-F":     ("yahoo", "YM=F"),

    # Indexes
    "FTSE-F":    ("yahoo", "^FTSE"),
    "FCHI":      ("yahoo", "^FCHI"),
    "HSI":       ("yahoo", "^HSI"),
    "GSPC":      ("yahoo", "^GSPC"),
    "IXIC":      ("yahoo", "^IXIC"),
    "SSEC":      ("yahoo", "000001.SS"),
    "Nikkei-F":  ("yahoo", "^N225"),
    "KOSPI-F":   ("yahoo", "^KS11"),
    "DAX-F":     ("yahoo", "^GDAXI"),
    "CAC-F":     ("yahoo", "^FCHI"),
    "FTSE":      ("yahoo", "^FTSE"),
    "NYSE":      ("yahoo", "^NYA"),

    # Macro Rates (FRED)
    "DGS10":   ("fred", "DGS10"),     # 10-Yr Treasury Yield
    "DGS5":    ("fred", "DGS5"),      # 5-Yr Treasury Yield
    "DAAA":    ("fred", "DAAA"),      # AAA Corporate Bond Yield
    "DBAA":    ("fred", "DBAA"),      # BAA Corporate Bond Yield
    "DTB3":    ("fred", "DTB3"),      # 3-Month Treasury Bill Rate
    "CTB3M":   ("fred", "DTB3M"),     # 3-Month Treasury Bill Secondary Market Rate
    "CTB6M":   ("fred", "DTB6M"),     # 6-Month Treasury Bill Secondary Market Rate
    "CTB1Y":   ("fred", "DTB1YR"),    # 1-Year Treasury Bill Secondary Market Rate
    "DTB4WK":  ("fred", "DTB4WK"),    # 4-Week Treasury Bill Secondary Market Rate
    "DTB6":    ("fred", "DTB6"),      # 6-Month Treasury Bill Rate (same as CTB6M?)

    # Currencies (Yahoo FX)
    "AUD":  ("yahoo", "AUDUSD=X"),
    "NZD":  ("yahoo", "NZDUSD=X"),
    "CAD":  ("yahoo", "CADUSD=X"),
    "EUR":  ("yahoo", "EURUSD=X"),
    "GBP":  ("yahoo", "GBPUSD=X"),
    "JPY":  ("yahoo", "JPYUSD=X"),
    "CHF":  ("yahoo", "CHFUSD=X"),
    "CNY":  ("yahoo", "CNYUSD=X"),

    # Dollar Index (Yahoo)
    "Dollar Index-F": ("yahoo", "DX-Y.NYB"),
    "Dollar Index":   ("yahoo", "DX-Y.NYB"),

    # Commodities
    "XAU":  ("yahoo", "GC=F"),    # Gold futures (same as Gold-F)
    "XAG":  ("yahoo", "SI=F"),    # Silver futures (same as silver-F)
}

# List tickers that are in SAMBA but no free source found (no proxies allowed)
missing_tickers = []

def fetch_yahoo(label, ticker):
    print(f"Downloading Yahoo {label} ({ticker})")
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        print(f"[WARN] No data for {label} ({ticker})")
        return None
    df = df.reset_index()
    # For SAMBA, keep only Date, Close as Price, Volume as Vol.
    df = df[["Date", "Close", "Volume"]]
    df.rename(columns={"Close": "Price", "Volume": "Vol."}, inplace=True)
    return df

def fetch_fred(label, ticker):
    print(f"Downloading FRED {label} ({ticker})")
    series = fred.get_series(ticker, observation_start=start, observation_end=end)
    if series.empty:
        print(f"[WARN] No data for {label} ({ticker})")
        return None
    df = series.to_frame(name="Value").reset_index()
    df.rename(columns={"index": "Date"}, inplace=True)
    return df

for label, (source, ticker) in mappings.items():
    try:
        if source == "yahoo":
            df = fetch_yahoo(label, ticker)
        elif source == "fred":
            df = fetch_fred(label, ticker)
        else:
            print(f"[ERROR] Unknown source for {label}: {source}")
            continue

        if df is not None:
            df.to_csv(os.path.join(output_dir, f"{label}.csv"), index=False)
            print(f"Saved {label}.csv")
        else:
            missing_tickers.append(label)
    except Exception as e:
        print(f"[ERROR] Failed to fetch {label}: {e}")
        missing_tickers.append(label)

if missing_tickers:
    print("\n--- Missing Data (no proxies, not available) ---")
    for miss in missing_tickers:
        print(f" * {miss}")
else:
    print("\n✅ All data fetched successfully!")