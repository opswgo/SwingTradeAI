#!/bin/bash

# SwingTradeAI - Backtesting Runner
# User Acceptance: User can query historical data and see backtest results

set -e

SYMBOL=${1:-AAPL}

echo "📈 Running SwingTradeAI Backtest for $SYMBOL..."

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Set environment variables
export SSL_CERT_FILE=$(python -m certifi)

echo "🔍 Fetching historical data for $SYMBOL..."
echo "⚙️  Running backtest analysis..."

# Create backtest runner if it doesn't exist
if [ ! -f "run_backtest.py" ]; then
    cat > run_backtest.py << 'EOF'
import asyncio
import sys
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from src.core.config import Settings
from src.features.feature_store import FeatureStore
from src.models.regime_model import RegimeModel

async def run_backtest(symbol: str):
    """Run comprehensive backtest for a symbol"""
    print(f"🚀 Starting backtest for {symbol}")
    
    # Initialize components
    settings = Settings()
    feature_store = FeatureStore(settings)
    regime_model = RegimeModel(settings)
    
    await feature_store.initialize()
    
    # Fetch historical data (1 year)
    print("📊 Fetching historical data...")
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1y")
    
    if data.empty:
        print(f"❌ No data available for {symbol}")
        return
    
    print(f"✅ Retrieved {len(data)} days of data")
    
    # Store features
    print("🔧 Processing features...")
    await feature_store.store_features(symbol, data)
    
    # Get regime predictions
    print("🤖 Generating predictions...")
    try:
        prediction = await regime_model.predict(symbol)
        print(f"📈 Current regime: {prediction.get('regime', 'Unknown')}")
        print(f"🎯 Confidence: {prediction.get('confidence', 0):.1%}")
    except Exception as e:
        print(f"⚠️ Prediction error: {str(e)}")
    
    # Calculate basic performance metrics
    print("📊 Calculating performance metrics...")
    
    # Simple buy-and-hold return
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    total_return = (end_price - start_price) / start_price
    
    # Volatility
    daily_returns = data['Close'].pct_change().dropna()
    volatility = daily_returns.std() * (252 ** 0.5)  # Annualized
    
    # Max drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (assuming 5% risk-free rate)
    risk_free_rate = 0.05
    excess_return = daily_returns.mean() * 252 - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0
    
    print("\n" + "="*50)
    print(f"📈 BACKTEST RESULTS FOR {symbol}")
    print("="*50)
    print(f"📅 Period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"💰 Total Return: {total_return:.1%}")
    print(f"📊 Annualized Volatility: {volatility:.1%}")
    print(f"📉 Max Drawdown: {max_drawdown:.1%}")
    print(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"💵 Start Price: ${start_price:.2f}")
    print(f"💵 End Price: ${end_price:.2f}")
    print("="*50)
    
    # Save results
    results = {
        'symbol': symbol,
        'start_date': str(data.index[0].date()),
        'end_date': str(data.index[-1].date()),
        'total_return': f"{total_return:.1%}",
        'volatility': f"{volatility:.1%}",
        'max_drawdown': f"{max_drawdown:.1%}",
        'sharpe_ratio': f"{sharpe_ratio:.2f}",
        'start_price': f"${start_price:.2f}",
        'end_price': f"${end_price:.2f}"
    }
    
    # Save to file
    import json
    with open(f"data/backtests/{symbol}_backtest_{datetime.now().strftime('%Y%m%d')}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to data/backtests/{symbol}_backtest_{datetime.now().strftime('%Y%m%d')}.json")
    print("✅ Backtest completed successfully!")

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    asyncio.run(run_backtest(symbol))
EOF
fi

python run_backtest.py $SYMBOL
