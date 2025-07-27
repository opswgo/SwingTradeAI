"""
Script to fetch fresh market data for training
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config import Settings
from src.data.pipeline import DataPipeline
from src.features.feature_store import FeatureStore
from src.features.technical_analysis import TechnicalAnalysis
from src.features.market_regime import MarketRegimeDetector


async def fetch_fresh_data():
    """Fetch fresh market data and store features."""
    print("📊 Fetching fresh market data for training...")
    
    try:
        # Initialize components
        settings = Settings()
        pipeline = DataPipeline(settings)
        await pipeline.initialize()
        
        feature_store = FeatureStore(settings)
        await feature_store.initialize()
        
        technical_analyzer = TechnicalAnalysis()
        regime_detector = MarketRegimeDetector()
        
        symbol = "AAPL"
        
        # Fetch 2 years of historical data
        print(f"🔄 Fetching 2 years of historical data for {symbol}...")
        historical_data, validation_report = await pipeline.fetch_historical_data(symbol, period="2y")
        
        print(f"✅ Fetched {len(historical_data)} days of historical data")
        print(f"📅 Date range: {historical_data.index[0]} to {historical_data.index[-1]}")
        
        # Calculate technical indicators
        print("🔧 Calculating technical indicators...")
        enhanced_data, indicator_report = technical_analyzer.calculate_all_indicators(historical_data)
        
        print(f"✅ Calculated {len(indicator_report.get('indicators', []))} technical indicators")
        
        # Add regime detection
        print("🎯 Detecting market regimes...")
        regime_result = await regime_detector.detect_current_regime(enhanced_data)
        enhanced_data['regime'] = regime_result.get('regime', 'bull_trending')
        enhanced_data['regime_confidence'] = regime_result.get('confidence', 0.8)
        
        # Store features
        print("💾 Storing features in feature store...")
        store_result = await feature_store.store_features(symbol, enhanced_data)
        
        if store_result.get('success'):
            print(f"✅ Successfully stored {len(enhanced_data)} feature records")
            
            # Verify storage
            retrieve_result = await feature_store.retrieve_features(symbol, limit=1000)
            if retrieve_result.get('success'):
                stored_data = retrieve_result.get('data', pd.DataFrame())
                print(f"✅ Verified: {len(stored_data)} records available in feature store")
                
                # Show data summary
                if not stored_data.empty:
                    print(f"📊 Data columns: {list(stored_data.columns)}")
                    print(f"📅 Date range: {stored_data.index[0]} to {stored_data.index[-1]}")
                    
                    if 'Close' in stored_data.columns:
                        returns = stored_data['Close'].pct_change(5).shift(-5)  # 5-day forward returns
                        returns_clean = returns.dropna()
                        if len(returns_clean) > 0:
                            print(f"📈 Return statistics:")
                            print(f"   Mean: {returns_clean.mean():.4f}")
                            print(f"   Std: {returns_clean.std():.4f}")
                            print(f"   Min: {returns_clean.min():.4f}")
                            print(f"   Max: {returns_clean.max():.4f}")
                            
                            # Check target class distribution
                            thresholds = {'strong_buy': 0.05, 'buy': 0.02, 'hold': -0.02}
                            targets = returns_clean.apply(lambda x: 
                                3 if x > thresholds['strong_buy'] else
                                2 if x > thresholds['buy'] else
                                1 if x > thresholds['hold'] else 0
                            )
                            target_counts = targets.value_counts().sort_index()
                            print(f"🎯 Target class distribution: {dict(target_counts)}")
            else:
                print("❌ Failed to verify stored features")
        else:
            print(f"❌ Failed to store features: {store_result.get('error')}")
            
    except Exception as e:
        print(f"❌ Failed to fetch fresh data: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(fetch_fresh_data())
