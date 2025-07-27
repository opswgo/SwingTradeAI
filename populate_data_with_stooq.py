"""
Populate feature store with real market data from Stooq
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.config import Settings
from src.features.feature_store import FeatureStore
from src.features.technical_analysis import TechnicalAnalysis
from src.features.market_regime import MarketRegimeDetector
from alpha_vantage_fetcher import StooqDataFetcher


async def populate_data_with_stooq():
    """Populate feature store with real market data from Stooq."""
    print("🚀 Populating feature store with real market data from Stooq...")
    
    try:
        # Initialize components
        settings = Settings()
        feature_store = FeatureStore(settings)
        await feature_store.initialize()
        
        technical_analyzer = TechnicalAnalysis()
        regime_detector = MarketRegimeDetector()
        stooq_fetcher = StooqDataFetcher()
        
        symbol = "AAPL"
        
        # Fetch real market data from Stooq
        print(f"📊 Fetching 2 years of real market data for {symbol} from Stooq...")
        historical_data, validation_report = await stooq_fetcher.fetch_historical_data(symbol, period="2y")
        
        print(f"✅ Fetched {len(historical_data)} days of real market data")
        print(f"📅 Date range: {historical_data.index[0].date()} to {historical_data.index[-1].date()}")
        print(f"💰 Price range: ${historical_data['Close'].min():.2f} - ${historical_data['Close'].max():.2f}")
        
        # Calculate technical indicators on real data
        print("🔧 Calculating technical indicators on real market data...")
        enhanced_data, indicator_report = technical_analyzer.calculate_all_indicators(historical_data)
        
        print(f"✅ Calculated {len(indicator_report.get('indicators', []))} technical indicators")
        
        # Add regime detection to real data
        print("🎯 Detecting market regimes on real data...")
        regime_result = await regime_detector.detect_current_regime(enhanced_data)
        enhanced_data['regime'] = regime_result.get('regime', 'bull_trending')
        enhanced_data['regime_confidence'] = regime_result.get('confidence', 0.8)
        
        print(f"📊 Detected regime: {regime_result.get('regime')} (confidence: {regime_result.get('confidence', 0.8):.2f})")
        
        # Store features in feature store
        print("💾 Storing real market features in feature store...")
        store_result = await feature_store.store_features(symbol, enhanced_data)
        
        if store_result.get('success'):
            print(f"✅ Successfully stored {len(enhanced_data)} real market feature records")
            
            # Verify storage and show target class distribution
            retrieve_result = await feature_store.retrieve_features(symbol, limit=1000)
            if retrieve_result.get('success'):
                stored_data = retrieve_result.get('data')
                print(f"✅ Verified: {len(stored_data)} records available in feature store")
                
                # Analyze target class distribution for training
                if 'Close' in stored_data.columns:
                    returns = stored_data['Close'].pct_change(5).shift(-5)  # 5-day forward returns
                    returns_clean = returns.dropna()
                    
                    if len(returns_clean) > 0:
                        print(f"\n📈 Real Market Data Analysis:")
                        print(f"   Return statistics (5-day forward):")
                        print(f"   Mean: {returns_clean.mean():.4f} ({returns_clean.mean()*100:.2f}%)")
                        print(f"   Std: {returns_clean.std():.4f} ({returns_clean.std()*100:.2f}%)")
                        print(f"   Min: {returns_clean.min():.4f} ({returns_clean.min()*100:.2f}%)")
                        print(f"   Max: {returns_clean.max():.4f} ({returns_clean.max()*100:.2f}%)")
                        
                        # Check target class distribution for ML training
                        thresholds = {'strong_buy': 0.05, 'buy': 0.02, 'hold': -0.02}
                        targets = returns_clean.apply(lambda x: 
                            3 if x > thresholds['strong_buy'] else
                            2 if x > thresholds['buy'] else
                            1 if x > thresholds['hold'] else 0
                        )
                        target_counts = targets.value_counts().sort_index()
                        total_targets = len(targets)
                        
                        print(f"\n🎯 Target Class Distribution for ML Training:")
                        class_names = {0: 'Sell', 1: 'Hold', 2: 'Buy', 3: 'Strong Buy'}
                        for class_id, count in target_counts.items():
                            percentage = (count / total_targets) * 100
                            print(f"   Class {class_id} ({class_names[class_id]}): {count} samples ({percentage:.1f}%)")
                        
                        print(f"\n✅ Training Data Quality Assessment:")
                        print(f"   Total samples: {total_targets}")
                        print(f"   Unique classes: {len(target_counts)}")
                        print(f"   Minimum samples per class: {target_counts.min()}")
                        print(f"   Data suitable for ML training: {'✅ YES' if len(target_counts) >= 2 and total_targets >= 100 else '❌ NO'}")
                        
                        if len(target_counts) >= 2 and total_targets >= 100:
                            print(f"\n🎉 SUCCESS: Real market data is ready for ML model training!")
                            print(f"   - {total_targets} training samples available")
                            print(f"   - {len(target_counts)} target classes for classification")
                            print(f"   - Diverse price movements captured")
                            return True
                        else:
                            print(f"\n⚠️ WARNING: Data may be insufficient for robust ML training")
                            return False
            else:
                print("❌ Failed to verify stored features")
                return False
        else:
            print(f"❌ Failed to store features: {store_result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to populate data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(populate_data_with_stooq())
    if success:
        print(f"\n🚀 Ready to proceed with Sprint 3 ML model training!")
    else:
        print(f"\n❌ Data population failed - check errors above")
