#!/usr/bin/env python3
"""
SwingTradeAI - Data Validation Dashboard
User Acceptance: User sees clean, validated data with clear error reporting
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import asyncio
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from src.core.config import Settings
from src.features.feature_store import FeatureStore

st.set_page_config(
    page_title="SwingTradeAI - Data Validation",
    page_icon="🔍",
    layout="wide"
)

async def validate_data_quality(symbol: str, days: int = 30):
    """Comprehensive data quality validation"""
    
    validation_results = {
        'symbol': symbol,
        'timestamp': datetime.now(),
        'tests': [],
        'overall_status': 'PASS',
        'data_quality_score': 0
    }
    
    try:
        # Test 1: Data Availability
        st.write("🔍 **Test 1: Data Availability**")
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{days}d")
        
        if data.empty:
            validation_results['tests'].append({
                'name': 'Data Availability',
                'status': 'FAIL',
                'message': f'No data available for {symbol}',
                'severity': 'CRITICAL'
            })
            validation_results['overall_status'] = 'FAIL'
            st.error(f"❌ No data available for {symbol}")
            return validation_results
        else:
            validation_results['tests'].append({
                'name': 'Data Availability',
                'status': 'PASS',
                'message': f'Retrieved {len(data)} days of data',
                'severity': 'INFO'
            })
            st.success(f"✅ Retrieved {len(data)} days of data")
        
        # Test 2: Data Completeness
        st.write("🔍 **Test 2: Data Completeness**")
        missing_data = data.isnull().sum().sum()
        completeness_score = (1 - missing_data / (len(data) * len(data.columns))) * 100
        
        if missing_data > 0:
            validation_results['tests'].append({
                'name': 'Data Completeness',
                'status': 'WARN',
                'message': f'{missing_data} missing values found ({completeness_score:.1f}% complete)',
                'severity': 'WARNING'
            })
            st.warning(f"⚠️ {missing_data} missing values found ({completeness_score:.1f}% complete)")
        else:
            validation_results['tests'].append({
                'name': 'Data Completeness',
                'status': 'PASS',
                'message': '100% data completeness',
                'severity': 'INFO'
            })
            st.success("✅ 100% data completeness")
        
        # Test 3: Price Validation
        st.write("🔍 **Test 3: Price Validation**")
        price_issues = []
        
        # Check for negative prices
        if (data[['Open', 'High', 'Low', 'Close']] < 0).any().any():
            price_issues.append("Negative prices detected")
        
        # Check for zero prices
        if (data[['Open', 'High', 'Low', 'Close']] == 0).any().any():
            price_issues.append("Zero prices detected")
        
        # Check for unrealistic price movements (>50% in one day)
        daily_returns = data['Close'].pct_change().abs()
        extreme_moves = (daily_returns > 0.5).sum()
        if extreme_moves > 0:
            price_issues.append(f"{extreme_moves} extreme price movements (>50%)")
        
        if price_issues:
            validation_results['tests'].append({
                'name': 'Price Validation',
                'status': 'WARN',
                'message': '; '.join(price_issues),
                'severity': 'WARNING'
            })
            st.warning(f"⚠️ Price issues: {'; '.join(price_issues)}")
        else:
            validation_results['tests'].append({
                'name': 'Price Validation',
                'status': 'PASS',
                'message': 'All prices within expected ranges',
                'severity': 'INFO'
            })
            st.success("✅ All prices within expected ranges")
        
        # Test 4: Volume Validation
        st.write("🔍 **Test 4: Volume Validation**")
        zero_volume_days = (data['Volume'] == 0).sum()
        if zero_volume_days > 0:
            validation_results['tests'].append({
                'name': 'Volume Validation',
                'status': 'WARN',
                'message': f'{zero_volume_days} days with zero volume',
                'severity': 'WARNING'
            })
            st.warning(f"⚠️ {zero_volume_days} days with zero volume")
        else:
            validation_results['tests'].append({
                'name': 'Volume Validation',
                'status': 'PASS',
                'message': 'Volume data looks healthy',
                'severity': 'INFO'
            })
            st.success("✅ Volume data looks healthy")
        
        # Test 5: Feature Store Integration
        st.write("🔍 **Test 5: Feature Store Integration**")
        try:
            settings = Settings()
            feature_store = FeatureStore(settings)
            await feature_store.initialize()
            
            # Try to store and retrieve features
            await feature_store.store_features(symbol, data)
            result = await feature_store.retrieve_features(symbol, limit=10)
            
            if result['success']:
                validation_results['tests'].append({
                    'name': 'Feature Store Integration',
                    'status': 'PASS',
                    'message': f'Successfully stored and retrieved {result["features_retrieved"]} features',
                    'severity': 'INFO'
                })
                st.success(f"✅ Feature store integration working ({result['features_retrieved']} features)")
            else:
                validation_results['tests'].append({
                    'name': 'Feature Store Integration',
                    'status': 'FAIL',
                    'message': f'Feature store error: {result.get("error", "Unknown error")}',
                    'severity': 'CRITICAL'
                })
                st.error(f"❌ Feature store error: {result.get('error', 'Unknown error')}")
                validation_results['overall_status'] = 'FAIL'
        
        except Exception as e:
            validation_results['tests'].append({
                'name': 'Feature Store Integration',
                'status': 'FAIL',
                'message': f'Feature store initialization failed: {str(e)}',
                'severity': 'CRITICAL'
            })
            st.error(f"❌ Feature store initialization failed: {str(e)}")
            validation_results['overall_status'] = 'FAIL'
        
        # Calculate overall data quality score
        pass_count = sum(1 for test in validation_results['tests'] if test['status'] == 'PASS')
        total_tests = len(validation_results['tests'])
        validation_results['data_quality_score'] = (pass_count / total_tests) * 100
        
        return validation_results
        
    except Exception as e:
        validation_results['tests'].append({
            'name': 'Data Validation',
            'status': 'FAIL',
            'message': f'Validation failed: {str(e)}',
            'severity': 'CRITICAL'
        })
        validation_results['overall_status'] = 'FAIL'
        st.error(f"❌ Validation failed: {str(e)}")
        return validation_results

def main():
    st.title("🔍 SwingTradeAI - Data Validation Dashboard")
    st.markdown("**User Acceptance**: User sees clean, validated data with clear error reporting")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Validation Controls")
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
    days = st.sidebar.slider("Days of Data", min_value=7, max_value=365, value=30)
    
    if st.sidebar.button("🚀 Run Validation", type="primary"):
        st.header(f"📊 Validating {symbol} ({days} days)")
        
        with st.spinner("Running comprehensive data validation..."):
            # Run async validation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(validate_data_quality(symbol, days))
            loop.close()
        
        # Display results
        st.header("📋 Validation Results")
        
        # Overall status
        if results['overall_status'] == 'PASS':
            st.success(f"🎉 Overall Status: {results['overall_status']}")
        else:
            st.error(f"❌ Overall Status: {results['overall_status']}")
        
        # Data quality score
        score = results['data_quality_score']
        if score >= 80:
            st.metric("📊 Data Quality Score", f"{score:.1f}%", delta="Excellent")
        elif score >= 60:
            st.metric("📊 Data Quality Score", f"{score:.1f}%", delta="Good")
        else:
            st.metric("📊 Data Quality Score", f"{score:.1f}%", delta="Needs Attention")
        
        # Detailed test results
        st.subheader("🔍 Detailed Test Results")
        
        for test in results['tests']:
            if test['status'] == 'PASS':
                st.success(f"✅ **{test['name']}**: {test['message']}")
            elif test['status'] == 'WARN':
                st.warning(f"⚠️ **{test['name']}**: {test['message']}")
            else:
                st.error(f"❌ **{test['name']}**: {test['message']}")
        
        # Save results
        results_file = f"data/validation_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            import json
            os.makedirs("data", exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            st.info(f"💾 Results saved to: {results_file}")
        except Exception as e:
            st.warning(f"⚠️ Could not save results: {str(e)}")
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Instructions")
    st.sidebar.markdown("""
    1. Enter a stock symbol (e.g., AAPL, MSFT)
    2. Select number of days to validate
    3. Click 'Run Validation'
    4. Review detailed results
    5. Check saved validation report
    """)
    
    # Quick actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    if st.sidebar.button("🏥 System Health Check"):
        st.info("Running system health check...")
        # Add system health check logic here
    
    if st.sidebar.button("📊 View Dashboard"):
        st.info("💡 Run: `./run_dashboard.sh` to open main dashboard")

if __name__ == "__main__":
    main()
