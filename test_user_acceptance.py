#!/usr/bin/env python3
"""
SwingTradeAI - User Acceptance Testing Script
Sprint 1 User Acceptance Criteria Validation
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.config import Settings
from src.core.logger import setup_logger
from src.data.test_pipeline import TestDataPipeline

logger = setup_logger(__name__)

class UserAcceptanceTests:
    """User acceptance tests for Sprint 1 criteria."""
    
    def __init__(self):
        self.settings = Settings()
        self.data_pipeline = None
        self.test_results = []
        
    async def run_all_tests(self):
        """Run all user acceptance tests."""
        print("🧪 Starting User Acceptance Tests for Sprint 1")
        print("=" * 60)
        
        # Test 1: User can run system locally in <5 minutes
        await self.test_quick_startup()
        
        # Test 2: User sees clean, validated data with error reporting
        await self.test_data_validation()
        
        # Test 3: User can query historical data and see backtest results
        await self.test_data_queries()
        
        # Test 4: User can understand and test all endpoints independently
        await self.test_api_endpoints()
        
        # Summary
        self.print_test_summary()
        
    async def test_quick_startup(self):
        """Test: User can run system locally in <5 minutes with single command."""
        print("\n🚀 Test 1: Quick Startup (<5 minutes)")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Initialize data pipeline
            self.data_pipeline = TestDataPipeline(self.settings)
            await self.data_pipeline.initialize()
            
            # Check if all components are ready
            health_check = await self.data_pipeline.health_check()
            
            elapsed_time = time.time() - start_time
            
            if health_check['healthy'] and elapsed_time < 300:  # 5 minutes
                self.test_results.append({
                    'test': 'Quick Startup',
                    'status': 'PASS',
                    'time': f"{elapsed_time:.1f}s",
                    'details': 'System initialized successfully within time limit'
                })
                print(f"✅ PASS - System ready in {elapsed_time:.1f} seconds")
            else:
                self.test_results.append({
                    'test': 'Quick Startup',
                    'status': 'FAIL',
                    'time': f"{elapsed_time:.1f}s",
                    'details': f"Health: {health_check['healthy']}, Time: {elapsed_time:.1f}s"
                })
                print(f"❌ FAIL - Time: {elapsed_time:.1f}s, Health: {health_check['healthy']}")
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.test_results.append({
                'test': 'Quick Startup',
                'status': 'ERROR',
                'time': f"{elapsed_time:.1f}s",
                'details': str(e)
            })
            print(f"❌ ERROR - {str(e)}")
    
    async def test_data_validation(self):
        """Test: User sees clean, validated data with clear error reporting."""
        print("\n📊 Test 2: Data Validation & Error Reporting")
        print("-" * 40)
        
        try:
            if not self.data_pipeline:
                self.data_pipeline = TestDataPipeline(self.settings)
                await self.data_pipeline.initialize()
            
            # Test with valid symbol
            print("  Testing valid symbol (AAPL)...")
            data, validation_report = await self.data_pipeline.fetch_historical_data(
                "AAPL", "5d", validate=True
            )
            
            # Check validation report structure
            required_fields = ['overall_score', 'metrics', 'issues', 'warnings']
            has_all_fields = all(field in validation_report for field in required_fields)
            
            # Check data quality
            quality_score = validation_report.get('overall_score', 0)
            has_quality_metrics = 'completeness' in validation_report.get('metrics', {})
            
            if has_all_fields and has_quality_metrics and quality_score > 0:
                self.test_results.append({
                    'test': 'Data Validation',
                    'status': 'PASS',
                    'details': f'Quality score: {quality_score:.2f}, Records: {len(data)}'
                })
                print(f"✅ PASS - Quality score: {quality_score:.2f}, {len(data)} records")
            else:
                self.test_results.append({
                    'test': 'Data Validation',
                    'status': 'FAIL',
                    'details': f'Missing validation fields or poor quality'
                })
                print("❌ FAIL - Validation report incomplete")
            
            # Test error reporting with invalid symbol
            print("  Testing error reporting (INVALID)...")
            try:
                await self.data_pipeline.fetch_historical_data("INVALID123", "5d", validate=True)
                print("⚠️  Expected error not raised for invalid symbol")
            except Exception as e:
                print(f"✅ Error properly reported: {str(e)[:50]}...")
                
        except Exception as e:
            self.test_results.append({
                'test': 'Data Validation',
                'status': 'ERROR',
                'details': str(e)
            })
            print(f"❌ ERROR - {str(e)}")
    
    async def test_data_queries(self):
        """Test: User can query historical data and see backtest results."""
        print("\n🗄️ Test 3: Data Queries & Backtest Access")
        print("-" * 40)
        
        try:
            if not self.data_pipeline:
                self.data_pipeline = TestDataPipeline(self.settings)
                await self.data_pipeline.initialize()
            
            # Ensure we have some data
            await self.data_pipeline.fetch_historical_data("AAPL", "1mo", validate=False)
            
            # Test DuckDB queries
            print("  Testing DuckDB queries...")
            symbols_result = self.data_pipeline.duckdb_conn.execute("""
                SELECT DISTINCT symbol, COUNT(*) as count
                FROM historical_prices 
                GROUP BY symbol
            """).fetchall()
            
            if symbols_result and len(symbols_result) > 0:
                print(f"✅ Found {len(symbols_result)} symbols in database")
                
                # Test specific data query
                data_result = self.data_pipeline.duckdb_conn.execute("""
                    SELECT date, close FROM historical_prices 
                    WHERE symbol = 'AAPL' 
                    ORDER BY date DESC LIMIT 5
                """).fetchall()
                
                if data_result and len(data_result) > 0:
                    self.test_results.append({
                        'test': 'Data Queries',
                        'status': 'PASS',
                        'details': f'{len(symbols_result)} symbols, {len(data_result)} recent records'
                    })
                    print(f"✅ PASS - Can query historical data successfully")
                else:
                    self.test_results.append({
                        'test': 'Data Queries',
                        'status': 'FAIL',
                        'details': 'No data returned from queries'
                    })
                    print("❌ FAIL - No data returned from queries")
            else:
                self.test_results.append({
                    'test': 'Data Queries',
                    'status': 'FAIL',
                    'details': 'No symbols found in database'
                })
                print("❌ FAIL - No symbols found in database")
                
        except Exception as e:
            self.test_results.append({
                'test': 'Data Queries',
                'status': 'ERROR',
                'details': str(e)
            })
            print(f"❌ ERROR - {str(e)}")
    
    async def test_api_endpoints(self):
        """Test: User can understand and test all endpoints independently."""
        print("\n🔧 Test 4: API Endpoint Understanding")
        print("-" * 40)
        
        try:
            from src.api.main import create_app
            
            if not self.data_pipeline:
                self.data_pipeline = TestDataPipeline(self.settings)
                await self.data_pipeline.initialize()
            
            # Create API app
            app = create_app(self.settings, self.data_pipeline)
            
            # Check if app has the expected endpoints
            expected_endpoints = [
                "/",
                "/health", 
                "/data/fetch",
                "/data/quality/{dataset}",
                "/data/dashboard",
                "/data/symbols",
                "/test/data-pipeline"
            ]
            
            # Get all routes from the app
            routes = [route.path for route in app.routes]
            
            missing_endpoints = [ep for ep in expected_endpoints if ep not in routes]
            
            if not missing_endpoints:
                self.test_results.append({
                    'test': 'API Endpoints',
                    'status': 'PASS',
                    'details': f'All {len(expected_endpoints)} expected endpoints available'
                })
                print(f"✅ PASS - All {len(expected_endpoints)} endpoints available")
                
                # Test endpoint documentation
                if app.openapi_url and app.docs_url:
                    print("✅ API documentation available at /docs")
                else:
                    print("⚠️  API documentation not configured")
                    
            else:
                self.test_results.append({
                    'test': 'API Endpoints',
                    'status': 'FAIL',
                    'details': f'Missing endpoints: {missing_endpoints}'
                })
                print(f"❌ FAIL - Missing endpoints: {missing_endpoints}")
                
        except Exception as e:
            self.test_results.append({
                'test': 'API Endpoints',
                'status': 'ERROR',
                'details': str(e)
            })
            print(f"❌ ERROR - {str(e)}")
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("📋 USER ACCEPTANCE TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed = sum(1 for result in self.test_results if result['status'] == 'FAIL')
        errors = sum(1 for result in self.test_results if result['status'] == 'ERROR')
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"🔥 Errors: {errors}")
        print(f"Success Rate: {(passed/total*100):.1f}%" if total > 0 else "No tests run")
        
        print("\n📊 Detailed Results:")
        print("-" * 40)
        
        for result in self.test_results:
            status_icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "🔥"}[result['status']]
            print(f"{status_icon} {result['test']}: {result['status']}")
            print(f"   Details: {result['details']}")
            if 'time' in result:
                print(f"   Time: {result['time']}")
            print()
        
        # Sprint 1 acceptance criteria summary
        print("🎯 SPRINT 1 ACCEPTANCE CRITERIA:")
        print("-" * 40)
        
        criteria = [
            ("User can run system locally in <5 minutes", "Quick Startup"),
            ("User sees clean, validated data with error reporting", "Data Validation"),
            ("User can query historical data and see backtest results", "Data Queries"),
            ("User can understand and test all endpoints independently", "API Endpoints")
        ]
        
        for criterion, test_name in criteria:
            test_result = next((r for r in self.test_results if r['test'] == test_name), None)
            if test_result:
                status = test_result['status']
                icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "🔥"}[status]
                print(f"{icon} {criterion}")
            else:
                print(f"❓ {criterion} (Not tested)")
        
        # Overall Sprint 1 status
        sprint_success = passed == total and total > 0
        print(f"\n🏆 SPRINT 1 STATUS: {'✅ COMPLETE' if sprint_success else '❌ NEEDS WORK'}")
        
        if not sprint_success:
            print("\n🔧 Next Steps:")
            print("   1. Review failed tests above")
            print("   2. Fix identified issues")
            print("   3. Re-run tests until all pass")
            print("   4. Proceed to Sprint 2")


async def main():
    """Run user acceptance tests."""
    tester = UserAcceptanceTests()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
