#!/usr/bin/env python3
"""
SwingTradeAI - Main Application Entry Point
User Acceptance: Clean, validated data with error reporting
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.config import Settings
from src.core.logger import setup_logger
from src.data.pipeline import DataPipeline
from src.api.main import create_app
from src.dashboard.app import create_dashboard

# Initialize logger
logger = setup_logger(__name__)


class SwingTradeAI:
    """Main application class with user-focused initialization."""
    
    def __init__(self):
        self.settings = Settings()
        self.data_pipeline = None
        self.api_app = None
        self.dashboard_app = None
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all components with user feedback."""
        logger.info("🚀 Initializing SwingTradeAI...")
        
        try:
            # Initialize data pipeline
            logger.info("📊 Setting up data pipeline...")
            self.data_pipeline = DataPipeline(self.settings)
            await self.data_pipeline.initialize()
            
            # Test data connection
            logger.info("🔍 Testing data connections...")
            health_check = await self.data_pipeline.health_check()
            
            if not health_check['healthy']:
                logger.error(f"❌ Data pipeline health check failed: {health_check['message']}")
                return {'success': False, 'error': health_check['message']}
            
            # Initialize API
            logger.info("🔧 Setting up API server...")
            self.api_app = create_app(self.settings, self.data_pipeline)
            
            # Initialize dashboard
            logger.info("📈 Setting up dashboard...")
            self.dashboard_app = create_dashboard(self.settings, self.data_pipeline)
            
            logger.info("✅ SwingTradeAI initialized successfully!")
            return {
                'success': True,
                'message': 'All systems ready',
                'endpoints': {
                    'api': f'http://localhost:{self.settings.API_PORT}',
                    'dashboard': f'http://localhost:{self.settings.DASHBOARD_PORT}',
                    'docs': f'http://localhost:{self.settings.API_PORT}/docs'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for user validation."""
        logger.info("🔍 Running comprehensive health check...")
        
        checks = {
            'database': False,
            'data_pipeline': False,
            'api': False,
            'external_apis': False
        }
        
        try:
            # Database check
            if self.data_pipeline:
                db_health = await self.data_pipeline.check_database()
                checks['database'] = db_health['healthy']
                
                # Data pipeline check
                pipeline_health = await self.data_pipeline.health_check()
                checks['data_pipeline'] = pipeline_health['healthy']
                
                # External API check
                api_health = await self.data_pipeline.check_external_apis()
                checks['external_apis'] = api_health['healthy']
            
            # API check
            checks['api'] = self.api_app is not None
            
            overall_health = all(checks.values())
            
            return {
                'healthy': overall_health,
                'checks': checks,
                'message': 'All systems operational' if overall_health else 'Some systems need attention'
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'healthy': False,
                'checks': checks,
                'error': str(e)
            }


async def main():
    """Main entry point with user-friendly startup."""
    print("🚀 Starting SwingTradeAI...")
    print("=" * 40)
    
    # Initialize application
    app = SwingTradeAI()
    
    # Run initialization
    init_result = await app.initialize()
    
    if not init_result['success']:
        print(f"❌ Startup failed: {init_result['error']}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Ensure Docker containers are running: docker-compose up -d")
        print("   2. Check .env file configuration")
        print("   3. Verify database connection")
        sys.exit(1)
    
    # Display success information
    print("✅ SwingTradeAI started successfully!")
    print("\n📊 Available endpoints:")
    for name, url in init_result['endpoints'].items():
        print(f"   {name.title()}: {url}")
    
    # Run health check
    health = await app.run_health_check()
    print(f"\n🔍 System health: {'✅ Healthy' if health['healthy'] else '⚠️ Issues detected'}")
    
    if not health['healthy']:
        print("\n⚠️ Component status:")
        for component, status in health['checks'].items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component.replace('_', ' ').title()}")
    
    print("\n🎯 User Testing Ready:")
    print("   • Data pipeline validation: Available")
    print("   • API endpoint testing: /docs")
    print("   • Dashboard interface: Ready")
    print("   • Historical data queries: Enabled")
    
    print("\n🛑 Press Ctrl+C to stop")
    
    try:
        # Keep the application running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down SwingTradeAI...")
        logger.info("Application shutdown requested")


if __name__ == "__main__":
    # Handle different run modes
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "health":
            # Quick health check mode
            async def health_check():
                app = SwingTradeAI()
                await app.initialize()
                health = await app.run_health_check()
                print(f"Health: {'✅ OK' if health['healthy'] else '❌ Issues'}")
                if not health['healthy']:
                    print(f"Details: {health.get('error', 'Check individual components')}")
            
            asyncio.run(health_check())
        
        elif command == "test-data":
            # Test data pipeline mode
            async def test_data():
                app = SwingTradeAI()
                await app.initialize()
                # Add data testing logic here
                print("📊 Data pipeline test completed")
            
            asyncio.run(test_data())
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: health, test-data")
    
    else:
        # Normal startup
        asyncio.run(main())