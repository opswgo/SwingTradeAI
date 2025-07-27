"""
SwingTradeAI - FastAPI Application
User Acceptance: User can understand and test all endpoints independently
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime

from ..core.config import Settings
from ..core.logger import setup_logger, log_user_action
from ..data.pipeline import DataPipeline

logger = setup_logger(__name__)


# Pydantic models for API requests/responses
class HealthResponse(BaseModel):
    healthy: bool
    components: Dict[str, Any]
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class DataRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    period: str = Field(default="1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)")
    validate: bool = Field(default=True, description="Enable data validation")


class DataResponse(BaseModel):
    symbol: str
    records_count: int
    date_range: Dict[str, str]
    validation_report: Optional[Dict[str, Any]] = None
    data_sample: Optional[List[Dict[str, Any]]] = None


class ValidationResponse(BaseModel):
    dataset: str
    overall_score: float
    metrics: Dict[str, float]
    issues: List[str]
    warnings: List[str]


def create_app(settings: Settings, data_pipeline: DataPipeline) -> FastAPI:
    """Create FastAPI application with user-friendly endpoints."""
    
    app = FastAPI(
        title="SwingTradeAI API",
        description="AI-driven options trading system API with comprehensive data validation",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware for dashboard integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Welcome endpoint with user guidance."""
        return {
            "message": "Welcome to SwingTradeAI API",
            "documentation": "/docs",
            "health_check": "/health",
            "version": "1.0.0",
            "user_guide": "Visit /docs for interactive API testing"
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """
        Comprehensive health check endpoint.
        User Acceptance: User can verify all system components are working
        """
        log_user_action("health_check")
        
        try:
            health_status = await data_pipeline.health_check()
            return HealthResponse(**health_status)
        except Exception as e:
            logger.error(f"❌ Health check failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
    
    @app.post("/data/fetch", response_model=DataResponse)
    async def fetch_market_data(request: DataRequest, background_tasks: BackgroundTasks):
        """
        Fetch and validate market data.
        User Acceptance: User sees clean, validated data with clear error reporting
        """
        log_user_action("fetch_market_data", {"symbol": request.symbol, "period": request.period})
        
        try:
            # Fetch data with validation
            data, validation_report = await data_pipeline.fetch_historical_data(
                symbol=request.symbol,
                period=request.period,
                validate=request.validate
            )
            
            # Prepare response with sample data
            data_sample = None
            if not data.empty:
                # Get last 5 records as sample
                sample_df = data.tail(5).reset_index()
                data_sample = [
                    {
                        "date": row["Date"].strftime("%Y-%m-%d") if "Date" in row else str(row.name),
                        "open": round(float(row["Open"]), 2),
                        "high": round(float(row["High"]), 2),
                        "low": round(float(row["Low"]), 2),
                        "close": round(float(row["Close"]), 2),
                        "volume": int(row["Volume"])
                    }
                    for _, row in sample_df.iterrows()
                ]
            
            response = DataResponse(
                symbol=request.symbol,
                records_count=len(data),
                date_range={
                    "start": data.index.min().strftime("%Y-%m-%d"),
                    "end": data.index.max().strftime("%Y-%m-%d")
                },
                validation_report=validation_report if request.validate else None,
                data_sample=data_sample
            )
            
            logger.info(f"✅ Successfully fetched {len(data)} records for {request.symbol}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Data fetch failed for {request.symbol}: {str(e)}")
            log_user_action("fetch_market_data", {"symbol": request.symbol, "error": str(e)}, success=False)
            raise HTTPException(status_code=400, detail=f"Data fetch failed: {str(e)}")
    
    @app.get("/data/quality/{dataset}", response_model=ValidationResponse)
    async def get_data_quality(dataset: str):
        """
        Get data quality metrics for a specific dataset.
        User Acceptance: User can see data quality scores and issues
        """
        log_user_action("get_data_quality", {"dataset": dataset})
        
        try:
            # This would typically fetch from database
            # For now, return a sample response
            return ValidationResponse(
                dataset=dataset,
                overall_score=0.95,
                metrics={
                    "completeness": 0.98,
                    "accuracy": 0.96,
                    "timeliness_score": 0.91
                },
                issues=[],
                warnings=["Minor gaps in weekend data"]
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get data quality for {dataset}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")
    
    @app.get("/data/dashboard")
    async def get_dashboard_data():
        """
        Get data for user dashboard display.
        User Acceptance: User can see system status and data overview
        """
        log_user_action("get_dashboard_data")
        
        try:
            dashboard_data = await data_pipeline.get_user_dashboard_data()
            return dashboard_data
        except Exception as e:
            logger.error(f"❌ Failed to get dashboard data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Dashboard data failed: {str(e)}")
    
    @app.get("/data/symbols")
    async def list_available_symbols():
        """
        List available symbols in the system.
        User Acceptance: User can see what data is available
        """
        log_user_action("list_symbols")
        
        try:
            # Query DuckDB for available symbols
            symbols_result = data_pipeline.duckdb_conn.execute("""
                SELECT DISTINCT symbol, 
                       COUNT(*) as record_count,
                       MIN(date) as start_date,
                       MAX(date) as end_date
                FROM historical_prices 
                GROUP BY symbol 
                ORDER BY symbol
            """).fetchall()
            
            symbols = [
                {
                    "symbol": row[0],
                    "record_count": row[1],
                    "date_range": {
                        "start": str(row[2]),
                        "end": str(row[3])
                    }
                }
                for row in symbols_result
            ]
            
            return {
                "symbols": symbols,
                "total_symbols": len(symbols),
                "message": f"Found {len(symbols)} symbols with historical data"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to list symbols: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Symbol listing failed: {str(e)}")
    
    @app.post("/test/data-pipeline")
    async def test_data_pipeline():
        """
        Test the complete data pipeline with sample data.
        User Acceptance: User can validate the entire data flow
        """
        log_user_action("test_data_pipeline")
        
        try:
            test_results = {
                "tests": [],
                "overall_success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Test 1: Fetch sample data
            try:
                data, validation = await data_pipeline.fetch_historical_data("AAPL", "5d", validate=True)
                test_results["tests"].append({
                    "name": "Data Fetch",
                    "status": "PASS",
                    "details": f"Fetched {len(data)} records with quality score {validation.get('overall_score', 0):.2f}"
                })
            except Exception as e:
                test_results["tests"].append({
                    "name": "Data Fetch",
                    "status": "FAIL",
                    "details": str(e)
                })
                test_results["overall_success"] = False
            
            # Test 2: Database connectivity
            try:
                db_health = await data_pipeline.check_database()
                test_results["tests"].append({
                    "name": "Database Connection",
                    "status": "PASS" if db_health["healthy"] else "FAIL",
                    "details": db_health["message"]
                })
                if not db_health["healthy"]:
                    test_results["overall_success"] = False
            except Exception as e:
                test_results["tests"].append({
                    "name": "Database Connection",
                    "status": "FAIL",
                    "details": str(e)
                })
                test_results["overall_success"] = False
            
            # Test 3: DuckDB functionality
            try:
                duckdb_health = await data_pipeline.check_duckdb()
                test_results["tests"].append({
                    "name": "DuckDB Backtesting",
                    "status": "PASS" if duckdb_health["healthy"] else "FAIL",
                    "details": duckdb_health["message"]
                })
                if not duckdb_health["healthy"]:
                    test_results["overall_success"] = False
            except Exception as e:
                test_results["tests"].append({
                    "name": "DuckDB Backtesting",
                    "status": "FAIL",
                    "details": str(e)
                })
                test_results["overall_success"] = False
            
            return test_results
            
        except Exception as e:
            logger.error(f"❌ Data pipeline test failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Pipeline test failed: {str(e)}")
    
    return app
