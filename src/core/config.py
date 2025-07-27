"""
SwingTradeAI - Configuration Management
User Acceptance: Clear configuration with validation and error reporting
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings with user-friendly validation."""
    
    # Application Info
    APP_NAME: str = "SwingTradeAI"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", json_schema_extra={"env": "ENVIRONMENT"})
    
    # Server Configuration
    API_HOST: str = Field(default="0.0.0.0", json_schema_extra={"env": "API_HOST"})
    API_PORT: int = Field(default=8000, json_schema_extra={"env": "API_PORT"})
    DASHBOARD_PORT: int = Field(default=8501, json_schema_extra={"env": "DASHBOARD_PORT"})
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://trader:secure_password_123@localhost:5432/swingtradeai",
        json_schema_extra={"env": "DATABASE_URL"}
    )
    REDIS_URL: str = Field(default="redis://localhost:6379", json_schema_extra={"env": "REDIS_URL"})
    
    # Data Sources
    YFINANCE_ENABLED: bool = Field(default=True, json_schema_extra={"env": "YFINANCE_ENABLED"})
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(default=None, json_schema_extra={"env": "ALPHA_VANTAGE_API_KEY"})
    POLYGON_API_KEY: Optional[str] = Field(default=None, json_schema_extra={"env": "POLYGON_API_KEY"})
    FINNHUB_API_KEY: Optional[str] = Field(default=None, json_schema_extra={"env": "FINNHUB_API_KEY"})
    
    # Trading Configuration
    DEFAULT_PORTFOLIO_SIZE: float = Field(default=100000.0, json_schema_extra={"env": "DEFAULT_PORTFOLIO_SIZE"})
    MAX_POSITION_SIZE: float = Field(default=0.05, json_schema_extra={"env": "MAX_POSITION_SIZE"})
    RISK_FREE_RATE: float = Field(default=0.05, json_schema_extra={"env": "RISK_FREE_RATE"})
    COMMISSION_PER_TRADE: float = Field(default=1.0, json_schema_extra={"env": "COMMISSION_PER_TRADE"})
    
    # Backtesting
    BACKTEST_START_DATE: str = Field(default="2020-01-01", json_schema_extra={"env": "BACKTEST_START_DATE"})
    BACKTEST_END_DATE: str = Field(default="2024-01-01", json_schema_extra={"env": "BACKTEST_END_DATE"})
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", json_schema_extra={"env": "LOG_LEVEL"})
    LOG_FILE: str = Field(default="logs/swingtradeai.log", json_schema_extra={"env": "LOG_FILE"})
    
    # Security
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production", json_schema_extra={"env": "SECRET_KEY"})
    JWT_SECRET: str = Field(default="dev-jwt-secret-change-in-production", json_schema_extra={"env": "JWT_SECRET"})
    
    # Performance
    MAX_WORKERS: int = Field(default=4, json_schema_extra={"env": "MAX_WORKERS"})
    CACHE_TTL: int = Field(default=300, json_schema_extra={"env": "CACHE_TTL"})  # 5 minutes
    
    # Data Directories
    DATA_DIR: Path = Field(default=Path("data"))
    RAW_DATA_DIR: Path = Field(default=Path("data/raw"))
    PROCESSED_DATA_DIR: Path = Field(default=Path("data/processed"))
    MODELS_DIR: Path = Field(default=Path("data/models"))
    BACKTESTS_DIR: Path = Field(default=Path("data/backtests"))
    
    @field_validator('MAX_POSITION_SIZE')
    @classmethod
    def validate_position_size(cls, v):
        if not 0 < v <= 1:
            raise ValueError('MAX_POSITION_SIZE must be between 0 and 1')
        return v
    
    @field_validator('RISK_FREE_RATE')
    @classmethod
    def validate_risk_free_rate(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('RISK_FREE_RATE must be between 0 and 1')
        return v
    
    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of {valid_levels}')
        return v.upper()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories with user feedback."""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODELS_DIR,
            self.BACKTESTS_DIR,
            Path("logs")
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_database_config(self) -> dict:
        """Get database configuration for user validation."""
        return {
            'url': self.DATABASE_URL,
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'pool_recycle': 3600
        }
    
    def get_api_keys_status(self) -> dict:
        """Check API keys status for user feedback."""
        return {
            'alpha_vantage': bool(self.ALPHA_VANTAGE_API_KEY),
            'polygon': bool(self.POLYGON_API_KEY),
            'finnhub': bool(self.FINNHUB_API_KEY),
            'yfinance': self.YFINANCE_ENABLED
        }
    
    def validate_configuration(self) -> dict:
        """Comprehensive configuration validation for users."""
        issues = []
        warnings = []
        
        # Check critical settings
        if self.ENVIRONMENT == "production":
            if self.SECRET_KEY == "dev-secret-key-change-in-production":
                issues.append("SECRET_KEY must be changed in production")
            if self.JWT_SECRET == "dev-jwt-secret-change-in-production":
                issues.append("JWT_SECRET must be changed in production")
        
        # Check API keys
        api_keys = self.get_api_keys_status()
        if not any(api_keys.values()):
            warnings.append("No external API keys configured - only yfinance data available")
        
        # Check directories
        if not self.DATA_DIR.exists():
            issues.append(f"Data directory does not exist: {self.DATA_DIR}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'api_keys': api_keys
        }
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True
    )
