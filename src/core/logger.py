"""
SwingTradeAI - Logging Configuration
User Acceptance: Clear, structured logging with user-friendly messages
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger
from datetime import datetime

def setup_logger(name: str, log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup structured logging with user-friendly formatting.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level
        log_file: Optional log file path
    
    Returns:
        Configured logger instance
    """
    
    # Remove default loguru handler
    loguru_logger.remove()
    
    # Console handler with user-friendly format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    loguru_logger.add(
        sys.stdout,
        format=console_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        loguru_logger.add(
            log_file,
            format=file_format,
            level=log_level,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=True
        )
    
    # Create a standard logger that forwards to loguru
    class LoguruHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Setup standard logger
    std_logger = logging.getLogger(name)
    std_logger.handlers.clear()
    std_logger.addHandler(LoguruHandler())
    std_logger.setLevel(getattr(logging, log_level))
    
    return std_logger


def log_user_action(action: str, details: dict = None, success: bool = True):
    """Log user actions with structured format for analytics."""
    timestamp = datetime.now().isoformat()
    status = "SUCCESS" if success else "FAILED"
    
    log_entry = {
        "timestamp": timestamp,
        "action": action,
        "status": status,
        "details": details or {}
    }
    
    if success:
        loguru_logger.info(f"🎯 USER_ACTION: {action} - {status}", extra=log_entry)
    else:
        loguru_logger.warning(f"⚠️ USER_ACTION: {action} - {status}", extra=log_entry)


def log_performance_metric(metric_name: str, value: float, unit: str = ""):
    """Log performance metrics for user monitoring."""
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "metric": metric_name,
        "value": value,
        "unit": unit
    }
    
    loguru_logger.info(f"📊 PERFORMANCE: {metric_name} = {value}{unit}", extra=log_entry)


def log_data_quality(dataset: str, quality_score: float, issues: list = None):
    """Log data quality metrics for user validation."""
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "dataset": dataset,
        "quality_score": quality_score,
        "issues": issues or []
    }
    
    if quality_score >= 0.9:
        loguru_logger.info(f"✅ DATA_QUALITY: {dataset} - Score: {quality_score:.2f}", extra=log_entry)
    elif quality_score >= 0.7:
        loguru_logger.warning(f"⚠️ DATA_QUALITY: {dataset} - Score: {quality_score:.2f}", extra=log_entry)
    else:
        loguru_logger.error(f"❌ DATA_QUALITY: {dataset} - Score: {quality_score:.2f}", extra=log_entry)


def log_system_health(component: str, healthy: bool, details: dict = None):
    """Log system health checks for user monitoring."""
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "component": component,
        "healthy": healthy,
        "details": details or {}
    }
    
    if healthy:
        loguru_logger.info(f"✅ HEALTH: {component} - OK", extra=log_entry)
    else:
        loguru_logger.error(f"❌ HEALTH: {component} - FAILED", extra=log_entry)


# Pre-configured loggers for different components
def get_data_logger():
    """Get logger for data pipeline operations."""
    return setup_logger("swingtradeai.data", log_file="logs/data.log")

def get_api_logger():
    """Get logger for API operations."""
    return setup_logger("swingtradeai.api", log_file="logs/api.log")

def get_model_logger():
    """Get logger for ML model operations."""
    return setup_logger("swingtradeai.models", log_file="logs/models.log")

def get_backtest_logger():
    """Get logger for backtesting operations."""
    return setup_logger("swingtradeai.backtest", log_file="logs/backtest.log")
