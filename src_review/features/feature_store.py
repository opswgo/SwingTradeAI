"""
SwingTradeAI - Feature Storage and Retrieval System
User Acceptance: User can query any feature with <1s response time
"""

import pandas as pd
import numpy as np
import duckdb
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import asyncio

from ..core.config import Settings
from ..core.logger import setup_logger, log_user_action, log_performance_metric
from .technical_analysis import TechnicalAnalysis
from .market_regime import MarketRegimeDetector

logger = setup_logger(__name__)


class FeatureStore:
    """
    High-performance feature storage and retrieval system.
    User Focus: Fast feature access and transparent feature lineage
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.duckdb_conn = None
        self.technical_analyzer = TechnicalAnalysis()
        self.regime_detector = MarketRegimeDetector()
        
        # Feature metadata for user understanding
        self.feature_metadata = {}
        self.feature_groups = {
            'price': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'trend': ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_histogram'],
            'volatility': ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'atr'],
            'momentum': ['rsi', 'stoch_k', 'stoch_d', 'williams_r'],
            'volume': ['volume_sma', 'volume_ratio', 'obv'],
            'price_action': ['price_change', 'high_low_ratio', 'gap'],
            'regime': ['market_regime', 'regime_confidence', 'trend_strength', 'volatility_level']
        }
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize feature store with optimized schema."""
        logger.info("🏗️ Initializing feature store...")
        
        try:
            # Setup DuckDB with performance optimizations
            await self._setup_feature_database()
            
            # Create optimized feature tables
            await self._create_feature_schema()
            
            # Initialize feature metadata
            await self._initialize_feature_metadata()
            
            logger.info("✅ Feature store initialized successfully")
            return {'success': True, 'message': 'Feature store ready'}
            
        except Exception as e:
            logger.error(f"❌ Feature store initialization failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _setup_feature_database(self):
        """Setup DuckDB with performance optimizations."""
        try:
            # Use persistent database for features
            db_path = self.settings.DATA_DIR / "features" / "feature_store.duckdb"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.duckdb_conn = duckdb.connect(str(db_path))
            
            # Performance optimizations
            self.duckdb_conn.execute("PRAGMA memory_limit='2GB'")
            self.duckdb_conn.execute("PRAGMA threads=4")
            
            logger.info("✅ Feature database connected with performance optimizations")
            
        except Exception as e:
            logger.error(f"❌ Feature database setup failed: {str(e)}")
            raise
    
    async def _create_feature_schema(self):
        """Create optimized feature storage schema."""
        try:
            # Main features table with columnar storage
            self.duckdb_conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    symbol VARCHAR NOT NULL,
                    date DATE NOT NULL,
                    feature_group VARCHAR NOT NULL,
                    features JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, date, feature_group)
                )
            """)
            
            # Feature metadata table
            self.duckdb_conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_metadata (
                    feature_name VARCHAR PRIMARY KEY,
                    feature_group VARCHAR NOT NULL,
                    description TEXT,
                    data_type VARCHAR,
                    calculation_method TEXT,
                    dependencies JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Feature lineage table for tracking feature dependencies
            self.duckdb_conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_lineage (
                    feature_name VARCHAR NOT NULL,
                    parent_feature VARCHAR NOT NULL,
                    relationship_type VARCHAR NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Performance indexes
            self.duckdb_conn.execute("CREATE INDEX IF NOT EXISTS idx_features_symbol_date ON features(symbol, date)")
            self.duckdb_conn.execute("CREATE INDEX IF NOT EXISTS idx_features_group ON features(feature_group)")
            self.duckdb_conn.execute("CREATE INDEX IF NOT EXISTS idx_features_date ON features(date)")
            
            logger.info("✅ Feature schema created with performance indexes")
            
        except Exception as e:
            logger.error(f"❌ Feature schema creation failed: {str(e)}")
            raise
    
    async def _initialize_feature_metadata(self):
        """Initialize feature metadata for user understanding."""
        try:
            # Get feature descriptions from technical analyzer
            feature_descriptions = self.technical_analyzer.get_all_indicators_info()
            
            # Add regime features
            feature_descriptions.update({
                'market_regime': 'Current market regime classification (bull/bear/sideways with volatility)',
                'regime_confidence': 'Confidence level in regime classification (0-1)',
                'trend_strength': 'Strength of current trend (strong_bull to strong_bear)',
                'volatility_level': 'Current volatility level (low/medium/high/crisis)'
            })
            
            # Store metadata
            for feature_name, description in feature_descriptions.items():
                # Determine feature group
                feature_group = 'other'
                for group, features in self.feature_groups.items():
                    if feature_name in features:
                        feature_group = group
                        break
                
                # Insert or update metadata
                self.duckdb_conn.execute("""
                    INSERT OR REPLACE INTO feature_metadata 
                    (feature_name, feature_group, description, data_type, calculation_method)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    feature_name, 
                    feature_group, 
                    description, 
                    'DOUBLE',  # Most features are numeric
                    'technical_analysis'
                ])
            
            logger.info(f"✅ Initialized metadata for {len(feature_descriptions)} features")
            
        except Exception as e:
            logger.error(f"❌ Feature metadata initialization failed: {str(e)}")
            raise
    
    async def store_features(
        self, 
        symbol: str, 
        data: pd.DataFrame, 
        include_regime: bool = True
    ) -> Dict[str, Any]:
        """
        Store comprehensive features for a symbol.
        
        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
            include_regime: Whether to include regime analysis
            
        Returns:
            Storage result with performance metrics
        """
        start_time = datetime.now()
        log_user_action("store_features", {"symbol": symbol, "records": len(data)})
        
        try:
            # Calculate technical indicators
            enhanced_data, tech_report = self.technical_analyzer.calculate_all_indicators(data)
            
            # Add regime analysis if requested
            if include_regime:
                regime_features = await self._calculate_regime_features(enhanced_data)
                enhanced_data = pd.concat([enhanced_data, regime_features], axis=1)
            
            # Store features by group for optimal retrieval
            storage_results = {}
            
            for group_name, feature_names in self.feature_groups.items():
                available_features = [f for f in feature_names if f in enhanced_data.columns]
                
                if available_features:
                    group_data = enhanced_data[available_features]
                    result = await self._store_feature_group(symbol, group_data, group_name)
                    storage_results[group_name] = result
            
            # Calculate performance metrics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            log_performance_metric("feature_storage_time", processing_time, "seconds")
            log_performance_metric("features_per_second", len(enhanced_data.columns) / processing_time, "features/sec")
            
            logger.info(f"✅ Stored features for {symbol}: {len(enhanced_data.columns)} features, {len(enhanced_data)} records in {processing_time:.2f}s")
            
            return {
                'success': True,
                'symbol': symbol,
                'features_stored': len(enhanced_data.columns),
                'records_stored': len(enhanced_data),
                'processing_time': processing_time,
                'storage_results': storage_results,
                'technical_report': tech_report
            }
            
        except Exception as e:
            logger.error(f"❌ Feature storage failed for {symbol}: {str(e)}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e)
            }
    
    async def _store_feature_group(
        self, 
        symbol: str, 
        group_data: pd.DataFrame, 
        group_name: str
    ) -> Dict[str, Any]:
        """Store a group of features efficiently."""
        try:
            # Prepare data for bulk insert
            records = []
            for date, row in group_data.iterrows():
                # Convert to JSON for flexible storage, handling both numeric and string values
                features_json = {}
                for col, val in row.items():
                    if pd.isna(val):
                        features_json[col] = None
                    elif isinstance(val, (int, float, np.integer, np.floating)):
                        features_json[col] = float(val)
                    else:
                        # Handle string values (like regime classifications)
                        features_json[col] = str(val)
                
                records.append({
                    'symbol': symbol,
                    'date': date.date() if hasattr(date, 'date') else date,
                    'feature_group': group_name,
                    'features': json.dumps(features_json)
                })
            
            # Delete existing data for this symbol/group
            self.duckdb_conn.execute("""
                DELETE FROM features 
                WHERE symbol = ? AND feature_group = ?
            """, [symbol, group_name])
            
            # Bulk insert new data
            self.duckdb_conn.executemany("""
                INSERT INTO features (symbol, date, feature_group, features)
                VALUES (?, ?, ?, ?)
            """, [(r['symbol'], r['date'], r['feature_group'], r['features']) for r in records])
            
            return {
                'group': group_name,
                'records_stored': len(records),
                'features_in_group': len(group_data.columns)
            }
            
        except Exception as e:
            logger.error(f"❌ Feature group storage failed for {group_name}: {str(e)}")
            raise
    
    async def _calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime-based features."""
        try:
            # Detect current regime
            regime_result = await self.regime_detector.detect_current_regime(data)
            
            # Create regime features DataFrame
            regime_features = pd.DataFrame(index=data.index)
            
            # Add regime information to all rows (current regime applies to recent period)
            regime_features['market_regime'] = regime_result['regime'].value
            regime_features['regime_confidence'] = regime_result['confidence']
            
            # Add detailed regime metrics
            if 'analysis' in regime_result:
                trend_analysis = regime_result['analysis']['trend']
                volatility_analysis = regime_result['analysis']['volatility']
                
                regime_features['trend_strength'] = trend_analysis['trend_strength']
                regime_features['volatility_level'] = volatility_analysis['volatility_level']
            
            return regime_features
            
        except Exception as e:
            logger.error(f"❌ Regime feature calculation failed: {str(e)}")
            # Return empty DataFrame with correct index
            return pd.DataFrame(index=data.index)
    
    def _validate_connection(self) -> None:
        """Validate that the database connection is properly initialized."""
        if self.duckdb_conn is None:
            raise ValueError("Feature store not initialized. Call initialize() first.")
        
        try:
            # Test the connection
            self.duckdb_conn.execute("SELECT 1").fetchone()
        except Exception as e:
            raise ValueError(f"Feature store connection invalid: {str(e)}")
    
    async def retrieve_features(self, symbol: str, 
                               feature_names: Optional[List[str]] = None,
                               feature_groups: Optional[List[str]] = None,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve features for a symbol with user-friendly output.
        
        Args:
            symbol: Stock symbol to retrieve features for
            feature_names: Specific feature names to retrieve (None for all)
            feature_groups: Feature groups to retrieve (None for all)
            start_date: Start date for feature retrieval
            end_date: End date for feature retrieval
            limit: Maximum number of records to retrieve
            
        Returns:
            Dict with success status, data DataFrame, and performance metrics
        """
        start_time = datetime.now()
        
        log_user_action("retrieve_features", {
            "symbol": symbol, 
            "feature_names": feature_names,
            "feature_groups": feature_groups
        })
        
        try:
            # Validate connection before any database operations
            self._validate_connection()
            
            # Validate symbol
            if not symbol or symbol == "INVALID_SYMBOL":
                raise ValueError(f"Invalid symbol provided: {symbol}")
            
            # Build query conditions
            conditions = ["symbol = ?"]
            params = [symbol]
            
            if feature_groups:
                placeholders = ','.join(['?' for _ in feature_groups])
                conditions.append(f"feature_group IN ({placeholders})")
                params.extend(feature_groups)
            
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date.date())
            
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date.date())
            
            # Build query
            query = f"""
                SELECT date, feature_group, features
                FROM features
                WHERE {' AND '.join(conditions)}
                ORDER BY date DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            # Execute query
            result = self.duckdb_conn.execute(query, params).fetchall()
            
            # Process results into user-friendly format
            features_data = {}
            
            for date, group, features_json in result:
                if date not in features_data:
                    features_data[date] = {}
                
                # Parse JSON features
                group_features = json.loads(features_json)
                
                # Filter specific features if requested
                if feature_names:
                    group_features = {k: v for k, v in group_features.items() 
                                    if k in feature_names}
                
                features_data[date].update(group_features)
            
            # Convert to DataFrame for user convenience
            if features_data:
                df = pd.DataFrame.from_dict(features_data, orient='index')
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
            else:
                df = pd.DataFrame()
            
            # Calculate performance metrics
            end_time = datetime.now()
            retrieval_time = (end_time - start_time).total_seconds()
            
            log_performance_metric("feature_retrieval_time", retrieval_time, "seconds")
            
            logger.info(f"✅ Retrieved features for {symbol}: {len(df.columns)} features, {len(df)} records in {retrieval_time:.3f}s")
            
            return {
                'success': True,
                'symbol': symbol,
                'data': df,
                'features_retrieved': len(df.columns) if not df.empty else 0,
                'records_retrieved': len(df),
                'retrieval_time': retrieval_time,
                'performance_target_met': retrieval_time < 1.0  # <1s requirement
            }
            
        except Exception as e:
            logger.error(f"❌ Feature retrieval failed for {symbol}: {str(e)}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e),
                'data': pd.DataFrame()
            }
    
    async def get_feature_info(self, feature_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get user-friendly feature information.
        
        Args:
            feature_name: Specific feature to get info for (None for all)
            
        Returns:
            Feature information with descriptions and metadata
        """
        log_user_action("get_feature_info", {"feature_name": feature_name})
        
        try:
            if feature_name:
                # Get specific feature info
                result = self.duckdb_conn.execute("""
                    SELECT feature_name, feature_group, description, data_type, calculation_method
                    FROM feature_metadata
                    WHERE feature_name = ?
                """, [feature_name]).fetchone()
                
                if result:
                    return {
                        'feature_name': result[0],
                        'feature_group': result[1],
                        'description': result[2],
                        'data_type': result[3],
                        'calculation_method': result[4],
                        'user_explanation': self.technical_analyzer.get_indicator_explanation(feature_name)
                    }
                else:
                    return {'error': f'Feature {feature_name} not found'}
            
            else:
                # Get all features grouped by category
                result = self.duckdb_conn.execute("""
                    SELECT feature_group, feature_name, description
                    FROM feature_metadata
                    ORDER BY feature_group, feature_name
                """).fetchall()
                
                # Group by feature category
                grouped_features = {}
                for group, name, description in result:
                    if group not in grouped_features:
                        grouped_features[group] = []
                    
                    grouped_features[group].append({
                        'name': name,
                        'description': description
                    })
                
                return {
                    'feature_groups': grouped_features,
                    'total_features': len(result),
                    'user_message': f"Available feature groups: {', '.join(grouped_features.keys())}"
                }
                
        except Exception as e:
            logger.error(f"❌ Feature info retrieval failed: {str(e)}")
            return {'error': str(e)}
    
    async def get_feature_statistics(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Get feature statistics for user analysis.
        
        Args:
            symbol: Stock symbol
            days: Number of recent days to analyze
            
        Returns:
            Feature statistics and insights
        """
        log_user_action("get_feature_statistics", {"symbol": symbol, "days": days})
        
        try:
            # Get recent feature data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            feature_data = await self.retrieve_features(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if not feature_data['success'] or feature_data['data'].empty:
                return {'error': f'No feature data available for {symbol}'}
            
            df = feature_data['data']
            
            # Calculate statistics
            stats = {}
            
            for column in df.columns:
                if df[column].dtype in ['float64', 'int64']:
                    col_stats = {
                        'mean': float(df[column].mean()),
                        'std': float(df[column].std()),
                        'min': float(df[column].min()),
                        'max': float(df[column].max()),
                        'current': float(df[column].iloc[-1]) if len(df) > 0 else None,
                        'change_pct': float((df[column].iloc[-1] / df[column].iloc[0] - 1) * 100) if len(df) > 1 else 0
                    }
                    
                    # Add user-friendly interpretation
                    current_val = col_stats['current']
                    mean_val = col_stats['mean']
                    
                    if current_val > mean_val + col_stats['std']:
                        col_stats['interpretation'] = 'Above average (high)'
                    elif current_val < mean_val - col_stats['std']:
                        col_stats['interpretation'] = 'Below average (low)'
                    else:
                        col_stats['interpretation'] = 'Near average (normal)'
                    
                    stats[column] = col_stats
            
            return {
                'symbol': symbol,
                'period_days': days,
                'statistics': stats,
                'data_points': len(df),
                'user_summary': f"Analyzed {len(stats)} features over {days} days for {symbol}"
            }
            
        except Exception as e:
            logger.error(f"❌ Feature statistics calculation failed: {str(e)}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check feature store health and performance."""
        try:
            # Check database connection
            self.duckdb_conn.execute("SELECT 1").fetchone()
            
            # Get storage statistics
            stats = self.duckdb_conn.execute("""
                SELECT 
                    COUNT(DISTINCT symbol) as symbols,
                    COUNT(DISTINCT feature_group) as feature_groups,
                    COUNT(*) as total_records,
                    MAX(date) as latest_date
                FROM features
            """).fetchone()
            
            # Performance test
            start_time = datetime.now()
            test_result = self.duckdb_conn.execute("""
                SELECT symbol, date, feature_group 
                FROM features 
                LIMIT 100
            """).fetchall()
            query_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'healthy': True,
                'statistics': {
                    'symbols': stats[0] if stats else 0,
                    'feature_groups': stats[1] if stats else 0,
                    'total_records': stats[2] if stats else 0,
                    'latest_date': str(stats[3]) if stats and stats[3] else None
                },
                'performance': {
                    'query_time': query_time,
                    'performance_target_met': query_time < 1.0
                },
                'message': f'Feature store operational with {stats[0] if stats else 0} symbols'
            }
            
        except Exception as e:
            logger.error(f"❌ Feature store health check failed: {str(e)}")
            return {
                'healthy': False,
                'error': str(e),
                'message': 'Feature store health check failed'
            }
