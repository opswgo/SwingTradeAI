"""
Regime-Aware ML Model for Options Trading Predictions

This module implements a sophisticated ML model that adapts its predictions
based on market regime detection, providing transparent and explainable
trading recommendations for options strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Local imports
from ..core.config import Settings
from ..core.logger import log_user_action, log_performance_metric
from loguru import logger
from ..features.market_regime import MarketRegimeDetector
from ..features.technical_analysis import TechnicalAnalysis
from ..features.feature_store import FeatureStore


@dataclass
class ModelPrediction:
    """Structure for model predictions with confidence and explanations."""
    signal: str  # 'BUY_CALL', 'BUY_PUT', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    expected_return: float  # Expected percentage return
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    regime: str  # Current market regime
    explanation: str  # User-friendly explanation
    features_importance: Dict[str, float]  # Feature contributions
    timestamp: datetime


@dataclass
class ModelMetrics:
    """Structure for model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    regime_accuracy: Dict[str, float]  # Per-regime performance
    feature_importance: Dict[str, float]
    training_samples: int
    validation_samples: int
    last_updated: datetime


class RegimeAwareModel:
    """
    Regime-aware ML model for options trading predictions.
    
    This model adapts its predictions based on detected market regimes,
    providing transparent and explainable trading recommendations.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.regime_detector = MarketRegimeDetector()
        self.technical_analyzer = TechnicalAnalysis()
        self.feature_store = FeatureStore(settings)
        
        # Model components
        self.regime_models = {}  # Separate models per regime
        self.meta_model = None  # Meta-model for regime selection
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Model state
        self.is_trained = False
        self.last_training_date = None
        self.model_version = "1.0.0"
        self.metrics = None
        
        # Feature configuration
        self.feature_columns = [
            # Technical indicators
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'stoch_k', 'stoch_d', 'williams_r',
            'atr', 'adx', 'cci', 'mfi',
            # Volume indicators
            'volume_ratio', 'obv', 'ad_line', 'cmf',
            # Regime features
            'regime_trend_score', 'regime_volatility_score', 'regime_breadth_score',
            'regime_confidence'
        ]
        
        # Target configuration
        self.prediction_horizon = 5  # 5 days ahead
        self.return_thresholds = {
            'strong_buy': 0.05,  # 5% expected return
            'buy': 0.02,         # 2% expected return
            'hold': 0.01,        # 1% expected return
            'sell': -0.02        # -2% expected return
        }
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the model and its dependencies."""
        try:
            logger.info("🤖 Initializing Regime-Aware ML Model...")
            
            # Initialize dependencies
            await self.feature_store.initialize()
            
            # Create model directory
            model_dir = Path(self.settings.DATA_DIR) / "trained_models"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to load existing model
            await self._load_model()
            
            logger.info("✅ Regime-Aware ML Model initialized successfully")
            return {
                'success': True,
                'model_version': self.model_version,
                'is_trained': self.is_trained,
                'last_training_date': self.last_training_date
            }
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def train_model(self, symbol: str, lookback_days: int = 252) -> Dict[str, Any]:
        """
        Train the regime-aware model with historical data.
        
        Args:
            symbol: Stock symbol to train on
            lookback_days: Days of historical data to use
            
        Returns:
            Training results and metrics
        """
        log_user_action("train_model", {"symbol": symbol, "lookback_days": lookback_days})
        
        try:
            logger.info(f"🎯 Training regime-aware model for {symbol}...")
            
            # Prepare training data
            training_data = await self._prepare_training_data(symbol, lookback_days)
            if len(training_data) < 100:
                raise ValueError(f"Insufficient data for training: {len(training_data)} samples (minimum 100 required)")
            
            # Split features and targets
            X, y, regimes = await self._prepare_features_and_targets(training_data)
            
            # Check if we have enough data for each regime and multiple target classes
            regime_counts = pd.Series(regimes).value_counts()
            target_counts = pd.Series(y).value_counts()
            
            logger.info(f"📊 Target class distribution: {dict(target_counts)}")
            logger.info(f"📊 Regime distribution: {dict(regime_counts)}")
            
            if len(target_counts) < 2:
                raise ValueError(f"Insufficient target diversity: only {len(target_counts)} target class(es) found. Need at least 2 classes for training.")
            
            for regime, count in regime_counts.items():
                if count < 10:
                    logger.warning(f"⚠️ Insufficient data for regime {regime} ({count} samples)")
            
            # Check if we have multiple regimes or need to use a single global model
            unique_regimes = np.unique(regimes)
            regime_metrics = {}
            
            if len(unique_regimes) == 1:
                # Single regime scenario - train one global model
                regime = unique_regimes[0]
                logger.info(f"📊 Single regime detected ({regime}) - training global model")
                
                # Check target class diversity
                unique_targets = np.unique(y)
                if len(unique_targets) < 2:
                    raise ValueError(f"Insufficient target diversity: only {len(unique_targets)} target class(es) found")
                
                # Train global model with all data
                global_model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                # Cross-validation
                cv_scores = cross_val_score(global_model, X, y, cv=min(3, len(unique_targets)), scoring='accuracy')
                global_model.fit(X, y)
                
                # Store as regime model
                self.regime_models[regime] = global_model
                regime_metrics[regime] = np.mean(cv_scores)
                
                logger.info(f"✅ Trained global model for {regime}: {np.mean(cv_scores):.3f} accuracy")
                
            else:
                # Multiple regimes - train regime-specific models
                for regime in unique_regimes:
                    regime_mask = regimes == regime
                    regime_X = X[regime_mask]
                    regime_y = y[regime_mask]
                    
                    if len(regime_X) < 10:
                        logger.warning(f"⚠️ Skipping regime {regime} - insufficient data ({len(regime_X)} samples)")
                        continue
                    
                    # Check target diversity for this regime
                    regime_unique_targets = np.unique(regime_y)
                    if len(regime_unique_targets) < 2:
                        logger.warning(f"⚠️ Skipping regime {regime} - insufficient target diversity ({len(regime_unique_targets)} classes)")
                        continue
                    
                    # Train regime-specific model
                    regime_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    
                    # Cross-validation for regime model
                    cv_scores = cross_val_score(regime_model, regime_X, regime_y, cv=min(3, len(regime_unique_targets)), scoring='accuracy')
                    regime_model.fit(regime_X, regime_y)
                    
                    self.regime_models[regime] = regime_model
                    regime_metrics[regime] = np.mean(cv_scores)
                    
                    logger.info(f"✅ Trained {regime} model: {np.mean(cv_scores):.3f} accuracy")
            
            # Train meta-model for regime prediction
            meta_metrics = await self._train_meta_model(X, regimes)
            
            # Calculate overall metrics
            overall_metrics = await self._calculate_overall_metrics(X, y, regimes)
            
            # Update model state
            self.is_trained = True
            self.last_training_date = datetime.now()
            self.metrics = ModelMetrics(
                accuracy=overall_metrics['accuracy'],
                precision=overall_metrics['precision'],
                recall=overall_metrics['recall'],
                f1_score=overall_metrics['f1_score'],
                regime_accuracy=regime_metrics,
                feature_importance=overall_metrics['feature_importance'],
                training_samples=len(X),
                validation_samples=int(len(X) * 0.2),
                last_updated=datetime.now()
            )
            
            # Save model
            await self._save_model()
            
            log_performance_metric("model_training_time", overall_metrics['training_time'])
            log_user_action("train_model", {"symbol": symbol, "accuracy": overall_metrics['accuracy']})
            
            logger.info(f"🎉 Model training complete: {overall_metrics['accuracy']:.3f} accuracy")
            
            return {
                'success': True,
                'accuracy': overall_metrics['accuracy'],
                'regime_accuracies': regime_metrics,
                'feature_importance': overall_metrics['feature_importance'],
                'training_samples': len(X),
                'model_version': self.model_version
            }
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {str(e)}")
            log_user_action("train_model", {"symbol": symbol, "error": str(e)}, success=False)
            return {'success': False, 'error': str(e)}
    
    async def predict(self, symbol: str, explain: bool = True) -> ModelPrediction:
        """
        Make a trading prediction for the given symbol.
        
        Args:
            symbol: Stock symbol to predict
            explain: Whether to include explanations
            
        Returns:
            Model prediction with confidence and explanations
        """
        log_user_action("model_predict", {"symbol": symbol})
        
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Get current features
            features_result = await self.feature_store.retrieve_features(symbol, limit=10)
            features_df = features_result.get('data', pd.DataFrame()) if features_result.get('success') else pd.DataFrame()
            if features_df.empty:
                raise ValueError(f"No features available for {symbol}")
            
            # Use the most recent data point
            features_df = features_df.iloc[-1:].copy()
            
            # Detect current regime
            regime_result = await self.regime_detector.detect_current_regime(features_df)
            detected_regime = regime_result['regime']
            
            # Convert regime enum to string if needed
            if hasattr(detected_regime, 'value'):
                current_regime = detected_regime.value
            else:
                current_regime = str(detected_regime).replace('MarketRegime.', '').lower()
            
            # Prepare features for prediction
            X = await self._prepare_prediction_features(features_df)
            
            # Get regime-specific model
            if current_regime not in self.regime_models:
                logger.error(f"❌ No model for regime {current_regime}, cannot proceed without real data")
                raise ValueError(f"No trained model available for regime {current_regime} - requires real data setup")
            
            model = self.regime_models[current_regime]
            
            # Make prediction
            prediction_proba = model.predict_proba(X)[0]
            prediction_class = model.predict(X)[0]
            confidence = np.max(prediction_proba)
            
            # Convert to trading signal
            signal = self._convert_to_trading_signal(prediction_class, confidence)
            
            # Calculate expected return
            expected_return = self._calculate_expected_return(prediction_class, confidence)
            
            # Determine risk level
            risk_level = self._determine_risk_level(confidence, current_regime)
            
            # Generate explanation
            explanation = ""
            features_importance = {}
            if explain:
                explanation, features_importance = await self._generate_explanation(
                    X, model, signal, current_regime, confidence
                )
            
            prediction = ModelPrediction(
                signal=signal,
                confidence=confidence,
                expected_return=expected_return,
                risk_level=risk_level,
                regime=current_regime,
                explanation=explanation,
                features_importance=features_importance,
                timestamp=datetime.now()
            )
            
            log_user_action("model_predict", {
                "symbol": symbol, 
                "signal": signal, 
                "confidence": confidence,
                "regime": current_regime
            })
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ Prediction failed for {symbol}: {str(e)}")
            log_user_action("model_predict", {"symbol": symbol, "error": str(e)}, success=False)
            
            # Return safe fallback prediction
            return ModelPrediction(
                signal='HOLD',
                confidence=0.0,
                expected_return=0.0,
                risk_level='HIGH',
                regime='unknown',
                explanation=f"Prediction failed: {str(e)}",
                features_importance={},
                timestamp=datetime.now()
            )
    
    async def get_model_metrics(self) -> Dict[str, Any]:
        """Get current model performance metrics."""
        if not self.metrics:
            return {'error': 'Model not trained yet'}
        
        return {
            'accuracy': self.metrics.accuracy,
            'precision': self.metrics.precision,
            'recall': self.metrics.recall,
            'f1_score': self.metrics.f1_score,
            'regime_accuracies': self.metrics.regime_accuracy,
            'feature_importance': dict(sorted(
                self.metrics.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]),  # Top 10 features
            'training_samples': self.metrics.training_samples,
            'model_version': self.model_version,
            'last_updated': self.metrics.last_updated.isoformat(),
            'is_trained': self.is_trained
        }
    
    async def _prepare_training_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Prepare training data with features and targets."""
        # Get historical features - use more data for better training
        features_result = await self.feature_store.retrieve_features(symbol, limit=max(lookback_days, 1000))
        features_df = features_result.get('data', pd.DataFrame()) if features_result.get('success') else pd.DataFrame()
        
        if features_df.empty:
            logger.warning(f"⚠️ No stored features for {symbol}, calculating fresh...")
            # Calculate features if not stored
            from ..data.pipeline import DataPipeline
            pipeline = DataPipeline(self.settings)
            await pipeline.initialize()
            
            data, _ = await pipeline.fetch_historical_data(symbol, period="2y")
            enhanced_data, _ = self.technical_analyzer.calculate_all_indicators(data)
            
            # Store features for future use
            await self.feature_store.store_features(symbol, enhanced_data)
            features_result = await self.feature_store.retrieve_features(symbol, limit=lookback_days)
            features_df = features_result.get('data', pd.DataFrame()) if features_result.get('success') else pd.DataFrame()
        
        return features_df
    
    async def _prepare_features_and_targets(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and targets for training."""
        # Select feature columns that exist in data
        available_features = [col for col in self.feature_columns if col in data.columns]
        X = data[available_features].fillna(0).values
        
        # Create targets based on future returns
        if 'Close' not in data.columns:
            raise ValueError("Cannot train model without real market data - 'Close' price column is required")
        
        returns = data['Close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        # Convert returns to categorical targets
        y = np.where(returns > self.return_thresholds['strong_buy'], 3,  # Strong buy
                    np.where(returns > self.return_thresholds['buy'], 2,      # Buy
                            np.where(returns > self.return_thresholds['hold'], 1,    # Hold
                                    0)))  # Sell
        
        # Get regime labels
        if 'regime' in data.columns and not data['regime'].isna().all():
            regimes = data['regime'].values
        else:
            # No fallback regimes - require real regime data
            raise ValueError("Training requires real market regime data - cannot train on estimated or default regimes")
        
        # Remove NaN rows
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | pd.isna(regimes))
        X = X[valid_mask]
        y = y[valid_mask]
        regimes = regimes[valid_mask]
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y, regimes
    
    async def _train_regime_model(self, regime: str, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train a model for a specific regime."""
        # Use Random Forest for regime-specific predictions
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
        
        # Train final model
        model.fit(X, y)
        
        # Store model
        self.regime_models[regime] = model
        
        return {
            'accuracy': np.mean(scores),
            'std': np.std(scores),
            'samples': len(X)
        }
    
    async def _train_meta_model(self, X: np.ndarray, regimes: np.ndarray) -> Dict[str, float]:
        """Train meta-model for regime prediction."""
        # Check if we have multiple regimes
        unique_regimes = np.unique(regimes)
        
        if len(unique_regimes) == 1:
            # Single regime - create a dummy meta-model that always predicts the same regime
            logger.info(f"📊 Single regime detected - creating dummy meta-model for {unique_regimes[0]}")
            
            # Encode the single regime
            self.label_encoder.fit(regimes)
            
            # Create a dummy classifier that always predicts the single regime
            from sklearn.dummy import DummyClassifier
            self.meta_model = DummyClassifier(strategy='constant', constant=0)
            self.meta_model.fit(X, np.zeros(len(X)))  # All samples get class 0
            
            return {
                'accuracy': 1.0,  # Perfect accuracy since there's only one class
                'std': 0.0
            }
        else:
            # Multiple regimes - train proper logistic regression
            logger.info(f"📊 Multiple regimes detected - training logistic regression meta-model")
            
            # Encode regime labels
            regimes_encoded = self.label_encoder.fit_transform(regimes)
            
            # Train logistic regression for regime classification
            self.meta_model = LogisticRegression(random_state=42, max_iter=1000)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = cross_val_score(self.meta_model, X, regimes_encoded, cv=tscv, scoring='accuracy')
            
            # Train final model
            self.meta_model.fit(X, regimes_encoded)
            
            return {
                'accuracy': np.mean(scores),
                'std': np.std(scores)
            }
    
    async def _calculate_overall_metrics(self, X: np.ndarray, y: np.ndarray, regimes: np.ndarray) -> Dict[str, Any]:
        """Calculate overall model performance metrics."""
        start_time = datetime.now()
        
        # Make predictions using regime-specific models
        predictions = []
        for i, (x_sample, regime) in enumerate(zip(X, regimes)):
            if regime in self.regime_models:
                pred = self.regime_models[regime].predict([x_sample])[0]
            else:
                pred = 1  # Default to hold
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions.round())
        precision = precision_score(y, predictions.round(), average='weighted', zero_division=0)
        recall = recall_score(y, predictions.round(), average='weighted', zero_division=0)
        f1 = f1_score(y, predictions.round(), average='weighted', zero_division=0)
        
        # Feature importance (from first regime model)
        feature_importance = {}
        if self.regime_models:
            first_model = list(self.regime_models.values())[0]
            if hasattr(first_model, 'feature_importances_'):
                available_features = [col for col in self.feature_columns if col in self.feature_columns]
                for i, importance in enumerate(first_model.feature_importances_):
                    if i < len(available_features):
                        feature_importance[available_features[i]] = float(importance)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': feature_importance,
            'training_time': training_time
        }
    
    async def _prepare_prediction_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction."""
        # Get the latest row for prediction
        latest_data = features_df.iloc[-1:].copy()
        
        # Check available features
        available_features = [col for col in self.feature_columns if col in latest_data.columns]
        
        if len(available_features) == 0:
            # If no training features available, get fresh features
            logger.warning(f"⚠️ No training features found in stored data, calculating fresh features...")
            
            # Calculate fresh technical indicators for prediction
            from ..features.technical_analysis import TechnicalAnalysis
            tech_analyzer = TechnicalAnalysis()
            
            # Get more historical data for technical indicators
            historical_result = await self.feature_store.retrieve_features(features_df.index[0], limit=50)
            if historical_result.get('success'):
                historical_data = historical_result.get('data')
                enhanced_data, _ = tech_analyzer.calculate_all_indicators(historical_data)
                latest_data = enhanced_data.iloc[-1:].copy()
                available_features = [col for col in self.feature_columns if col in latest_data.columns]
        
        logger.info(f"📊 Using {len(available_features)} features for prediction: {available_features[:5]}...")
        
        if len(available_features) == 0:
            raise ValueError(f"No valid features available for prediction. Expected features: {self.feature_columns[:5]}...")
        
        # Extract and scale features
        X = latest_data[available_features].fillna(0).values
        
        if X.shape[1] == 0:
            raise ValueError(f"Feature extraction resulted in 0 features. Available columns: {list(latest_data.columns)}")
        
        X = self.scaler.transform(X)
        return X
    
    def _convert_to_trading_signal(self, prediction_class: int, confidence: float) -> str:
        """Convert model prediction to trading signal."""
        if confidence < 0.6:  # Low confidence
            return 'HOLD'
        
        if prediction_class >= 3:
            return 'BUY_CALL'
        elif prediction_class >= 2:
            return 'BUY_CALL'
        elif prediction_class <= 0:
            return 'BUY_PUT'
        else:
            return 'HOLD'
    
    def _calculate_expected_return(self, prediction_class: int, confidence: float) -> float:
        """Calculate expected return based on prediction."""
        base_returns = {3: 0.05, 2: 0.02, 1: 0.01, 0: -0.02}
        base_return = base_returns.get(prediction_class, 0.0)
        return base_return * confidence
    
    def _determine_risk_level(self, confidence: float, regime: str) -> str:
        """Determine risk level based on confidence and regime."""
        if confidence < 0.5:
            return 'HIGH'
        elif confidence < 0.7:
            return 'MEDIUM'
        elif regime in ['crisis', 'volatile']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    async def _generate_explanation(self, X: np.ndarray, model, signal: str, regime: str, confidence: float) -> Tuple[str, Dict[str, float]]:
        """Generate user-friendly explanation for the prediction."""
        # Get feature importance from the model
        feature_importance = {}
        if hasattr(model, 'feature_importances_') and len(model.feature_importances_) > 0:
            # Get available features that were actually used in training
            available_features = [col for col in self.feature_columns if col in self.feature_columns]
            if len(available_features) == len(model.feature_importances_):
                for i, importance in enumerate(model.feature_importances_):
                    if i < len(available_features):
                        feature_importance[available_features[i]] = float(importance)
        
        # If no feature importance available, use stored training importance
        if not feature_importance and hasattr(self.metrics, 'feature_importance'):
            feature_importance = dict(self.metrics.feature_importance)
        
        # Generate detailed explanation
        confidence_desc = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"
        regime_desc = regime.replace('_', ' ').title()
        
        explanation = f"🎯 PREDICTION ANALYSIS:\n"
        explanation += f"Market Regime: {regime_desc} (detected)\n"
        explanation += f"Confidence Level: {confidence:.1%} ({confidence_desc})\n"
        explanation += f"Recommended Action: {signal}\n\n"
        
        # Add signal reasoning
        if signal == 'BUY_CALL':
            explanation += "📈 BULLISH SIGNALS DETECTED:\n"
            explanation += "• Technical indicators suggest upward price movement\n"
            explanation += "• Market conditions favor call option strategies\n"
        elif signal == 'BUY_PUT':
            explanation += "📉 BEARISH SIGNALS DETECTED:\n"
            explanation += "• Technical indicators suggest downward price movement\n"
            explanation += "• Market conditions favor put option strategies\n"
        else:
            explanation += "⚖️ NEUTRAL/MIXED SIGNALS:\n"
            explanation += "• Technical indicators show conflicting signals\n"
            explanation += "• Recommend holding current positions\n"
        
        # Add top contributing factors with explanations
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            explanation += "\n🔍 KEY TECHNICAL FACTORS:\n"
            
            feature_explanations = {
                'rsi': 'Relative Strength Index - momentum oscillator',
                'macd': 'MACD - trend following momentum indicator', 
                'macd_signal': 'MACD Signal Line - trend confirmation',
                'bb_upper': 'Bollinger Band Upper - volatility measure',
                'bb_lower': 'Bollinger Band Lower - support level',
                'sma_20': '20-day Simple Moving Average - short-term trend',
                'sma_50': '50-day Simple Moving Average - medium-term trend',
                'ema_12': '12-day Exponential Moving Average - recent price action',
                'williams_r': 'Williams %R - momentum indicator',
                'stoch_k': 'Stochastic %K - momentum oscillator',
                'atr': 'Average True Range - volatility measure',
                'adx': 'Average Directional Index - trend strength'
            }
            
            for feat, importance in top_features:
                feat_desc = feature_explanations.get(feat, feat.replace('_', ' ').title())
                explanation += f"• {feat_desc}: {importance:.1%} influence\n"
        
        explanation += f"\n⚠️ RISK ASSESSMENT:\n"
        if confidence < 0.6:
            explanation += "• HIGH RISK: Low confidence prediction - consider smaller position sizes\n"
        elif confidence < 0.8:
            explanation += "• MODERATE RISK: Reasonable confidence - standard position sizing\n"
        else:
            explanation += "• LOW RISK: High confidence prediction - consider normal position sizes\n"
        
        explanation += f"• Market regime ({regime_desc}) affects risk profile\n"
        explanation += f"• Always use proper risk management and stop losses\n"
        
        return explanation, feature_importance
    
    async def _save_model(self):
        """Save the trained model to disk."""
        try:
            model_dir = Path(self.settings.DATA_DIR) / "trained_models"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save regime models
            for regime, model in self.regime_models.items():
                model_path = model_dir / f"regime_model_{regime}.joblib"
                joblib.dump(model, model_path)
            
            # Save meta model
            if self.meta_model:
                joblib.dump(self.meta_model, model_dir / "meta_model.joblib")
            
            # Save scaler and label encoder
            joblib.dump(self.scaler, model_dir / "scaler.joblib")
            joblib.dump(self.label_encoder, model_dir / "label_encoder.joblib")
            
            # Save metadata
            metadata = {
                'model_version': self.model_version,
                'last_training_date': self.last_training_date.isoformat() if self.last_training_date else None,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }
            
            import json
            with open(model_dir / "model_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("✅ Model saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model: {str(e)}")
    
    async def _load_model(self):
        """Load existing model from disk."""
        try:
            model_dir = Path(self.settings.DATA_DIR) / "trained_models"
            
            if not model_dir.exists():
                logger.info("📁 No existing model found, will train from scratch")
                return
            
            # Load metadata
            metadata_path = model_dir / "model_metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.model_version = metadata.get('model_version', '1.0.0')
                self.feature_columns = metadata.get('feature_columns', self.feature_columns)
                self.is_trained = metadata.get('is_trained', False)
                
                if metadata.get('last_training_date'):
                    self.last_training_date = datetime.fromisoformat(metadata['last_training_date'])
            
            # Load regime models
            for model_file in model_dir.glob("regime_model_*.joblib"):
                regime = model_file.stem.replace("regime_model_", "")
                self.regime_models[regime] = joblib.load(model_file)
            
            # Load meta model
            meta_model_path = model_dir / "meta_model.joblib"
            if meta_model_path.exists():
                self.meta_model = joblib.load(meta_model_path)
            
            # Load scaler and label encoder
            scaler_path = model_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            encoder_path = model_dir / "label_encoder.joblib"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
            
            if self.regime_models:
                logger.info(f"✅ Loaded existing model with {len(self.regime_models)} regime models")
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing model: {str(e)}")
            self.is_trained = False
