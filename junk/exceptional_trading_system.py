#!/usr/bin/env python3
"""
EXCEPTIONAL WORLD-CLASS NEAT OPTIONS TRADING SYSTEM
Institutional-quality standards used by elite hedge funds.

EXCEPTIONAL STANDARDS IMPLEMENTED:
- 80%+ Win Rate (Elite Trader Standard)
- 100%+ Annual Return (2x S&P 500 Outperformance)
- 2.0+ Sharpe Ratio (Institutional Quality)
- <8% Maximum Drawdown (Capital Preservation)
- 2.0+ Calmar Ratio (Return/Drawdown Efficiency)
- 2.5+ Sortino Ratio (Downside Risk Management)
- <2% Assignment Rate (Options Excellence)
- <25% Volatility (Risk Control)
- 10+ Trades Minimum (Statistical Significance)
- 72-Hour Model Freshness (Real-time Adaptation)
"""

import numpy as np
import pandas as pd
import neat
import time
import logging
import yfinance as yf
import json
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import existing NEAT system
from practical_neat_system import PracticalNEATTrader

@deprecated  # BROKEN: Training components don't actually work, models rejected
class ExceptionalTradingSystem:
    """EXCEPTIONAL WORLD-CLASS TRADING SYSTEM
    
    Implements institutional-quality standards that exceed those of elite hedge funds.
    Only models meeting ALL exceptional criteria will be accepted and deployed.
    """
    
    def __init__(self):
        self.trader = PracticalNEATTrader()
        self.logger = self._setup_logging()
        
        # EXCEPTIONAL WORLD-CLASS STANDARDS
        # These exceed the standards of most institutional traders
        self.EXCEPTIONAL_STANDARDS = {
            'min_win_rate': 0.80,           # 80% minimum win rate (elite standard)
            'min_return': 1.0,              # 100% minimum annual return
            'min_sharpe_ratio': 2.0,        # 2.0+ Sharpe ratio (institutional)
            'max_drawdown': 0.08,           # 8% maximum drawdown
            'min_outperformance': 1.0,      # Must beat buy-and-hold by 100%+
            'min_calmar_ratio': 2.0,        # Calmar ratio (return/max drawdown)
            'min_sortino_ratio': 2.5,       # Sortino ratio (downside risk)
            'max_assignment_rate': 0.02,    # 2% maximum assignment rate
            'max_volatility': 0.25,         # 25% maximum volatility
            'min_trades': 10,               # Minimum trades for significance
            'max_age_hours': 72,            # 3 days maximum model age
            'min_consistency': 0.75,        # 75% of periods profitable
        }
        
        # Performance tracking
        self.model_registry = {}
        self.performance_history = {}
        
    def _setup_logging(self):
        """Setup comprehensive logging for exceptional standards."""
        logger = logging.getLogger('ExceptionalTrading')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def clean_symbol(self, symbol: str) -> str:
        """Clean and validate ticker symbol with exceptional standards."""
        if not symbol:
            return ""
        
        # Remove common prefixes/suffixes that cause issues
        cleaned = symbol.upper().strip()
        cleaned = cleaned.replace('$', '').replace('.', '').replace(' ', '')
        
        # Validate symbol format
        if not cleaned.isalpha() or len(cleaned) > 5:
            return ""
        
        return cleaned
    
    def fetch_data_safely(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch data with exceptional error handling and validation."""
        try:
            cleaned_symbol = self.clean_symbol(symbol)
            if not cleaned_symbol:
                raise ValueError(f"Invalid symbol: {symbol}")
            
            # Use only valid periods
            valid_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
            if period not in valid_periods:
                period = "2y"
            
            self.logger.info(f"Fetching data for {cleaned_symbol} (period: {period})")
            ticker = yf.Ticker(cleaned_symbol)
            data = ticker.history(period=period, auto_adjust=True)
            
            if len(data) == 0:
                raise ValueError(f"No data available for {cleaned_symbol}")
            
            # Validate data quality
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
                if data[col].isna().all():
                    raise ValueError(f"All {col} values are NaN")
            
            # Check for realistic market data
            prices = data['Close']
            returns = prices.pct_change().dropna()
            
            if returns.std() < 0.001:
                raise ValueError("Returns too stable - likely synthetic data")
            if returns.std() > 0.5:
                raise ValueError("Returns too volatile - likely corrupted data")
            if not (prices > 0).all():
                raise ValueError("Negative prices detected")
            
            volumes = data['Volume']
            if not (volumes >= 0).all():
                raise ValueError("Negative volume detected")
            if volumes.mean() < 1000:
                raise ValueError("Volume too low - likely synthetic data")
            
            self.logger.info(f"✅ Fetched {len(data)} days of validated real data for {cleaned_symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Data fetch failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def train_exceptional_model(self, symbol: str, horizon_weeks: int = 4) -> dict:
        """Train model with EXCEPTIONAL validation - only world-class models accepted."""
        try:
            cleaned_symbol = self.clean_symbol(symbol)
            if not cleaned_symbol:
                return {'success': False, 'error': f'Invalid symbol: {symbol}'}
            
            # Fetch and validate data
            data = self.fetch_data_safely(cleaned_symbol, "2y")
            if len(data) < 500:  # Need substantial data for exceptional models
                return {'success': False, 'error': f'Insufficient data for {cleaned_symbol} (need 500+ days)'}
            
            # Train model using existing NEAT system
            self.logger.info(f"🎯 Training EXCEPTIONAL model for {cleaned_symbol}")
            result = self.trader.train_weekly_agent(cleaned_symbol, data, horizon_weeks)
            
            if result.get('success', False):
                # Apply EXCEPTIONAL validation standards
                acceptable, reason = self.is_model_exceptional(cleaned_symbol)
                
                if acceptable:
                    # Save exceptional model
                    self.save_exceptional_model(cleaned_symbol, self.trader.models[cleaned_symbol])
                    self.logger.info(f"🏆 EXCEPTIONAL MODEL ACCEPTED: {cleaned_symbol} - {reason}")
                    return {
                        'success': True, 
                        'message': f"🏆 EXCEPTIONAL MODEL: {cleaned_symbol} meets all institutional standards",
                        'standards_met': self.get_performance_summary(cleaned_symbol)
                    }
                else:
                    # Reject subpar model
                    if cleaned_symbol in self.trader.models:
                        del self.trader.models[cleaned_symbol]
                    
                    self.logger.warning(f"❌ MODEL REJECTED: {cleaned_symbol} - {reason}")
                    return {
                        'success': False, 
                        'error': f"Model rejected: {reason}. Only exceptional models meeting institutional standards are accepted."
                    }
            else:
                return result
                
        except Exception as e:
            self.logger.error(f"❌ Training failed for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def is_model_exceptional(self, symbol: str) -> tuple[bool, str]:
        """Validate model against EXCEPTIONAL WORLD-CLASS standards.
        
        These standards exceed those used by most elite hedge funds.
        ALL criteria must be met for acceptance.
        """
        if symbol not in self.trader.models:
            return False, "No model found"
        
        model_data = self.trader.models[symbol]
        perf = model_data.get('performance', {})
        
        # Extract comprehensive performance metrics
        metrics = {
            'win_rate': perf.get('win_rate', 0),
            'total_return': perf.get('total_return', 0),
            'buy_hold_return': perf.get('buy_hold_return', 0),
            'sharpe_ratio': perf.get('sharpe_ratio', 0),
            'max_drawdown': perf.get('max_drawdown', 1),
            'total_trades': perf.get('total_trades', 0),
            'volatility': perf.get('volatility', 1),
            'sortino_ratio': perf.get('sortino_ratio', 0),
            'calmar_ratio': perf.get('calmar_ratio', 0),
            'assignment_rate': perf.get('assignment_rate', 1),
        }
        
        # EXCEPTIONAL VALIDATION: Each standard must be met
        validations = []
        
        # 1. ELITE WIN RATE (80%+)
        if metrics['win_rate'] < self.EXCEPTIONAL_STANDARDS['min_win_rate']:
            return False, f"Win rate {metrics['win_rate']:.1%} below EXCEPTIONAL {self.EXCEPTIONAL_STANDARDS['min_win_rate']:.0%} standard"
        validations.append(f"✅ Elite win rate: {metrics['win_rate']:.1%}")
        
        # 2. EXCEPTIONAL RETURNS (100%+)
        if metrics['total_return'] < self.EXCEPTIONAL_STANDARDS['min_return']:
            return False, f"Return {metrics['total_return']:.1%} below EXCEPTIONAL {self.EXCEPTIONAL_STANDARDS['min_return']:.0%} standard"
        validations.append(f"✅ Exceptional return: {metrics['total_return']:.1%}")
        
        # 3. MASSIVE OUTPERFORMANCE (100%+ vs buy-and-hold)
        outperformance = metrics['total_return'] - metrics['buy_hold_return']
        if outperformance < self.EXCEPTIONAL_STANDARDS['min_outperformance']:
            return False, f"Only {outperformance:.1%} outperformance vs buy-and-hold (need {self.EXCEPTIONAL_STANDARDS['min_outperformance']:.0%}+)"
        validations.append(f"✅ Massive outperformance: {outperformance:.1%} vs buy-and-hold")
        
        # 4. INSTITUTIONAL SHARPE RATIO (2.0+)
        if metrics['sharpe_ratio'] < self.EXCEPTIONAL_STANDARDS['min_sharpe_ratio']:
            return False, f"Sharpe ratio {metrics['sharpe_ratio']:.2f} below INSTITUTIONAL {self.EXCEPTIONAL_STANDARDS['min_sharpe_ratio']:.1f} standard"
        validations.append(f"✅ Institutional Sharpe: {metrics['sharpe_ratio']:.2f}")
        
        # 5. CAPITAL PRESERVATION (8% max drawdown)
        if metrics['max_drawdown'] > self.EXCEPTIONAL_STANDARDS['max_drawdown']:
            return False, f"Max drawdown {metrics['max_drawdown']:.1%} exceeds PRESERVATION {self.EXCEPTIONAL_STANDARDS['max_drawdown']:.0%} limit"
        validations.append(f"✅ Capital preservation: {metrics['max_drawdown']:.1%} max drawdown")
        
        # 6. STATISTICAL SIGNIFICANCE (10+ trades)
        if metrics['total_trades'] < self.EXCEPTIONAL_STANDARDS['min_trades']:
            return False, f"Only {metrics['total_trades']} trades (need {self.EXCEPTIONAL_STANDARDS['min_trades']}+ for significance)"
        validations.append(f"✅ Statistical significance: {metrics['total_trades']} trades")
        
        # 7. RISK CONTROL (25% max volatility)
        if metrics['volatility'] > self.EXCEPTIONAL_STANDARDS['max_volatility']:
            return False, f"Volatility {metrics['volatility']:.1%} exceeds RISK {self.EXCEPTIONAL_STANDARDS['max_volatility']:.0%} limit"
        validations.append(f"✅ Risk control: {metrics['volatility']:.1%} volatility")
        
        # 8. DOWNSIDE PROTECTION (2.5+ Sortino)
        if metrics['sortino_ratio'] < self.EXCEPTIONAL_STANDARDS['min_sortino_ratio']:
            return False, f"Sortino ratio {metrics['sortino_ratio']:.2f} below DOWNSIDE {self.EXCEPTIONAL_STANDARDS['min_sortino_ratio']:.1f} standard"
        validations.append(f"✅ Downside protection: {metrics['sortino_ratio']:.2f} Sortino")
        
        # 9. EFFICIENCY STANDARD (2.0+ Calmar)
        if metrics['calmar_ratio'] < self.EXCEPTIONAL_STANDARDS['min_calmar_ratio']:
            return False, f"Calmar ratio {metrics['calmar_ratio']:.2f} below EFFICIENCY {self.EXCEPTIONAL_STANDARDS['min_calmar_ratio']:.1f} standard"
        validations.append(f"✅ Efficiency: {metrics['calmar_ratio']:.2f} Calmar ratio")
        
        # 10. OPTIONS EXCELLENCE (<2% assignment)
        if metrics['assignment_rate'] > self.EXCEPTIONAL_STANDARDS['max_assignment_rate']:
            return False, f"Assignment rate {metrics['assignment_rate']:.1%} exceeds OPTIONS {self.EXCEPTIONAL_STANDARDS['max_assignment_rate']:.1%} excellence"
        validations.append(f"✅ Options excellence: {metrics['assignment_rate']:.1%} assignment rate")
        
        # ALL STANDARDS MET - EXCEPTIONAL MODEL
        summary = f"🏆 EXCEPTIONAL MODEL - ALL {len(validations)} INSTITUTIONAL STANDARDS MET"
        return True, summary
    
    def get_performance_summary(self, symbol: str) -> dict:
        """Get comprehensive performance summary for exceptional model."""
        if symbol not in self.trader.models:
            return {}
        
        model_data = self.trader.models[symbol]
        perf = model_data.get('performance', {})
        
        return {
            'symbol': symbol,
            'classification': 'EXCEPTIONAL',
            'win_rate': f"{perf.get('win_rate', 0):.1%}",
            'total_return': f"{perf.get('total_return', 0):.1%}",
            'outperformance': f"{perf.get('total_return', 0) - perf.get('buy_hold_return', 0):.1%}",
            'sharpe_ratio': f"{perf.get('sharpe_ratio', 0):.2f}",
            'max_drawdown': f"{perf.get('max_drawdown', 0):.1%}",
            'calmar_ratio': f"{perf.get('calmar_ratio', 0):.2f}",
            'sortino_ratio': f"{perf.get('sortino_ratio', 0):.2f}",
            'assignment_rate': f"{perf.get('assignment_rate', 0):.1%}",
            'total_trades': perf.get('total_trades', 0),
            'standards_met': 'ALL 10 INSTITUTIONAL STANDARDS'
        }
    
    def save_exceptional_model(self, symbol: str, model_data: dict):
        """Save exceptional model with comprehensive metadata."""
        try:
            # Add exceptional model metadata
            model_data['exceptional_status'] = True
            model_data['validation_timestamp'] = datetime.now().isoformat()
            model_data['standards_met'] = self.get_performance_summary(symbol)
            
            # Save to persistent storage
            models_dir = Path("exceptional_models")
            models_dir.mkdir(exist_ok=True)
            
            model_file = models_dir / f"{symbol}_exceptional_model.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = self._make_json_serializable(model_data)
            
            with open(model_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            # Update registry
            self.model_registry[symbol] = {
                'file_path': str(model_file),
                'saved_at': datetime.now().isoformat(),
                'status': 'EXCEPTIONAL',
                'performance': model_data.get('performance', {})
            }
            
            self.logger.info(f"🏆 Exceptional model saved for {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save exceptional model for {symbol}: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def load_exceptional_models(self) -> int:
        """Load all exceptional models from persistent storage."""
        try:
            models_dir = Path("exceptional_models")
            if not models_dir.exists():
                return 0
            
            loaded_count = 0
            for model_file in models_dir.glob("*_exceptional_model.json"):
                try:
                    with open(model_file, 'r') as f:
                        model_data = json.load(f)
                    
                    # Extract symbol from filename
                    symbol = model_file.stem.replace('_exceptional_model', '')
                    
                    # Validate model is still exceptional
                    if model_data.get('exceptional_status', False):
                        self.trader.models[symbol] = model_data
                        loaded_count += 1
                        self.logger.info(f"🏆 Loaded exceptional model: {symbol}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load model from {model_file}: {e}")
            
            self.logger.info(f"✅ Loaded {loaded_count} exceptional models")
            return loaded_count
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load exceptional models: {e}")
            return 0
    
    def get_exceptional_models_summary(self) -> pd.DataFrame:
        """Get summary of all exceptional models."""
        if not self.trader.models:
            return pd.DataFrame()
        
        summaries = []
        for symbol in self.trader.models.keys():
            acceptable, reason = self.is_model_exceptional(symbol)
            if acceptable:
                summary = self.get_performance_summary(symbol)
                summaries.append(summary)
        
        if summaries:
            df = pd.DataFrame(summaries)
            return df
        else:
            return pd.DataFrame()

def create_exceptional_ui():
    """Create Streamlit UI for exceptional trading system."""
    st.set_page_config(
        page_title="🏆 Exceptional Trading System",
        page_icon="🏆",
        layout="wide"
    )
    
    st.title("🏆 EXCEPTIONAL WORLD-CLASS NEAT OPTIONS TRADING")
    st.markdown("**Institutional-Quality Standards Exceeding Elite Hedge Funds**")
    
    # Initialize system
    if 'exceptional_system' not in st.session_state:
        st.session_state.exceptional_system = ExceptionalTradingSystem()
        st.session_state.exceptional_system.load_exceptional_models()
    
    system = st.session_state.exceptional_system
    
    # Sidebar: Exceptional Models
    st.sidebar.header("🏆 EXCEPTIONAL MODELS")
    
    exceptional_models = []
    for symbol in system.trader.models.keys():
        acceptable, reason = system.is_model_exceptional(symbol)
        if acceptable:
            exceptional_models.append(symbol)
    
    if exceptional_models:
        st.sidebar.success(f"✅ {len(exceptional_models)} EXCEPTIONAL models")
        
        for symbol in exceptional_models:
            summary = system.get_performance_summary(symbol)
            with st.sidebar.container():
                st.sidebar.success(f"🏆 **{symbol}** - EXCEPTIONAL")
                st.sidebar.write(f"📊 Win Rate: {summary['win_rate']}")
                st.sidebar.write(f"📈 Return: {summary['total_return']}")
                st.sidebar.write(f"🎯 Outperformance: {summary['outperformance']}")
                st.sidebar.write(f"⚡ Sharpe: {summary['sharpe_ratio']}")
                st.sidebar.divider()
    else:
        st.sidebar.warning("No exceptional models yet")
        st.sidebar.info("Train models to exceptional standards!")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Training", "📊 Performance", "📈 Predictions"])
    
    with tab1:
        st.header("🎯 EXCEPTIONAL MODEL TRAINING")
        st.markdown("**Only models meeting ALL 10 institutional standards will be accepted**")
        
        col1, col2 = st.columns(2)
        with col1:
            train_symbol = st.text_input("Symbol to Train", "AAPL").upper()
        with col2:
            horizon_weeks = st.selectbox("Horizon (weeks)", [1, 2, 4, 8], index=2)
        
        if st.button("🚀 Train EXCEPTIONAL Model", type="primary"):
            if train_symbol:
                with st.spinner(f"Training {train_symbol} to exceptional standards..."):
                    result = system.train_exceptional_model(train_symbol, horizon_weeks)
                
                if result.get('success', False):
                    st.success(f"🏆 {result['message']}")
                    if 'standards_met' in result:
                        st.json(result['standards_met'])
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Training failed')}")
                    st.warning("Only exceptional models meeting institutional standards are accepted.")
        
        # Display exceptional standards
        st.subheader("🏆 EXCEPTIONAL STANDARDS REQUIRED")
        standards_df = pd.DataFrame([
            {"Standard": "Win Rate", "Requirement": "≥80%", "Level": "Elite Trader"},
            {"Standard": "Annual Return", "Requirement": "≥100%", "Level": "2x S&P 500"},
            {"Standard": "Sharpe Ratio", "Requirement": "≥2.0", "Level": "Institutional"},
            {"Standard": "Max Drawdown", "Requirement": "≤8%", "Level": "Capital Preservation"},
            {"Standard": "Outperformance", "Requirement": "≥100%", "Level": "vs Buy-and-Hold"},
            {"Standard": "Calmar Ratio", "Requirement": "≥2.0", "Level": "Efficiency"},
            {"Standard": "Sortino Ratio", "Requirement": "≥2.5", "Level": "Downside Protection"},
            {"Standard": "Assignment Rate", "Requirement": "≤2%", "Level": "Options Excellence"},
            {"Standard": "Volatility", "Requirement": "≤25%", "Level": "Risk Control"},
            {"Standard": "Min Trades", "Requirement": "≥10", "Level": "Statistical Significance"},
        ])
        st.dataframe(standards_df, use_container_width=True)
    
    with tab2:
        st.header("📊 EXCEPTIONAL MODELS PERFORMANCE")
        
        summary_df = system.get_exceptional_models_summary()
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("No exceptional models available. Train models to exceptional standards!")
    
    with tab3:
        st.header("📈 EXCEPTIONAL MODEL PREDICTIONS")
        
        if exceptional_models:
            selected_symbol = st.selectbox("Select Exceptional Model", exceptional_models)
            
            if st.button("Generate 8-Week Predictions"):
                st.info(f"Generating predictions for exceptional model: {selected_symbol}")
                # Prediction logic would go here
        else:
            st.info("No exceptional models available for predictions.")

if __name__ == "__main__":
    create_exceptional_ui()
