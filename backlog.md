# 📋 SwingTradeAI - Consolidated Backlog

## Critical Priority Items

### #0 - Model Persistence & Evolution System
- [ ] **Incremental Model Updates**: Models should be updatable instead of retrained from scratch every time
  - Implement model versioning with incremental learning capabilities
  - Save training checkpoints to resume from previous state
  - Use transfer learning to build on existing model knowledge
  - Implement model comparison to ensure updates improve performance
- [ ] **Top 10 Winning Models Archive**: Preserve the 10 best performing models per ticker/horizon combination
  - Save complete model metadata: generation created, fitness score, performance metrics, training period
  - Include model genealogy: parent models, mutation/crossover history for breeding insights
  - Store model performance across different market conditions (bull/bear/sideways)
  - Enable model ensemble/breeding: combine top performers for hybrid strategies
  - Implement model backtesting comparison: test historical performance of archived models
  - Auto-archive models when new ones exceed performance thresholds
  - Provide UI to browse, compare, and select from archived models
  - Benefits: Prevent loss of high-performing models, enable genetic algorithm breeding, allow rollback to previous versions, support ensemble methods

## High Priority Items

### #1 - Finviz Integration
- [ ] **Integrate Finviz 52-week high screener** (https://finviz.com/screener.ashx?v=111&f=ta_perf_52w500o&ft=4)
- [ ] **Finviz API Integration**: Implement Finviz screener API with filters:
  - Base filter: https://finviz.com/screener.ashx?v=111&f=cap_midover,sh_avgvol_o500,sh_opt_option&ft=4
  - Exclude earnings: https://finviz.com/screener.ashx?v=111&f=cap_midover%2Cearningsdate_thismonth%2Csh_avgvol_o500%2Csh_opt_option&ft=4

### #2 - Advanced Strategy Features
- [ ] **Dynamic Strike Distance Adjustment**: Model adjusts strike distance dynamically based on confidence
- [ ] **Early Position Management**: Model learns when to roll puts or buy back early for risk management
- [ ] **Continuous Retraining**: Implement continuous retraining with recent data using backtest/forward test requirements

### #3 - Enhanced Risk Management
- [ ] **Sector Correlation Monitoring**: <25% exposure per sector with correlation display
- [ ] **Position Sizing Based on Kelly Criterion**: Optimize position sizes mathematically
- [ ] **Assignment Probability Integration**: Incorporate probability of assignment into effective yield calculations
- [ ] **User-Configurable Risk/Reward Profiles**: Allow users to tailor aggressiveness of strike selection

### #4 - Highly Usable UI (NEW)
- [ ] **Multi-Symbol Training Interface**: Allow users to add new symbols for training
- [ ] **Success Rate Statistics Dashboard**: Show training stats and success rates per symbol
- [ ] **1-8 Week Prediction Charts**: Single chart showing predictions like Bollinger bands
- [ ] **Historical Validation Charts**: 
  - Pick dates before corrections and chart actual vs predicted
  - Pick dates before bull runs and chart actual vs predicted
  - Interactive date selection for validation testing

## Medium Priority Items

### Data & Screening
- [ ] **Earnings Calendar Integration**: Avoid earnings during option period
- [ ] **VIX-based Market Regime Detection**: Enhanced market condition awareness
- [ ] **Alternative Screening Tools**: Research better alternatives to Finviz
- [ ] **Architecture Review + MVP Acceptance Criteria Definition**
- [ ] **Additional Data Sources Research + Options ML Feature Engineering**

### Options & Greeks
- [ ] **Options Chain Analysis and Greeks Optimization**: Advanced options pricing
- [ ] **Dynamic Strike Distance Based on Confidence**: Formula-based strike selection
- [ ] **Early Put Rolling and Buyback Strategies**: Risk management automation

### User Experience
- [ ] **Capital Deployment Configuration**: Ask user for available capital (default $10k)
- [ ] **Sector Filtering Interface**: User-configurable sector preferences
- [ ] **Risk Profile Selection**: Multiple risk/reward profiles
- [ ] **Sentiment Analysis**: News and social media integration
- [ ] **Multi-timeframe Analysis**: Daily, weekly, monthly views

## Low Priority Items

### Evolution & Adaptation
- [ ] **Weekly NEAT Evolution**: Align with option expiration cycles
- [ ] **Monthly Retraining**: Rolling window retraining with latest data
- [ ] **Regime-Triggered Evolution**: Accelerated retraining during market shifts
- [ ] **Performance-Based Adaptation**: Dynamic evolution parameter adjustment

### Monitoring & Validation
- [ ] **Assignment Rate Monitoring**: Real-time tracking and alerting
- [ ] **Premium Yield Tracking**: Continuous monitoring of weekly yield targets
- [ ] **Performance Attribution**: Detailed breakdown of returns by strategy component
- [ ] **Database Schema Design + Monitoring Infrastructure Planning**

### Extended Features
- [ ] **Cryptocurrency and Forex Support**: Expand beyond equities
- [ ] **International Markets**: European, Asian market support
- [ ] **Mobile App**: Trade notifications and mobile interface
- [ ] **Voice Alerts and Notifications**: Audio feedback system
- [ ] **Social Trading and Copy Trading Features**: Community features

---

## Success Metrics

### Financial Performance Targets
- **Annual Return:** >20% (2x S&P 500)
- **Mag7 Outperformance:** >37.5% (1.5x Mag7)
- **Sharpe Ratio:** >2.0
- **Maximum Drawdown:** <15%
- **Win Rate:** >75%

### Technical Performance Targets
- **Prediction Accuracy:** >80% per agent
- **Retraining Frequency:** <Weekly per agent
- **System Uptime:** >99.5%
- **Data Quality:** >99% clean data
- **Response Time:** <100ms for predictions

### User Experience Targets
- **Setup Time:** <5 minutes
- **Trade Execution:** <30 seconds
- **Dashboard Load Time:** <3 seconds
- **User Satisfaction:** >4.5/5
- **Support Response:** <24 hours
