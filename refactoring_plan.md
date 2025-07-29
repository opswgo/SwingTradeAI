# SwingTradeAI Refactoring Plan

## Overview
This document tracks the compliance assessment and refactoring decisions for the SwingTradeAI project based on the strict standards defined in AI_Index.md. The primary goal is to identify compliant components that can be preserved and rebuild non-compliant components according to proper patterns.

## Directory Organization Strategy
- `src/`: Target architecture for verified compliant code
- `src_review/`: Temporary directory for files under assessment
- `forReference/`: Files with useful patterns but requiring substantial refactoring
- `junk/`: Non-compliant files that violate core standards

## Compliance Criteria (Based on AI_Index.md)
1. **NEAT Implementation**: Must use proper `population.run(fitness_function, max_generations)` pattern with no manual evolution loops
2. **Data Usage**: Only real market data, absolutely no synthetic/mock data
3. **Validation**: Proper validation with meaningful business metrics (not gamed/fabricated)
4. **Error Handling**: Graceful failure on insufficient data rather than fallbacks to synthetic data
5. **Documentation**: Clear purpose, usage, and performance expectations

## Known Violations (Already Moved to junk/)
- **high_performance_neat_system.py**: Uses manual evolution loops instead of proper NEAT pattern
- **practical_neat_system.py**: Manual NEAT evolution violating framework design
- **proper_validation_charts.py**: Uses synthetic data generation
- **fixed_validation_charts.py**: Contains fabricated validation
- **generate_validation_charts.py**: Creates misleading performance charts
- **All pre-trained models**: Based on improper NEAT implementation

## Directory Assessment

### Root Directory
**Status**: Triaged into appropriate categories
- **Moved to junk/**: NEAT trainer implementations, validation dashboards, model generators
- **Moved to forReference/**: Interface files, dashboard templates
- **Moved to root_review/**: Potentially useful components (data fetchers, trading advisors, agent files)
- **Retained in root/**: Documentation, configuration files, shell scripts

**Action**: Files in root_review/ require detailed assessment; only essential non-code files kept in root directory.

### src/trading/
**Status**: Most promising for reuse
- **pnl_tracker.py**: Potentially compliant tracking system for P&L
- **risk_manager.py**: Position sizing and risk management that may be compliant
- **trade_logger.py**: Logging system that appears to use real data

**Action**: Detailed assessment needed, but likely candidates for preservation.

### src/models/
**Status**: Needs major refactoring
- **regime_model.py**: Contains sophisticated ML, but needs validation for synthetic data usage
- **backtesting.py**: Requires review for data source integrity

**Action**: Extract useful patterns but plan for rebuild following proper NEAT patterns.

### src/features/
**Status**: Potentially useful
- **feature_store.py**: Infrastructure may be sound, but data sources need validation
- **market_regime.py**: Feature engineering potentially useful if no synthetic data
- **options_chain.py**: Options chain handling needs validation for real data
- **technical_analysis.py**: Technical indicators may be reusable

**Action**: Review for real data usage, preserve if compliant.

### src/data/
**Status**: Critical foundation component
- **pipeline.py**: Data pipeline architecture may be reusable if no synthetic data
- **postgres_manager.py**: Database connection may be sound

**Action**: Verify real data sources, ensure no synthetic fallbacks.

### src/backtesting/
**Status**: Requires careful assessment
- **advanced_backtester.py**: May contain useful backtesting infrastructure
- **model_competition.py**: Framework for comparing models could be useful
- **strategy_reporter.py**: Reporting structure may be reusable

**Action**: Check for synthetic data, fabricated results, and proper validation.

### src/database/
**Status**: Likely reusable
- **agent_database.py**: Database schema and connections likely sound

**Action**: Preserve database infrastructure after verifying no synthetic data storage.

### src/api/, src/core/, src/dashboard/, src/utils/
**Status**: Supporting infrastructure
- Likely reusable with minimal changes if data sources are proper

**Action**: Verify real data usage, preserve infrastructure components.

### data/ and models/ Directories
**Status**: Content evaluation needed
- No Python files found directly in these directories
- May contain data files and serialized models

**Action**: Verify real data, discard any synthetic data or models trained on it.

## Next Steps

### 1. NEAT Implementation Rebuild (Highest Priority)
- Create proper NEAT implementation following official pattern
- Use `population.run(fitness_function, max_generations)` as single call
- Implement proper fitness function evaluating one genome at a time

### 2. Data Validation (Critical)
- Verify all data sources use real market data
- Remove any synthetic data generation or fallbacks
- Implement proper error handling for insufficient data cases

### 3. Component Review Process
- Assess each component in src_review against strict compliance criteria
- Determine which components can be preserved as-is
- Identify components requiring moderate refactoring
- Mark components needing complete rebuild

### 4. New Clean Architecture
- Implement new src/ structure with verified components
- Create clean documentation of architecture and data flows
- Establish proper testing with real data only

This refactoring plan will be continuously updated as assessment progresses.
