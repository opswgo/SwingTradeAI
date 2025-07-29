# SwingTradeAI Code Violations Report

## 1. NEAT Implementation Violations

### high_performance_neat_system.py:
- **Violation**: Implements manual control of NEAT evolution with custom loops
- **Line ~239-320**: Contains parallel fitness evaluation that circumvents NEAT's built-in evolution 
- **AI_Index Violation**: Directly violates "BANNED PRACTICES: Manual while loops for generations, Manual calls to reproduction/speciation, Fighting against framework design patterns"

### practical_neat_system.py:
- **Violation**: Fails to follow the single approved NEAT pattern
- **Line ~249-270**: Uses a broken implementation as noted in the @deprecated comment ("Training takes <1 second, produces 46% win rate models")
- **Line ~313**: Uses `population.run(fitness_wrapper, 1)` in a for-loop instead of a single call with all generations
- **AI_Index Violation**: "APPROVED PATTERN ONLY: population.run(fitness_function, max_generations)"

## 2. Synthetic Data Violations

### proper_validation_charts.py:
- **Violation**: Uses entirely synthetic data for validation
- **Line 29-30**: `np.random.seed(42)` indicates synthetic data generation
- **Line 51-56**: Explicitly generates fake NEAT predictions with `base_trend = 0.5 + 0.3 * np.sin(i / 20)` and `noise = np.random.normal(0, 0.1)`
- **AI_Index Violation**: "DATA FABRICATION GUARDRAILS: No synthetic market data, Any form of data fabrication that masks real data availability issues"

### apy_improvement_plan.md:
- **Violation**: Contains contradictory performance metrics
- **Line 1-2**: Claims current performance is 10.2% APY while targeting 100%+ APY
- **Line 179-183**: Reports current win rate as 97.1% but current assignment rate as 25.7% (impossible to have both)
- **AI_Index Violation**: "OUTCOME VALIDATION REQUIREMENTS: Verify calculations and check for unrealistic results"

## 3. Quality Standard Violations

### high_performance_neat_system.py:
- **Violation**: Code appears sophisticated but likely doesn't deliver real functionality
- **Line ~17-46**: Implements complex GPU acceleration code that's unnecessary for the task
- **AI_Index Violation**: "WORKING CODE IS PARAMOUNT: All code must use real data only and be world class, production ready"

### practical_neat_system.py:
- **Violation**: Self-identified as broken but still included in codebase
- **Line 19**: `@deprecated # BROKEN: Training takes <1 second, produces 46% win rate models`
- **AI_Index Violation**: "Code Quality Standards: Working Code First - All code must be immediately runnable and functional"

## 4. Validation/Testing Violations

### proper_validation_charts.py:
- **Violation**: Creates misleading validation charts with fabricated data
- **Line ~184-296**: Generates elaborate charts that simulate success without real data validation
- **AI_Index Violation**: "Model Validation Requirements: Validate all models against real market data" and "No False Success Claims"

## 5. Overall System Integrity Issues

- Multiple implementations of similar functionality with inconsistent approaches
- Presence of known broken code throughout the repository
- Fake validation results and contradictory performance metrics
- No evidence of the system working with real market data

This report documents clear violations of the strict development guidelines in AI_Index.md, particularly around synthetic data use, improper NEAT implementation, and misleading validation. These issues directly contradict the user's explicit requirements as documented in the provided memories.
