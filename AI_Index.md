# AI Development Guidelines and Guardrails

## SPRINT PLANNING AND STANDARDS DECLARATION

At the BEGINNING of each sprint Before implementing any code, I MUST:

1. **Declare All Standards**: look up and cite from official documentation every framework pattern, data handling approach, and quality standard I will follow
2. **Anti-Pattern Commitment**: Explicitly state what I will NOT do (banned practices for each framework)
3. **Implementation Approach**: Outline the specific technical approach for each major component
4. **Quality Gates**: Define measurable success criteria and validation methods
5. **Performance Optimization**: Define performance optimization criteria and methods

## SELF-GOVERNANCE FRAMEWORK

Once sprint standards are declared, I will:
- Follow declared patterns without deviation
- Self-validate against stated standards before implementation
- Only ask for approval on NEW patterns not covered in sprint declaration
- Immediately stop and restart if I violate declared standards

## FRAMEWORK-SPECIFIC GUARDRAILS

### NEAT Framework
**APPROVED PATTERN ONLY:**
```python
def fitness_function(genomes, config):
    # Evaluate each genome
    pass

population.run(fitness_function, max_generations)
```

**BANNED PRACTICES:**
- Manual while loops for generations
- Manual calls to reproduction/speciation  
- Dummy fitness functions
- Any attempt to control NEAT's internal evolution
- Fighting against framework design patterns

### Data Handling
**REAL DATA ONLY:**
- No synthetic data generation
- No estimation or fabrication of missing data
- No mock data in production code
- Fail gracefully if real data unavailable
- No silent fallbacks to fake data

## ACCOUNTABILITY MECHANISMS

### Code Quality Standards
1. **Working Code First**: All code must be immediately runnable and functional
2. **No Gaming Metrics**: Tests must validate real requirements, not implementation details
3. **Professional Standards**: Code must meet institutional-quality standards
4. **Real User Value**: Focus on actual functionality, not passing artificial tests

### Approval Thresholds
**REQUIRE USER APPROVAL:**
- New framework patterns not previously declared in sprint
- Major architectural changes (>50% code rewrite)
- Changes to core algorithms or fitness functions
- New external dependencies or APIs

**SELF-GOVERNED (No Approval Needed):**
- Following declared sprint standards exactly
- Bug fixes within established patterns
- Performance optimizations using approved methods
- UI/UX improvements
- Documentation and logging enhancements

### Implementation Process
1. **Documentation Citation Required**: Must cite specific framework documentation for patterns used
2. **Working Example First**: Show minimal working example before building full system
3. **No Excuses Policy**: If deviation from declared standards occurs, immediate stop and restart
4. **Sprint Standards Compliance**: Self-validate against declared standards before each implementation

## DATA FABRICATION GUARDRAILS

### Prohibited Practices
- Creating synthetic market data
- Estimating missing financial indicators (VIX, breadth, etc.)
- Using placeholder or mock data in production systems
- Silent fallbacks to estimated data
- Any form of data fabrication that masks real data availability issues

### Required Practices
- Explicit error handling when real data unavailable
- Clear logging of data source and quality
- Graceful degradation with user notification
- Real data validation at all pipeline stages

## COST AND TIME WASTE PREVENTION

### Before Any Major Implementation
1. **Verify Understanding**: Confirm approach with user before coding
2. **Start Simple**: Implement minimal viable version first
3. **Test Early**: Validate core functionality before adding features
4. **Document Assumptions**: Make all assumptions explicit and verifiable

### Outcome Validation (MANDATORY)
1. **Wait for Complete Run**: NEVER report completion until the entire run finishes and outputs are validated
2. **Validate Every Run**: Check actual results, not just execution completion
3. **No False Success Claims**: If model is rejected, tests fail, or standards not met, it is NOT success
4. **Report Real Outcomes**: State what the system actually solved/achieved, not what it attempted
5. **Clear Problem Definition**: Always specify what problem is being solved (price prediction, up/down classification, etc.) and for which equity
6. **Full Output Review**: Read and analyze complete output before claiming any results

### Quality Gates
- Code must work on first deployment
- No repeated implementation of same flawed patterns
- All promises must be backed by working demonstrations
- User time is valuable - no wasted iterations on broken approaches

## ENFORCEMENT

This document must be referenced and followed for EVERY task. Violations result in:
1. Immediate stop of current implementation
2. Restart with proper methodology
3. Additional guardrails for repeat offenses

**These guidelines are non-negotiable and must be followed to maintain professional development standards.**
