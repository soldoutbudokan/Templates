# RikkenBot Improvement Proposal

**Generated**: 2025-10-27
**Status**: Comprehensive Analysis & Roadmap

---

## Executive Summary

RikkenBot is a reinforcement learning project for the Dutch card game Rikken. While the infrastructure for training and analysis is solid, **the core game engine is currently a mock implementation** that doesn't enforce real Rikken rules. This proposal outlines a comprehensive path to transform RikkenBot into a production-quality RL environment.

---

## Current State Analysis

### ✅ Strengths

1. **Solid Training Infrastructure**
   - PPO-based RL with Stable-Baselines3
   - Action masking support
   - Both maskable and recurrent (LSTM) policy options
   - Comprehensive evaluation framework

2. **Excellent Analysis Tools**
   - Automated strategy extraction
   - Performance visualization (win rates, contract analysis, trump preferences)
   - Strategy guides generation
   - JSON and text output formats

3. **Good Code Structure**
   - Clean separation between environment, training, and analysis
   - Gymnasium-compliant API
   - Type hints and dataclasses

### ❌ Critical Issues

#### 1. **Mock Game Logic** (CRITICAL)
The `rikclaude.py` environment explicitly states it's "NOT full game rules":

```python
# From rikclaude.py:1-3
# Minimal, self-contained Rikken-like environment with the API expected by the trainer/analyzer.
# This is a lightweight mock (NOT full game rules) so you can train/run the pipeline end-to-end.
# Replace this with your real environment later, but keep the same attributes/methods.
```

**Problems:**
- Trick winners chosen randomly with bias (lines 200-214)
- No suit-following enforcement
- Simplified trump mechanics
- No real point-based scoring (just trick counting)
- Random partner assignment instead of bidding-based
- Tricks capped at 32 plays instead of full deck (52 cards / 4 players = 13 tricks)

#### 2. **Missing Game Rules Implementation**
Real Rikken requires:
- Proper trick-taking (must follow suit, trump hierarchy)
- Complex bidding system (Rik, Troela, Solo variants, Misère, Piek)
- Point-based scoring (not just trick counting)
- Partner selection in team contracts
- Blind/widow card management
- Proper contract fulfillment conditions

#### 3. **No Dependency Management**
- No `requirements.txt` or `environment.yml`
- Dependencies scattered in comments
- Version requirements unclear

#### 4. **Code Duplication**
- `RikkenOut.py` appears to duplicate analyzer functionality from `rikken_rl_train.py`

#### 5. **No Testing**
- No unit tests for game logic
- No integration tests
- High risk of bugs in rule implementation

#### 6. **Incomplete Documentation**
- No README specific to RikkenBot
- No explanation of Rikken rules for non-Dutch speakers
- No setup instructions

---

## Proposed Solution: Phased Improvement Plan

### Phase 1: Foundation & Cleanup (Week 1)

**Goal:** Establish proper project infrastructure

#### Tasks:
1. **Dependency Management**
   - Create `requirements.txt` with pinned versions
   - Document Python version requirement
   - Add optional dev dependencies (pytest, black, mypy)

2. **Project Documentation**
   - Create comprehensive README.md
   - Document Rikken rules in English
   - Add architecture diagram
   - Setup and installation guide

3. **Code Cleanup**
   - Remove or clarify `RikkenOut.py` duplication
   - Add proper module docstrings
   - Organize code into packages

4. **Testing Framework**
   - Setup pytest
   - Create test structure
   - Add basic environment sanity tests

**Deliverables:**
- `requirements.txt`
- `RikkenBot/README.md`
- `RikkenBot/RULES.md`
- `tests/` directory structure
- Clean, deduplicated codebase

---

### Phase 2: Core Game Engine (Week 2-3)

**Goal:** Implement real Rikken rules

#### Tasks:

1. **Card & Deck Management**
   - Proper 32-card deck (7-8-9-10-J-Q-K-A in 4 suits)
   - Card comparison with trump/suit-following rules
   - Hand evaluation utilities

2. **Trick-Taking Engine**
   - Enforce suit following
   - Implement trump logic
   - Proper trick winner determination
   - Lead player rotation

3. **Bidding System**
   - Sequential bidding with pass/bid options
   - Contract hierarchy validation
   - Partner selection for Rik/Troela
   - Blind/widow handling

4. **Scoring System**
   - Point values (Ace=11, Ten=10, King=4, Queen=3, Jack=2)
   - Contract-specific win conditions
   - Misère/Piek special rules
   - Team scoring

5. **Game State Management**
   - Phase transitions (deal → bid → play → score)
   - Legal action generation per phase
   - Proper terminal state detection

**Deliverables:**
- `rikken_engine.py` (new comprehensive engine)
- `tests/test_engine.py` (comprehensive unit tests)
- Backward-compatible API with current wrapper
- Validation against hand-coded game examples

---

### Phase 3: Enhanced Observations (Week 4)

**Goal:** Improve state representation for better learning

#### Current Issues:
- Observations are minimal (just hand + trump + seat + phase)
- No memory of played cards
- No bidding history encoding
- Limited information about opponents

#### Improvements:

1. **Rich Observations**
   - One-hot encoding of full game state
   - Cards played this trick
   - All cards seen in previous tricks
   - Bidding history (who bid what)
   - Current contract details
   - Trick winners so far
   - Point totals

2. **Observation Versions**
   - Minimal (current, for baseline)
   - Standard (enhanced but compact)
   - Full (everything the player can legally see)
   - Options for partial observability

3. **Observation Space Documentation**
   - Clear documentation of each feature
   - Dimension calculations
   - Serialization format

**Deliverables:**
- Refactored observation builder
- Multiple observation configs
- Unit tests for observation consistency
- Documentation of feature engineering choices

---

### Phase 4: Training Improvements (Week 5)

**Goal:** Enhance training for better convergence

#### Current Issues:
- Single shared policy may not learn seat-specific strategies
- No curriculum learning
- Limited reward shaping
- Short episodes due to mock (32 plays vs 52 cards)

#### Improvements:

1. **Reward Shaping**
   - Dense rewards (per trick, per card played)
   - Bidding bonuses/penalties
   - Contract fulfillment rewards
   - Optional auxiliary rewards (e.g., high-card points)

2. **Training Strategies**
   - Self-play with policy diversity
   - Curriculum learning (start with simple contracts)
   - Opponent modeling options
   - Prioritized experience replay for rare events

3. **Hyperparameter Tuning**
   - Systematic HPO (learning rate, entropy coefficient, etc.)
   - Multiple PPO variants (MaskablePPO, RecurrentPPO)
   - A/B testing framework

4. **Experiment Tracking**
   - Weights & Biases integration
   - TensorBoard logs
   - Checkpoint management
   - Reproducibility tools (seed management)

**Deliverables:**
- Enhanced training script with reward shaping
- Curriculum learning implementation
- W&B/TensorBoard integration
- Hyperparameter search scripts
- Training best practices documentation

---

### Phase 5: Advanced Features (Week 6+)

**Goal:** Add competitive play and evaluation tools

#### Features:

1. **Human-Play Interface**
   - CLI for human vs. AI games
   - Move validation and hints
   - Game state visualization
   - Replay saving/loading

2. **Tournament Mode**
   - Multiple agents compete
   - ELO rating system
   - Round-robin tournaments
   - Knockout brackets

3. **Strategy Comparison**
   - Baseline strategies (random, rule-based heuristics)
   - Historical agent snapshots
   - Transfer learning experiments

4. **Advanced Analysis**
   - Opening move analysis
   - Bidding strategy clustering
   - Contract success prediction
   - Counterfactual analysis (what-if scenarios)

5. **Web Interface** (Optional)
   - Browser-based play
   - Live training visualization
   - Strategy explorer

**Deliverables:**
- CLI play interface
- Tournament system
- Enhanced analyzer tools
- Optional web UI

---

## Technical Architecture

### Proposed Structure

```
RikkenBot/
├── README.md                   # Main documentation
├── RULES.md                    # Rikken rules explanation
├── requirements.txt            # Dependencies
├── setup.py                    # Package installation
│
├── rikken/                     # Main package
│   ├── __init__.py
│   ├── engine/                 # Core game logic
│   │   ├── __init__.py
│   │   ├── cards.py           # Card, Deck, Hand classes
│   │   ├── contracts.py       # Contract types and validation
│   │   ├── tricks.py          # Trick-taking logic
│   │   ├── bidding.py         # Bidding system
│   │   └── scoring.py         # Point calculation
│   │
│   ├── env/                    # RL environment
│   │   ├── __init__.py
│   │   ├── rikken_env.py      # Main Gymnasium environment
│   │   ├── observations.py    # Observation builders
│   │   └── wrappers.py        # Utility wrappers
│   │
│   ├── agents/                 # RL and baseline agents
│   │   ├── __init__.py
│   │   ├── ppo_agent.py       # PPO training
│   │   ├── baselines.py       # Random, heuristic agents
│   │   └── policy_configs.py  # Policy architectures
│   │
│   ├── analysis/               # Analysis tools
│   │   ├── __init__.py
│   │   ├── analyzer.py        # Game analyzer
│   │   ├── visualizer.py      # Plotting utilities
│   │   └── strategy_extraction.py
│   │
│   └── cli/                    # Command-line interface
│       ├── __init__.py
│       ├── play.py            # Human play
│       └── tournament.py      # Tournament mode
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_engine/
│   ├── test_env/
│   └── test_agents/
│
├── configs/                    # Configuration files
│   ├── training/
│   └── experiments/
│
├── scripts/                    # Utility scripts
│   ├── train.py
│   ├── evaluate.py
│   └── benchmark.py
│
├── output/                     # Generated files (gitignored)
│   ├── models/
│   ├── logs/
│   └── analysis/
│
└── docs/                       # Extended documentation
    ├── architecture.md
    ├── training_guide.md
    └── api_reference.md
```

---

## Risk Assessment

### High Priority Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Rule implementation bugs** | Critical - trained agents would learn incorrect strategies | Extensive unit testing, hand-verified game examples, peer review by Rikken players |
| **Performance regression** | High - real rules may slow training significantly | Profile code, optimize hot paths, consider Cython for bottlenecks |
| **API breaking changes** | Medium - existing trained models become incompatible | Maintain v1 mock API alongside v2, provide migration guide |
| **Scope creep** | Medium - project becomes overwhelming | Stick to phased plan, defer optional features to Phase 5+ |

### Medium Priority Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Rikken rule ambiguity** | Some rules may be unclear or have regional variants | Document assumptions, make configurable where possible |
| **Training time increase** | Real games take longer than mock | Optimize environment, use vectorized envs, consider GPU acceleration |
| **Hyperparameter sensitivity** | May need retuning after rule changes | Systematic HPO, document baseline configs |

---

## Success Metrics

### Phase 1-2 (Foundation + Engine)
- ✅ All unit tests passing (>90% coverage for engine)
- ✅ Can play full legal game end-to-end
- ✅ Hand-verified games match expected outcomes
- ✅ Documentation complete and clear

### Phase 3-4 (Observations + Training)
- ✅ Agents learn to win >60% vs random baseline
- ✅ Agents learn to bid strategically (not just pass)
- ✅ Training converges within 1M timesteps
- ✅ Agents can play all contract types

### Phase 5 (Advanced Features)
- ✅ Human players rate AI as "competent" (survey)
- ✅ Top agents win >70% vs rule-based heuristics
- ✅ Distinct strategies emerge for different contracts
- ✅ Analysis reveals interpretable patterns

---

## Resource Requirements

### Time Estimate
- **Phase 1**: 1-2 weeks (20-30 hours)
- **Phase 2**: 2-3 weeks (40-60 hours) - Most complex
- **Phase 3**: 1 week (15-20 hours)
- **Phase 4**: 1-2 weeks (20-30 hours)
- **Phase 5**: Open-ended (optional enhancements)

**Total Core Development**: 6-8 weeks (100-140 hours)

### Technical Requirements
- Python 3.10+
- PyTorch / Stable-Baselines3
- ~8GB RAM for training
- GPU recommended but not required
- ~1GB disk space for checkpoints

### Knowledge Requirements
- Deep understanding of Rikken rules (consult Dutch players?)
- RL fundamentals (PPO, self-play)
- Python software engineering
- Basic ML experiment tracking

---

## Recommendations

### Immediate Next Steps (This Week)

1. **Validate this proposal** with domain experts
   - Do you understand Rikken rules well enough?
   - Do you have access to Rikken players for validation?
   - Is the scope reasonable for your goals?

2. **Start Phase 1** if approved:
   - Create requirements.txt
   - Write RULES.md (forces rule clarity)
   - Setup testing framework
   - Clean up current code

3. **Create minimal test cases**
   - Write down 3-5 example games by hand
   - Document expected outcomes
   - Use these as integration tests

### Decision Points

**Before Phase 2:**
- Confirm rule interpretations
- Decide on Rikken variant (32-card vs 52-card? Regional rules?)
- Approve new architecture

**Before Phase 3:**
- Evaluate Phase 2 performance
- Decide on observation complexity
- Review Phase 1-2 learnings

**Before Phase 4:**
- Baseline agent performance acceptable?
- Training infrastructure stable?
- Ready for longer training runs?

---

## Alternatives Considered

### Alternative 1: Incremental Rule Addition
**Approach:** Keep mock, gradually add real rules
**Pros:** Less disruptive, continuous progress
**Cons:** Risk of half-implemented rules, harder to validate, technical debt
**Verdict:** ❌ Not recommended - clean rewrite better

### Alternative 2: Use Existing Card Game Framework
**Approach:** Build on libraries like RLCard or OpenSpiel
**Pros:** Faster development, proven infrastructure
**Cons:** Less control, may not support Rikken complexity, learning curve
**Verdict:** 🤔 Worth exploring if timeline is critical

### Alternative 3: Focus on Mock, Defer Real Rules
**Approach:** Continue with mock, focus on RL techniques
**Pros:** Can experiment with training methods now
**Cons:** Strategies won't transfer, limited real-world value
**Verdict:** ❌ Defeats project purpose

---

## Conclusion

RikkenBot has **excellent infrastructure** but needs a **real game engine** to be valuable. The proposed phased approach:

1. ✅ Minimizes risk through incremental development
2. ✅ Maintains working code at each phase
3. ✅ Allows early exit if priorities change
4. ✅ Builds foundation for advanced features

**Recommended action:** Approve Phase 1-2, reassess after engine implementation.

---

## Questions for Discussion

1. **Rule Validation:** Do you have access to experienced Rikken players for rule validation?

2. **Scope:** Is the full 6-8 week timeline acceptable, or should we focus on minimal viable engine?

3. **Priority:** Are you more interested in:
   - Strong playing AI (focus on training)
   - Strategy analysis (focus on interpretability)
   - Human play (focus on interface)

4. **Rikken Variant:** Which exact rule set? (Regional variations exist)

5. **Timeline:** Prefer quick MVP or comprehensive implementation?

---

**Next Steps:** Review this proposal, answer questions above, and I can begin implementation of your chosen path.
