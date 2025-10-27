# RikkenBot Quick Start Guide

## TL;DR - What's Wrong and How to Fix It

### Current Problem
```
❌ rikclaude.py is a MOCK environment (not real Rikken rules)
   - Trick winners = random + bias
   - No suit-following enforcement
   - No real scoring
   - Simplified everything
```

### The Fix
```
✅ Implement real Rikken game engine
✅ Keep the excellent training/analysis infrastructure
✅ Add proper testing
✅ Document everything
```

---

## Critical Issues Prioritized

### 🔴 P0 - Blockers
1. **Mock game engine** - Current environment doesn't play real Rikken
   - **Impact:** AI learns wrong strategies
   - **Fix:** Phase 2 (implement real engine)

### 🟡 P1 - Major Issues
2. **No dependency management** - Missing requirements.txt
   - **Impact:** Can't reproduce environment
   - **Fix:** 30 minutes

3. **No tests** - Zero test coverage
   - **Impact:** Can't trust rule implementation
   - **Fix:** Phase 1-2 (setup + implement)

4. **Missing docs** - No README, no rule explanations
   - **Impact:** Hard for others to contribute
   - **Fix:** Phase 1 (2-3 hours)

### 🟢 P2 - Nice to Have
5. **Code duplication** - RikkenOut.py duplicates analyzer
6. **Limited observations** - Could be richer for better learning
7. **No human play mode** - Can't test against AI

---

## If You Have 30 Minutes: Quick Wins

### 1. Create requirements.txt
```bash
cd /home/user/Templates/RikkenBot

cat > requirements.txt << 'EOF'
# Core RL
gymnasium>=0.29.0
stable-baselines3>=2.3.0
sb3-contrib>=2.3.0
torch>=2.0.0

# Data & Analysis
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Development
pytest>=7.4.0
black>=23.0.0
mypy>=1.5.0
EOF

echo "✅ Created requirements.txt"
```

### 2. Setup Testing Framework
```bash
mkdir -p tests/test_engine tests/test_env tests/test_agents
touch tests/__init__.py

cat > tests/test_engine/__init__.py << 'EOF'
# Engine tests
EOF

cat > tests/conftest.py << 'EOF'
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
EOF

echo "✅ Test structure created"
```

### 3. Add Basic README
```bash
cat > README.md << 'EOF'
# RikkenBot

Reinforcement Learning agent for the Dutch card game Rikken.

## Status: In Development

⚠️ **Current environment is a MOCK** - does not implement full Rikken rules yet.
See `IMPROVEMENT_PROPOSAL.md` for roadmap.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Train (using mock environment)
python rikken_rl_train.py --timesteps 100000

# Evaluate
python rikken_rl_train.py --timesteps 100000 --episodes 500
```

## Documentation

- `IMPROVEMENT_PROPOSAL.md` - Comprehensive improvement plan
- `QUICKSTART.md` - This file
- `output/` - Training results and analysis

## License

[Your License]
EOF

echo "✅ Created README.md"
```

---

## If You Have 2 Hours: Phase 1 Foundation

### Checklist
- [ ] Create requirements.txt (5 min)
- [ ] Install dependencies (10 min)
- [ ] Setup testing (15 min)
- [ ] Write README.md (20 min)
- [ ] Document Rikken rules in RULES.md (60 min)
- [ ] Clean up RikkenOut.py duplication (10 min)

### Script to Run All Quick Wins
```bash
cd /home/user/Templates/RikkenBot

# 1. Dependencies
cat > requirements.txt << 'EOF'
gymnasium>=0.29.0
stable-baselines3>=2.3.0
sb3-contrib>=2.3.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.4.0
black>=23.0.0
mypy>=1.5.0
EOF

# 2. Install (optional, may take time)
# pip install -r requirements.txt

# 3. Testing structure
mkdir -p tests/test_engine tests/test_env tests/test_agents
touch tests/__init__.py tests/conftest.py

# 4. Verify current code runs
python rikken_rl_train.py --timesteps 10000 --episodes 50

echo "✅ Phase 1 quick setup complete!"
echo "Next: Write RULES.md to clarify Rikken rules"
```

---

## If You Have 1 Week: Minimal Viable Engine

### Goal
Get RikkenBot playing **legal** Rikken games (even if not smart yet)

### Day 1-2: Rule Documentation
1. Write comprehensive RULES.md
2. Document all contracts, scoring, trick-taking rules
3. Create 3 hand-written example games with expected outcomes
4. Get validation from Rikken players

### Day 3-5: Core Engine
1. Implement Card/Deck classes (proper 32-card Rikken deck)
2. Implement trick-taking with suit-following
3. Implement bidding system
4. Implement scoring

### Day 6-7: Integration & Testing
1. Wrap engine in Gymnasium environment
2. Write comprehensive unit tests
3. Validate against hand-written games
4. Test with existing training code

### Deliverables
- `rikken/engine/` package with real rules
- `tests/` with >80% coverage
- Backward-compatible API
- Verified correct game play

---

## Recommended Path Forward

### Option A: Full Implementation (Recommended)
**Timeline:** 6-8 weeks
**Outcome:** Production-quality Rikken RL environment

1. Week 1: Foundation (Phase 1)
2. Weeks 2-3: Core Engine (Phase 2)
3. Week 4: Enhanced Observations (Phase 3)
4. Week 5: Training Improvements (Phase 4)
5. Week 6+: Advanced Features (Phase 5)

**Best for:** Serious research, publishable work, reusable framework

### Option B: Minimal Viable Engine
**Timeline:** 1-2 weeks
**Outcome:** Correct but basic Rikken environment

1. Days 1-2: RULES.md + test cases
2. Days 3-5: Core engine implementation
3. Days 6-7: Testing + integration

**Best for:** Quick validation, learning RL with real rules

### Option C: Mock Iteration
**Timeline:** Ongoing
**Outcome:** Experiment with RL techniques on mock environment

Continue using current mock, focus on:
- Different PPO configurations
- Reward shaping experiments
- Observation engineering
- Analysis improvements

**Best for:** RL technique experimentation, not Rikken-specific

---

## Critical Questions to Answer

Before starting, decide:

1. **Do you know Rikken rules well enough to implement them?**
   - If NO: Find Rikken expert or detailed rulebook first
   - If YES: Document in RULES.md before coding

2. **What's your goal?**
   - Strong Rikken AI → Full implementation
   - Learn RL techniques → Mock is OK for now
   - Research project → Full implementation
   - Personal project → Minimal viable

3. **How much time can you commit?**
   - 1 week → Minimal viable engine
   - 6-8 weeks → Full implementation
   - Ongoing part-time → Phased approach

4. **Need it to work with existing models?**
   - YES → Maintain API compatibility carefully
   - NO → Clean rewrite easier

---

## Getting Help

### If Stuck on Rikken Rules
- Consult: https://nl.wikipedia.org/wiki/Rikken_(kaartspel)
- Ask Dutch friends/family who play
- Check regional variations
- Document assumptions when unclear

### If Stuck on RL Implementation
- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/
- Gymnasium docs: https://gymnasium.farama.org/
- Similar projects: RLCard, OpenSpiel

### If Stuck on Architecture
- Review: IMPROVEMENT_PROPOSAL.md
- Look at: OpenSpiel's game implementations
- Pattern: Separate engine from RL wrapper

---

## What NOT to Do

❌ **Don't** start coding the engine without documented rules
❌ **Don't** skip testing - rule bugs are insidious
❌ **Don't** try to implement everything at once
❌ **Don't** optimize before it works correctly
❌ **Don't** train for long runs on mock environment

✅ **Do** write RULES.md first
✅ **Do** create hand-verified test games
✅ **Do** implement incrementally with tests
✅ **Do** validate with Rikken players
✅ **Do** use phased approach

---

## Next Steps

1. **Read IMPROVEMENT_PROPOSAL.md** (this covers everything in detail)
2. **Decide your path** (Option A, B, or C above)
3. **Answer critical questions** (above section)
4. **Start Phase 1** (or minimal engine)
5. **Commit to git** as you go
6. **Ask for help** when stuck

---

## Quick Reference: File Purposes

| File | Purpose | Status |
|------|---------|--------|
| `rikclaude.py` | Environment (MOCK) | ⚠️ Not real rules |
| `rikken_rl_train.py` | Training + Analysis | ✅ Good infrastructure |
| `RikkenOut.py` | Analysis (old?) | ⚠️ Duplicate? |
| `output/` | Results | ✅ Working |
| `rikken_ppo.zip` | Trained model | ⚠️ Based on mock |

**After Phase 2:**
| File | Purpose | Status |
|------|---------|--------|
| `rikken/engine/` | Real game engine | 🎯 To be created |
| `tests/` | Unit tests | 🎯 To be created |
| `RULES.md` | Rule documentation | 🎯 To be created |
| `requirements.txt` | Dependencies | 🎯 To be created |

---

**Ready to start?** Pick your path and let's build a real Rikken AI!
