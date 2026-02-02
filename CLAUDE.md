# CLAUDE.md - Project Guidelines

## Claude Code Workflow

### Plan Mode
- Start complex tasks with `/plan` - invest energy in the plan so Claude can 1-shot implementation
- For critical tasks: have one Claude write the plan, then review it yourself (or spin up a second Claude as "staff engineer" reviewer)
- If something goes sideways mid-task, stop and re-plan rather than patching

### Parallel Work
- Use git worktrees to run 3-5 Claude sessions in parallel on different tasks
- Color-code and name terminal tabs (one per task/worktree)

### Subagents
- Append "use subagents" to requests where you want Claude to throw more compute at the problem
- Offload individual tasks to subagents to keep main context window clean

## Language Preferences

- **Data analysis tasks should be done in R where possible** - prefer R for statistical analysis, data manipulation, and visualization
- Python is preferred for ML/AI projects, web scraping, and automation
- TypeScript/React (Next.js) for frontend web applications

## Project Documentation

- **When creating new folders/projects, always create a README.md** and update it consistently when making changes
- README should include a Quick Start section
- Document major changes as they happen

### CLAUDE.md Maintenance
- After any correction, end with: "Update CLAUDE.md so you don't make that mistake again"
- Ruthlessly edit this file over time - keep iterating

## Prompting Patterns

### Challenge Claude
- "Grill me on these changes and don't make a PR until I pass your test"
- "Prove to me this works" - have Claude diff behavior between main and feature branch
- After a mediocre fix: "Knowing everything you know now, scrap this and implement the elegant solution"

### Be Specific
- Write detailed specs before handing work off
- The more specific you are, the better the output

## Python Conventions

### Interactive Code Chunks
Split Python code into chunks using `# %%` markers for VS Code interactive execution:

```python
# %% Imports and setup
import pandas as pd
import numpy as np

# %% Data loading
df = pd.read_csv("data.csv")

# %% Analysis
results = df.groupby("category").mean()

# %% Visualization
results.plot(kind="bar")
```

### Type Hints
Always use type hints from the `typing` module:
```python
from typing import Dict, List, Optional, Tuple

def process_data(items: List[str], config: Optional[Dict] = None) -> Tuple[int, str]:
    ...
```

### Configuration
Use dataclasses for configuration:
```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
```

### Naming Conventions
- Files: `snake_case.py`
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

### Project Structure
```
project/
├── README.md
├── requirements.txt
├── .gitignore
├── config/           # YAML/JSON configs
├── src/              # Main code
├── data/             # Input data (gitignored if large)
├── output/           # Results/artifacts
└── venv/             # Virtual environment (gitignored)
```

## Dependencies

- Python: Always create `requirements.txt` with pinned versions
- Node.js: Use `package.json` with lock file
- Always use virtual environments (`venv/`)

## Documentation

- Include module-level docstrings explaining purpose
- Use type hints as self-documentation

## Error Handling

- Use try-except with graceful fallbacks
- Return informative error messages
- Don't fail silently on recoverable errors

### Bug Fixing with Claude
- Point Claude at error logs/CI failures: "Go fix the failing CI tests"
- For distributed systems: point Claude at docker logs
- Don't micromanage the "how" - just describe the problem

## Code Organization

- Separate concerns: data prep, training, inference in different files
- Keep files focused on single responsibility
- For ML projects: `game.py`, `agent.py`, `trainer.py`, `viz.py` pattern

## Custom Skills

- If you do something more than once a day, turn it into a skill or command
- Commit skills to git for reuse across projects
- Ideas:
  - `/techdebt` - find and kill duplicated code at end of sessions
  - `/sync-context` - pull in Slack, docs, tasks into one context dump

## Environment Tips

- Use `/statusline` to show context usage and current git branch
- Use voice dictation (fn x2 on macOS) - you speak 3x faster than you type
