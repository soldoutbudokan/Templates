# CLAUDE.md - Project Guidelines

## Language Preferences

- **Data analysis tasks should be done in R where possible** - prefer R for statistical analysis, data manipulation, and visualization
- Python is preferred for ML/AI projects, web scraping, and automation
- TypeScript/React (Next.js) for frontend web applications

## Project Documentation

- **When creating new folders/projects, always create a README.md** and update it consistently when making changes
- README should include a Quick Start section
- Document major changes as they happen

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

## Code Organization

- Separate concerns: data prep, training, inference in different files
- Keep files focused on single responsibility
- For ML projects: `game.py`, `agent.py`, `trainer.py`, `viz.py` pattern
