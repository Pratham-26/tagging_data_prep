# AGENTS.md

Coding agent instructions for this repository.

## Project Overview

DSPy-based hierarchical text classifier. Processes text through multi-level label hierarchies using sequential level-by-level classification with checkpointing support.

## Build/Lint/Test Commands

```bash
# Install dependencies
uv sync --all-extras

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_schemas.py

# Run a single test function
uv run pytest tests/test_schemas.py::test_label_node_is_leaf

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific test marker (if configured)
uv run pytest -m "not slow"

# Linting
uv run ruff check src tests

# Format code
uv run ruff format src tests

# Format check (without modifying)
uv run ruff format --check src tests
```

## Code Style Guidelines

### Imports

```python
# Order: stdlib -> third-party -> local (separated by blank lines)
from __future__ import annotations  # Always first if used

import json
from pathlib import Path
from typing import Optional

import dspy
from pydantic import BaseModel

from .schemas.labels import LabelHierarchy, LabelNode
from ..validation import validate_hierarchy
```

- Use `from __future__ import annotations` for files with type hints referencing own class
- Use relative imports within the package (`from ..schemas`, `from .module`)
- Group imports: stdlib, blank line, third-party, blank line, local
- Import specific names, avoid `from module import *`

### Formatting

- Line length: 100 characters
- Use `ruff format` for auto-formatting
- No trailing whitespace
- Files end with newline

### Type Annotations

```python
# Use modern union syntax (Python 3.10+)
def load(path: str | Path) -> LabelHierarchy: ...

# Use lowercase generics (Python 3.9+)
def get_children(self) -> list[LabelNode]: ...
def get_descriptions(self) -> dict[str, str]: ...

# Optional for clarity when needed
from typing import Optional
def get_child(self, id: str) -> Optional[LabelNode]: ...

# Use tuple for fixed structures
def classify(self) -> tuple[str, int]: ...
```

### Naming Conventions

```python
# Classes: PascalCase
class LabelNode(BaseModel): ...
class HierarchicalClassifier: ...

# Functions/methods/variables: snake_case
def get_child_descriptions(self) -> dict[str, str]: ...
completed_paths: dict[int, list[str]] = {}

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3
DEFAULT_CHECKPOINT_DIR = "checkpoints"

# Private methods: leading underscore
def _run_classification(self, state: BatchState) -> list[ClassificationResult]: ...

# Module-level exports via __all__
__all__ = ["LabelNode", "LabelHierarchy"]
```

### Pydantic Models

```python
class LabelNode(BaseModel):
    id: str
    description: str = ""
    children: list[LabelNode] = []

    def is_leaf(self) -> bool:
        return len(self.children) == 0
```

- Fields without defaults come first
- Use `= []` and `= {}` for mutable defaults (Pydantic handles safely)
- Methods follow fields

### Enums

```python
from enum import Enum

class ClassificationStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"
```

- Inherit from `str, Enum` for JSON-serializable enums

### Error Handling

```python
from dataclasses import dataclass

@dataclass
class ClassificationError(Exception):
    text: str
    attempted_label: str
    valid_candidates: list[str]
    retries_used: int
```

- Use dataclass-style exceptions with relevant context
- Raise with descriptive messages: `raise ValueError(f"Invalid hierarchy: {errors}")`
- Return errors as values when appropriate (see `classify_batch` return type)

### Docstrings

```python
class ClassifyLevel(dspy.Signature):
    """Classify text into exactly one of the provided categories."""

def get_node(self, path: list[str]) -> Optional[LabelNode]:
    """Get node at given path. Empty path returns root."""
```

- One-line docstrings for simple functions
- Use triple double quotes
- No comments in code bodies per project preference

### Module Structure

```python
# __init__.py pattern
from .labels import LabelNode, LabelHierarchy
from .classification import ClassificationStatus, ClassificationResult

__all__ = [
    "LabelNode",
    "LabelHierarchy",
    "ClassificationStatus",
    "ClassificationResult",
]
```

- Re-export public API from `__init__.py`
- Use explicit `__all__` lists

### DSPy Signatures

```python
class ClassifyLevel(dspy.Signature):
    """Classify text into exactly one of the provided categories."""

    text: str = dspy.InputField(desc="Text to classify")
    candidate_labels: list[str] = dspy.InputField(desc="List of possible label IDs")
    predicted_label: str = dspy.OutputField(desc="The predicted label ID")
```

- Class docstring explains signature purpose
- Use `desc` parameter for field descriptions
- Type annotations on fields

## Testing Guidelines

```python
# tests/conftest.py for shared fixtures
import pytest
from src.schemas import LabelHierarchy, LabelNode

@pytest.fixture
def sample_hierarchy() -> LabelHierarchy:
    return LabelHierarchy(
        root=LabelNode(
            id="__root__",
            children=[LabelNode(id="category_a", children=[])]
        )
    )

# Test file naming: test_<module>.py
# Test function naming: test_<function>_<scenario>
def test_get_node_returns_none_for_invalid_path(sample_hierarchy):
    result = sample_hierarchy.get_node(["nonexistent"])
    assert result is None
```

## Project Structure

```
src/
├── schemas/          # Pydantic models (LabelNode, ClassificationResult)
├── signatures/       # DSPy signatures (ClassifyLevel)
├── modules/          # DSPy modules (LevelClassifier, HierarchicalClassifier)
├── validation/       # Validation logic
├── loader.py         # Load hierarchy from JSON
└── config.py         # DSPy LM configuration
```
