# DSPy Hierarchical Text Classifier

## 1. Project Overview

A hierarchical text classification system built with DSPy that processes text through a multi-level label hierarchy. The classifier uses **sequential level-by-level processing** where each level's predictions inform the next, enabling efficient caching and batch optimization.

**Key Features:**
- Single reusable `ClassifyLevel` signature for all hierarchy levels
- Batch processing with automatic grouping by parent path
- Checkpointing for resume-from-failure capability
- Configurable retry strategies with fallback to partial paths
- Strict validation of label hierarchy structure

---

## 2. Project Structure

```
dspy-classifier/
├── src/
│   ├── __init__.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── labels.py              # LabelNode, LabelHierarchy (Pydantic)
│   │   └── classification.py      # ClassificationResult, BatchState
│   ├── signatures/
│   │   ├── __init__.py
│   │   └── classify.py            # ClassifyLevel signature (reusable)
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── level_classifier.py    # Wraps dspy.Predict for single level
│   │   └── hierarchical_classifier.py  # Orchestrates batch flow
│   ├── validation/
│   │   ├── __init__.py
│   │   └── label_validator.py     # Validate label JSON structure
│   ├── loader.py                  # Load LabelHierarchy from JSON
│   └── config.py                  # LM setup (placeholder)
├── labels.json                    # Example hierarchy
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_signatures.py
│   ├── test_level_classifier.py
│   ├── test_hierarchical_classifier.py
│   └── test_validation.py
├── examples/
│   ├── basic_usage.py
│   └── batch_usage.py
├── pyproject.toml
└── README.md
```

---

## 3. Core Schemas

### 3.1 LabelNode

Represents a single node in the label hierarchy.

```python
from pydantic import BaseModel, Field
from typing import Optional

class LabelNode(BaseModel):
    """A node in the label hierarchy tree."""
    id: str = Field(..., description="Unique identifier for this label")
    name: str = Field(..., description="Human-readable label name")
    description: str = Field(..., description="Description for LLM context")
    children: list["LabelNode"] = Field(default_factory=list)
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def find_by_id(self, label_id: str) -> Optional["LabelNode"]:
        """Recursively find a node by ID."""
        ...
    
    def get_children_of(self, parent_id: str) -> list["LabelNode"]:
        """Get all children of a given parent node."""
        ...
```

### 3.2 LabelHierarchy

The complete label tree with lookup utilities.

```python
class LabelHierarchy(BaseModel):
    """Complete label hierarchy with efficient lookup."""
    root: LabelNode
    max_depth: int = Field(..., description="Maximum depth of the tree")
    
    def get_level_labels(self, level: int, parent_path: list[str]) -> list[LabelNode]:
        """Get candidate labels for a specific level given parent path."""
        ...
    
    def get_path_to_node(self, label_id: str) -> list[str]:
        """Get the full path from root to a node."""
        ...
```

### 3.3 ClassificationStatus

Enum for tracking classification state.

```python
from enum import Enum

class ClassificationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Completed with partial path due to failure
```

### 3.4 ClassificationResult

Result for a single text classification.

```python
class ClassificationResult(BaseModel):
    """Result of classifying a single text."""
    text_id: str
    text: str
    status: ClassificationStatus
    path: list[str] = Field(default_factory=list)  # List of label IDs from root to leaf
    confidence_scores: dict[int, float] = Field(default_factory=dict)  # level -> score
    error: Optional[str] = None
    attempts: int = 0
```

### 3.5 BatchState

State for checkpointing and resuming batch jobs.

```python
class BatchState(BaseModel):
    """Checkpointable state for batch processing."""
    batch_id: str
    total_items: int
    completed_items: int = 0
    current_level: int = 0
    results: dict[str, ClassificationResult] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    
    def get_pending_for_level(self, level: int) -> list[ClassificationResult]:
        """Get items pending classification at a specific level."""
        ...
    
    def to_json(self) -> str:
        """Serialize for checkpointing."""
        ...
    
    @classmethod
    def from_json(cls, json_str: str) -> "BatchState":
        """Deserialize from checkpoint."""
        ...
```

---

## 4. Signature Design

### ClassifyLevel

A **single reusable signature** for all hierarchy levels. This design enables DSPy caching across levels.

```python
import dspy

class ClassifyLevel(dspy.Signature):
    """Classify text into one of the provided candidate labels.
    
    Given a text and a list of candidate labels with their descriptions,
    select the most appropriate label.
    """
    text: str = dspy.InputField(desc="The text to classify")
    parent_path: list[str] = dspy.InputField(desc="Path of parent labels from root")
    candidate_labels: list[tuple[str, str]] = dspy.InputField(
        desc="List of (label_id, description) tuples to choose from"
    )
    predicted_label: str = dspy.OutputField(desc="The ID of the predicted label")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1")
```

**Why a single signature?**
- Enables DSPy's built-in caching mechanism
- Simplifies the codebase (no signature-per-level)
- Consistent prompt structure across levels
- Easier to optimize with DSPy teleprompters

---

## 5. Modules

### 5.1 LevelClassifier

Wraps `dspy.Predict` for classifying at a single level.

```python
class LevelClassifier:
    """Classifies texts at a single hierarchy level."""
    
    def __init__(self, max_retries: int = 3):
        self.predictor = dspy.Predict(ClassifyLevel)
        self.max_retries = max_retries
    
    def classify(
        self,
        text: str,
        parent_path: list[str],
        candidates: list[LabelNode]
    ) -> tuple[str, float]:
        """
        Classify text into one of the candidate labels.
        
        Returns:
            Tuple of (predicted_label_id, confidence_score)
        
        Raises:
            ClassificationError: After max_retries exhausted
        """
        ...
    
    def classify_batch(
        self,
        items: list[tuple[str, list[str], list[LabelNode]]]
    ) -> list[tuple[str, float]]:
        """Batch classification with shared context."""
        ...
```

### 5.2 HierarchicalClassifier

Orchestrates the full classification flow across all levels.

```python
class HierarchicalClassifier:
    """Orchestrates hierarchical classification across all levels."""
    
    def __init__(
        self,
        hierarchy: LabelHierarchy,
        level_classifier: LevelClassifier,
        checkpoint_dir: Optional[Path] = None
    ):
        self.hierarchy = hierarchy
        self.classifier = level_classifier
        self.checkpoint_dir = checkpoint_dir
    
    def classify(self, text: str, text_id: str) -> ClassificationResult:
        """Classify a single text through all hierarchy levels."""
        ...
    
    def classify_batch(
        self,
        texts: list[tuple[str, str]],  # [(text_id, text), ...]
        batch_id: Optional[str] = None,
        resume: bool = False
    ) -> BatchState:
        """
        Classify a batch of texts with checkpointing support.
        
        Process level-by-level, grouping items by parent path for efficiency.
        """
        ...
    
    def _process_level(
        self,
        state: BatchState,
        level: int
    ) -> BatchState:
        """Process all items at a single level."""
        ...
```

---

## 6. Batch Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BATCH PROCESSING FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: 100 texts to classify
       Hierarchy: 3 levels (L1 → L2 → L3)

┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: All texts start with empty parent_path                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Group: parent_path=[]                                                      │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │  Text 1  ─┐                                                           │  │
│   │  Text 2  ─┼──▶ ClassifyLevel(parent_path=[], candidates=[L1_labels]) │  │
│   │  ...     ─┤                                                           │  │
│   │  Text 100─┘                                                           │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   Results:                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Text 1  → L1_A      Text 51 → L1_A                                  │   │
│   │ Text 2  → L1_A      Text 52 → L1_B                                  │   │
│   │ Text 3  → L1_B      ...                                             │   │
│   │ ...               Text 100 → L1_C                                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Checkpoint saved ◀──────────────────────────────────────────────────────  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 2: Group texts by their L1 prediction                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Group: parent_path=[L1_A]                                                  │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  Text 1  ─┐                                                         │    │
│   │  Text 2  ─┼──▶ ClassifyLevel(parent_path=[L1_A], candidates=...)   │    │
│   │  Text 51 ─┘                                                         │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   Group: parent_path=[L1_B]                                                  │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  Text 3  ─┐                                                         │    │
│   │  Text 4  ─┴──▶ ClassifyLevel(parent_path=[L1_B], candidates=...)   │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   Group: parent_path=[L1_C]                                                  │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  Text 100 ──▶ ClassifyLevel(parent_path=[L1_C], candidates=...)    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   Checkpoint saved ◀──────────────────────────────────────────────────────  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 3: Group texts by their L1+L2 path                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Group: parent_path=[L1_A, L2_X]                                            │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  Text 1  ─┐                                                         │    │
│   │  Text 51 ─┴──▶ ClassifyLevel(parent_path=[L1_A, L2_X], candidates=.)│   │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   Group: parent_path=[L1_A, L2_Y]                                            │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  Text 2  ──▶ ClassifyLevel(parent_path=[L1_A, L2_Y], candidates=.) │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   ... (similar groups for other paths)                                       │
│                                                                              │
│   Checkpoint saved ◀──────────────────────────────────────────────────────  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT: ClassificationResult for each text with full path                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Text 1  → [L1_A, L2_X, L3_1]  ✓ COMPLETED                                  │
│   Text 2  → [L1_A, L2_Y, L3_2]  ✓ COMPLETED                                  │
│   Text 3  → [L1_B, L2_Z]        ⚠ PARTIAL (failed at L3, fell back)         │
│   ...                                                                        │
│   Text 100 → [L1_C, L2_W, L3_9] ✓ COMPLETED                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Points:**
1. **Level-by-level**: Complete all items at level N before moving to level N+1
2. **Grouping**: Items with same parent_path are processed together
3. **Checkpointing**: State saved after each level completes
4. **Caching benefit**: Same signature + similar contexts = better DSPy caching

---

## 7. Checkpointing

### Strategy

Batch processing saves state after each level completion:

```python
# Checkpoint file naming
checkpoint_file = checkpoint_dir / f"{batch_id}_level_{level}.json"
```

### Checkpoint Structure

```json
{
  "batch_id": "batch_20240115_143022",
  "total_items": 100,
  "completed_items": 100,
  "current_level": 3,
  "results": {
    "text_001": {
      "text_id": "text_001",
      "text": "Original text...",
      "status": "completed",
      "path": ["L1_A", "L2_X", "L3_1"],
      "confidence_scores": {"0": 0.95, "1": 0.87, "2": 0.92}
    },
    "text_002": {
      "text_id": "text_002",
      "status": "partial",
      "path": ["L1_B", "L2_Z"],
      "error": "Max retries exceeded at level 2"
    }
  },
  "created_at": "2024-01-15T14:30:22Z",
  "updated_at": "2024-01-15T14:45:33Z"
}
```

### Resume Flow

```python
def resume_batch(batch_id: str) -> BatchState:
    # Find latest checkpoint
    checkpoints = sorted(checkpoint_dir.glob(f"{batch_id}_level_*.json"))
    latest = checkpoints[-1]
    
    # Load state
    state = BatchState.from_json(latest.read_text())
    
    # Resume from next level
    return classifier.classify_batch(
        texts=remaining_texts,
        batch_id=batch_id,
        resume=True
    )
```

---

## 8. Retry Strategy

### Configuration

```python
@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0
    exponential_base: float = 2.0
    fallback_to_partial: bool = True
```

### Retry Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    RETRY FLOW (per item)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Attempt #1     │
                    │  ClassifyLevel  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
         Success                       Failure
              │                             │
              ▼                             ▼
        Return result              Wait (exponential backoff)
                                           │
                                           ▼
                                 ┌─────────────────┐
                                 │  Attempt #2     │
                                 └────────┬────────┘
                                          │
                           ┌──────────────┴──────────────┐
                           │                             │
                      Success                       Failure
                           │                             │
                           ▼                             ▼
                     Return result         Continue until max_retries
                                                         │
                                                         ▼
                                              ┌─────────────────────┐
                                              │ max_retries reached │
                                              └──────────┬──────────┘
                                                         │
                                    ┌────────────────────┴────────────────────┐
                                    │                                         │
                           fallback_to_partial=True              fallback_to_partial=False
                                    │                                         │
                                    ▼                                         ▼
                        Return partial path                        Raise ClassificationError
                        status=PARTIAL                             status=FAILED
```

### Fallback Behavior

When `fallback_to_partial=True`:
- Return the path accumulated so far
- Mark status as `PARTIAL`
- Include error message in result
- Continue processing other items

---

## 9. Validation

### Label Hierarchy Validation Rules

```python
class LabelValidator:
    """Validates label hierarchy structure."""
    
    @staticmethod
    def validate(hierarchy: LabelHierarchy) -> list[str]:
        """
        Validate hierarchy and return list of errors.
        Returns empty list if valid.
        """
        errors = []
        
        # Rule 1: All IDs must be unique
        # Rule 2: No empty names or descriptions
        # Rule 3: Consistent depth (all paths to leaf should be same length)
        # Rule 4: At least 2 levels required
        # Rule 5: Each level must have at least 1 label
        # Rule 6: Leaf nodes must be reachable
        
        return errors
```

### Validation Rules

| Rule | Description | Severity |
|------|-------------|----------|
| Unique IDs | All `id` fields must be globally unique | Error |
| Non-empty names | `name` cannot be empty or whitespace | Error |
| Non-empty descriptions | `description` required for LLM context | Error |
| Minimum depth | Hierarchy must have at least 2 levels | Error |
| Consistent depth | All leaf nodes should be at the same depth | Warning |
| Reachable leaves | All leaf nodes must be reachable from root | Error |
| ID format | IDs should match `^[a-zA-Z0-9_-]+$` | Warning |

### Usage

```python
from src.validation import LabelValidator
from src.loader import load_hierarchy

hierarchy = load_hierarchy("labels.json")
errors = LabelValidator.validate(hierarchy)

if errors:
    print("Validation failed:")
    for error in errors:
        print(f"  - {error}")
```

---

## 10. Dependencies

### pyproject.toml

```toml
[project]
name = "dspy-classifier"
version = "0.1.0"
description = "Hierarchical text classification with DSPy"
requires-python = ">=3.10"

dependencies = [
    "dspy-ai>=2.5.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

### Dependency Purposes

| Package | Purpose |
|---------|---------|
| `dspy-ai` | Core DSPy framework for LM programming |
| `pydantic` | Data validation and schema definition |
| `pytest` | Testing framework |
| `pytest-asyncio` | Async test support |
| `pytest-cov` | Coverage reporting |
| `ruff` | Fast linter and formatter |
| `mypy` | Static type checking |

---

## Example Usage

### Basic Classification

```python
from src.loader import load_hierarchy
from src.modules import LevelClassifier, HierarchicalClassifier

# Load hierarchy
hierarchy = load_hierarchy("labels.json")

# Setup classifier
level_classifier = LevelClassifier(max_retries=3)
classifier = HierarchicalClassifier(
    hierarchy=hierarchy,
    level_classifier=level_classifier
)

# Classify single text
result = classifier.classify(
    text="Customer complaint about delayed shipping",
    text_id="query_001"
)

print(f"Path: {' → '.join(result.path)}")
print(f"Status: {result.status}")
```

### Batch Classification with Checkpointing

```python
from pathlib import Path

classifier = HierarchicalClassifier(
    hierarchy=hierarchy,
    level_classifier=level_classifier,
    checkpoint_dir=Path("./checkpoints")
)

# Prepare batch
texts = [
    ("doc_001", "First document text..."),
    ("doc_002", "Second document text..."),
    # ... 1000 more
]

# Run with checkpointing
state = classifier.classify_batch(
    texts=texts,
    batch_id="daily_batch_20240115"
)

# If interrupted, resume later:
# state = resume_batch("daily_batch_20240115")

# Access results
for text_id, result in state.results.items():
    if result.status == "completed":
        print(f"{text_id}: {result.path}")
```
