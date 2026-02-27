# DSPy Hierarchical Text Classifier

A hierarchical text classification system built with DSPy that processes text through multi-level label hierarchies using sequential level-by-level classification with GEPA-based prompt optimization.

## Features

- **Index-based output** - Uses integer indices instead of label strings to minimize token costs
- **Batch processing** - Automatic grouping by parent path for efficient classification
- **Checkpointing** - Resume interrupted classification runs from last saved state
- **Configurable retries** - Graceful handling of classification failures with fallback to partial paths
- **GEPA optimization** - Automatically optimize classification instructions per path
- **Path-specific prompts** - Different prompts for each level in the hierarchy

## Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Quick Start

### Basic Classification

```python
import dspy
from src.loader import load_hierarchy
from src.modules import HierarchicalClassifier

# Configure your LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Load label hierarchy
hierarchy = load_hierarchy("labels.json")

# Create classifier
classifier = HierarchicalClassifier(hierarchy, max_retries=3)

# Classify texts
texts = ["Sample text to classify", "Another text"]
results = classifier.classify_batch(texts)

for result in results:
    print(f"Text: {result.text}")
    print(f"Path: {' -> '.join(result.path)}")
    print(f"Status: {result.status}")
```

### Using Optimized Prompts

```python
from src.loader import load_hierarchy
from src.modules import HierarchicalClassifier
from src.gepa_pipeline.config import load_config

# Load config with optimized prompts path
config = load_config("config.yaml")

hierarchy = load_hierarchy(config.paths.hierarchy)
classifier = HierarchicalClassifier(
    hierarchy=hierarchy,
    prompts_path=config.paths.prompts  # Uses path-specific optimized prompts
)

results = classifier.classify_batch(["Text to classify"])
```

## CLI Usage

### Optimize Prompts

Run GEPA optimization to generate path-specific classification prompts:

```bash
uv run python main.py optimize --config config.yaml
```

This will:
1. Sample texts from your corpus
2. Label them using a high-quality LM
3. Run GEPA optimization for each unique path
4. Save optimized prompts to `prompts.json`

### Classify Texts

```bash
# Classify a single text
uv run python main.py classify --config config.yaml --text "Your text here"

# Classify texts from a file (one per line)
uv run python main.py classify --config config.yaml --input texts.txt
```

## Configuration

Create a `config.yaml` file:

```yaml
lms:
  optimizer:
    model: "openai/gpt-4o"
    api_key: "env:OPENAI_API_KEY"
  classifier:
    model: "openai/gpt-4o-mini"
    api_key: "env:OPENAI_API_KEY"

data:
  corpus_path: "./data/corpus.txt"
  sample_size: 100
  delimiter: null

gepa:
  max_metric_calls_per_path: 50
  run_dir: "./gepa_runs"

paths:
  hierarchy: "./labels.json"
  prompts: "./prompts.json"
```

### LM Configuration

- `optimizer` - High-quality LM for labeling training data and GEPA reflection
- `classifier` - LM for classification (used during optimization and inference)

API keys can be:
- Direct values: `api_key: "sk-..."`
- Environment variables: `api_key: "env:OPENAI_API_KEY"`

### Data Configuration

- `corpus_path` - Path to unlabeled text data (one per line or CSV)
- `sample_size` - Number of texts to sample for optimization
- `delimiter` - CSV delimiter (null for one text per line)

## Label Hierarchy Format

```json
{
  "root": {
    "id": "__root__",
    "description": "Root node",
    "children": [
      {
        "id": "category_a",
        "description": "First level category A",
        "children": [
          {
            "id": "subcategory_a1",
            "description": "Second level subcategory",
            "children": []
          }
        ]
      }
    ]
  }
}
```

## Optimized Prompts Format

After running optimization, `prompts.json` contains:

```json
{
  "[]": {
    "signature": {
      "instructions": "Optimized instruction for root level...",
      "fields": [...]
    },
    "demos": []
  },
  "[category_a]": {
    "signature": {
      "instructions": "Optimized instruction for category_a children...",
      "fields": [...]
    },
    "demos": []
  }
}
```

Keys are JSON arrays representing the parent path:
- `"[]"` - Root level (no parent)
- `"[category_a]"` - Children of category_a
- `"[category_a, subcategory_a1]"` - Children of subcategory_a1

## GEPA Optimization Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ corpus.txt  │    │ Subsample N │    │ Label with  │    │    GEPA     │
│ (unlabeled) │───▶│   random    │───▶│ high-quality│───▶│  optimize   │
│             │    │   texts     │    │     LM      │    │  per-path   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
                                                        ┌─────────────┐
                                                        │prompts.json │
                                                        │(per-path    │
                                                        │instructions)│
                                                        └─────────────┘
```

1. **Sample** - Random texts from your corpus
2. **Label** - Classify through full hierarchy using optimizer LM
3. **Decompose** - Break full paths into per-level training instances
4. **Optimize** - Run GEPA for each unique parent path
5. **Save** - Store optimized instructions in prompts.json

## Checkpointing

Resume interrupted classification runs:

```python
results = classifier.classify_batch(
    texts,
    checkpoint_path="checkpoints/batch_001.json"
)

# If interrupted, resume later
results = classifier.resume_from_checkpoint(
    "checkpoints/batch_001.json",
    new_texts=["Additional text"]  # Optional
)
```

## Architecture

```
src/
├── gepa_pipeline/
│   ├── __init__.py
│   ├── config.py              # YAML config loader
│   ├── sampler.py             # Corpus sampling
│   ├── labeler.py             # Label texts through hierarchy
│   ├── training_data.py       # Build per-path training instances
│   ├── adapter.py             # GEPA adapter for optimization
│   └── optimizer.py           # Orchestrate GEPA runs
├── prompts/
│   ├── __init__.py
│   ├── defaults.py            # Default instruction
│   └── loader.py              # Load prompts.json, create predictors
├── schemas/
│   ├── labels.py              # LabelNode, LabelHierarchy
│   └── classification.py      # ClassificationResult, BatchState
├── signatures/
│   └── classify.py            # ClassifyLevel DSPy signature
├── modules/
│   ├── level_classifier.py    # Single-level classification
│   └── hierarchical_classifier.py  # Multi-level orchestration
├── validation/
│   └── label_validator.py     # Hierarchy validation
├── loader.py                  # Load hierarchy from JSON
└── config.py                  # LM configuration
```

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing

# Linting
uv run ruff check src

# Format code
uv run ruff format src
```

## License

MIT
