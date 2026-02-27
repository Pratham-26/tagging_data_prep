from __future__ import annotations

"""Load and validate label hierarchies from JSON files.

Provides the load_hierarchy function for parsing JSON configuration files
into LabelHierarchy objects with automatic validation of the hierarchy structure.
"""

import json
from pathlib import Path

from .schemas.labels import LabelHierarchy, LabelNode
from .validation import validate_hierarchy


def load_hierarchy(path: str | Path) -> LabelHierarchy:
    path = Path(path)
    try:
        raw = path.read_text()
    except FileNotFoundError:
        raise ValueError(f"Label hierarchy file not found: {path}")
    except OSError as e:
        raise ValueError(f"Failed to read label hierarchy file '{path}': {e}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in label hierarchy file '{path}': {e}")

    if "labels" in data:
        root = LabelNode(
            id="__root__",
            description="",
            children=[LabelNode.model_validate(label) for label in data["labels"]],
        )
    else:
        root = LabelNode(id="__root__", description="", children=[LabelNode.model_validate(data)])

    hierarchy = LabelHierarchy(root=root)

    errors = validate_hierarchy(hierarchy)
    if errors:
        raise ValueError(f"Invalid label hierarchy: {errors}")

    return hierarchy
