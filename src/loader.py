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
    data = json.loads(Path(path).read_text())

    if "labels" in data:
        root = LabelNode(
            id="__root__",
            description="",
            children=[LabelNode.model_validate(label) for label in data["labels"]],
        )
    else:
        root = LabelNode(
            id="__root__", description="", children=[LabelNode.model_validate(data)]
        )

    hierarchy = LabelHierarchy(root=root)

    errors = validate_hierarchy(hierarchy)
    if errors:
        raise ValueError(f"Invalid label hierarchy: {errors}")

    return hierarchy
