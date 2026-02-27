from __future__ import annotations

"""
Pydantic models for tracking hierarchical classification state and results.

Contains ClassificationStatus enum for result states, ClassificationResult for
individual classification outcomes, and BatchState for managing batch processing
with checkpointing support.
"""

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class ClassificationStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"


class ClassificationResult(BaseModel):
    text: str
    path: list[str]
    status: ClassificationStatus
    failed_at_level: int | None = None
    retry_count: int = 0


class BatchState(BaseModel):
    results: list[ClassificationResult] = []
    pending_texts: list[str] = []
    completed_paths: dict[int, list[str]] = {}
    current_level: int = 0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BatchState:
        data = json.loads(Path(path).read_text())
        return cls.model_validate(data)

    def get_in_progress_texts(self) -> list[tuple[int, str, list[str]]]:
        """Returns list of (original_idx, text, current_path) for texts in progress."""
        result = []
        for idx, path in self.completed_paths.items():
            if idx < len(self.pending_texts):
                result.append((idx, self.pending_texts[idx], path))
        return result
