from __future__ import annotations

"""Level classifier module for single-level text classification using DSPy.

Provides LevelClassifier which wraps dspy.Predict to classify text against candidate labels
at a single hierarchy level, with retry logic and ClassificationError for failed predictions.
"""

import json
from dataclasses import dataclass

from ..prompts import PromptLoader


@dataclass
class ClassificationError(Exception):
    text: str
    attempted_index: int | None
    valid_candidates: list[str]
    retries_used: int


class LevelClassifier:
    def __init__(self, prompts_path: str | None = None):
        self.prompt_loader = PromptLoader(prompts_path)

    def classify(
        self,
        text: str,
        candidates: list[str],
        descriptions: dict[str, str],
        max_retries: int = 3,
        parent_path: list[str] | None = None,
    ) -> tuple[str, int]:
        descriptions_json = json.dumps(descriptions, ensure_ascii=False)
        attempted_index = None
        predictor = self.prompt_loader.get_predictor(parent_path or [])

        for attempt in range(max_retries):
            result = predictor(
                text=text,
                candidate_labels=candidates,
                label_descriptions=descriptions_json,
            )

            attempted_index = result.predicted_index

            if isinstance(attempted_index, int) and 0 <= attempted_index < len(candidates):
                return candidates[attempted_index], attempt

        raise ClassificationError(
            text=text,
            attempted_index=attempted_index,
            valid_candidates=candidates,
            retries_used=max_retries,
        )

    def classify_multiple(
        self,
        texts: list[str],
        candidates: list[str],
        descriptions: dict[str, str],
        max_retries: int = 3,
        parent_path: list[str] | None = None,
    ) -> list[tuple[str | None, int, ClassificationError | None]]:
        results = []
        for text in texts:
            try:
                label, retries = self.classify(
                    text, candidates, descriptions, max_retries, parent_path
                )
                results.append((label, retries, None))
            except ClassificationError as e:
                results.append((None, e.retries_used, e))
        return results
