from __future__ import annotations

"""Level classifier module for single-level text classification using DSPy.

Provides LevelClassifier which wraps dspy.Predict to classify text against candidate labels
at a single hierarchy level, with retry logic and ClassificationError for failed predictions.
"""

import dspy
import json
from dataclasses import dataclass
from signatures import ClassifyLevel


@dataclass
class ClassificationError(Exception):
    text: str
    attempted_label: str
    valid_candidates: list[str]
    retries_used: int


class LevelClassifier:
    def __init__(self):
        self.predictor = dspy.Predict(ClassifyLevel)

    def classify(
        self,
        text: str,
        candidates: list[str],
        descriptions: dict[str, str],
        max_retries: int = 3,
    ) -> tuple[str, int]:
        descriptions_json = json.dumps(descriptions, ensure_ascii=False)
        predicted = None

        for attempt in range(max_retries):
            result = self.predictor(
                text=text,
                candidate_labels=candidates,
                label_descriptions=descriptions_json,
            )

            predicted = result.predicted_label.strip()

            if predicted in candidates:
                return predicted, attempt

        raise ClassificationError(
            text=text,
            attempted_label=predicted,
            valid_candidates=candidates,
            retries_used=max_retries,
        )

    def classify_batch(
        self,
        texts: list[str],
        candidates: list[str],
        descriptions: dict[str, str],
        max_retries: int = 3,
    ) -> list[tuple[str | None, int, ClassificationError | None]]:
        results = []
        for text in texts:
            try:
                label, retries = self.classify(
                    text, candidates, descriptions, max_retries
                )
                results.append((label, retries, None))
            except ClassificationError as e:
                results.append((None, e.retries_used, e))
        return results
