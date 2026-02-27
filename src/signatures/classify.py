"""DSPy signature for single-level text classification.

This module defines the ClassifyLevel signature that takes text and candidate labels
with their descriptions, and predicts the most appropriate label for the input text.
"""

import dspy


class ClassifyLevel(dspy.Signature):
    """Classify text into exactly one of the provided categories by index."""

    text: str = dspy.InputField(desc="Text to classify")
    candidate_labels: list[str] = dspy.InputField(desc="List of possible label IDs")
    label_descriptions: str = dspy.InputField(desc="JSON string mapping label ID to description")
    predicted_index: int = dspy.OutputField(
        desc="Index (0-based) of the predicted label in candidate_labels"
    )
