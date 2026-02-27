"""DSPy signature for single-level text classification.

This module defines the ClassifyLevel signature that takes text and candidate labels
with their descriptions, and predicts the most appropriate label for the input text.
"""

import dspy


class ClassifyLevel(dspy.Signature):
    """Classify text into exactly one of the provided categories."""

    text: str = dspy.InputField(desc="Text to classify")
    candidate_labels: list[str] = dspy.InputField(desc="List of possible label IDs")
    label_descriptions: str = dspy.InputField(
        desc="JSON string mapping label ID to description"
    )
    predicted_label: str = dspy.OutputField(
        desc="The predicted label ID, must be exactly one from candidate_labels"
    )
