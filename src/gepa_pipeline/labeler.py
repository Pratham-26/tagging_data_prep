from __future__ import annotations

"""Corpus labeling through hierarchical classification.

Provides CorpusLabeler which uses a high-quality LM to label texts
through the full label hierarchy, producing LabeledExample instances.
"""

from dataclasses import dataclass

import dspy

from ..modules.hierarchical_classifier import HierarchicalClassifier
from ..schemas.classification import ClassificationStatus
from ..schemas.labels import LabelHierarchy


@dataclass
class LabeledExample:
    text: str
    full_path: list[str]


class CorpusLabeler:
    def __init__(self, hierarchy: LabelHierarchy, lm: dspy.LM):
        self.hierarchy = hierarchy
        self.lm = lm

    def label_texts(self, texts: list[str]) -> list[LabeledExample]:
        """Label texts through full hierarchy using high-quality LM."""
        dspy.configure(lm=self.lm)
        classifier = HierarchicalClassifier(self.hierarchy, max_retries=3)
        results = classifier.classify_batch(texts)

        labeled: list[LabeledExample] = []
        for result in results:
            if result.status == ClassificationStatus.SUCCESS:
                labeled.append(LabeledExample(text=result.text, full_path=result.path))

        return labeled
