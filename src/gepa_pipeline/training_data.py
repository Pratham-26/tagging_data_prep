from __future__ import annotations

"""Training data construction for per-path optimization.

Decomposes labeled examples into per-path training instances,
where each instance represents a single classification decision
at a specific point in the hierarchy.
"""

from dataclasses import dataclass

from ..schemas.labels import LabelHierarchy
from .labeler import LabeledExample


@dataclass
class PathTrainingInstance:
    text: str
    expected_label: str
    candidates: list[str]
    descriptions: dict[str, str]


def build_training_data(
    examples: list[LabeledExample],
    hierarchy: LabelHierarchy,
) -> dict[tuple[str, ...], list[PathTrainingInstance]]:
    """
    Decompose labeled examples into per-path training instances.

    For path [A, B, C], create instances for:
    - key=() with expected=A
    - key=(A,) with expected=B
    - key=(A, B) with expected=C
    """
    training_data: dict[tuple[str, ...], list[PathTrainingInstance]] = {}

    for example in examples:
        path = example.full_path

        for level, expected_label in enumerate(path):
            parent_path = tuple(path[:level])
            node = hierarchy.get_node(list(parent_path))

            if node is None:
                continue

            child = node.get_child(expected_label)
            if child is None:
                continue

            instance = PathTrainingInstance(
                text=example.text,
                expected_label=expected_label,
                candidates=node.get_child_ids(),
                descriptions=node.get_child_descriptions(),
            )

            if parent_path not in training_data:
                training_data[parent_path] = []
            training_data[parent_path].append(instance)

    return training_data
