"""Label hierarchy validation module.

Provides validation functions to verify the structural integrity of label hierarchies,
ensuring they conform to expected format and constraints before use in classification.
"""

from ..schemas.labels import LabelHierarchy


def validate_hierarchy(hierarchy: LabelHierarchy) -> list[str]:
    return hierarchy.validate_structure()
