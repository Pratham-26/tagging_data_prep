from __future__ import annotations

from src.schemas.labels import LabelHierarchy, LabelNode
from src.validation import validate_hierarchy


class TestValidateHierarchy:
    def test_validate_hierarchy_returns_list(self, simple_hierarchy):
        result = validate_hierarchy(simple_hierarchy)
        assert isinstance(result, list)

    def test_validate_hierarchy_delegates(self):
        hierarchy = LabelHierarchy(
            root=LabelNode(
                id="__root__",
                children=[
                    LabelNode(
                        id="cat_a",
                        children=[
                            LabelNode(id="dup"),
                            LabelNode(id="dup"),
                        ],
                    ),
                ],
            )
        )

        errors = validate_hierarchy(hierarchy)
        direct_errors = hierarchy.validate_structure()
        assert errors == direct_errors
