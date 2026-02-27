from __future__ import annotations

from src.schemas.labels import LabelHierarchy, LabelNode


class TestLabelNode:
    def test_is_leaf_true(self):
        node = LabelNode(id="test", children=[])
        assert node.is_leaf() is True

    def test_is_leaf_false(self):
        node = LabelNode(id="test", children=[LabelNode(id="child")])
        assert node.is_leaf() is False

    def test_get_child_ids(self, simple_hierarchy):
        ids = simple_hierarchy.root.get_child_ids()
        assert ids == ["cat_a", "cat_b"]

    def test_get_child_descriptions(self, simple_hierarchy):
        descriptions = simple_hierarchy.root.get_child_descriptions()
        assert descriptions == {"cat_a": "Category A", "cat_b": "Category B"}

    def test_get_child_found(self, simple_hierarchy):
        child = simple_hierarchy.root.get_child("cat_a")
        assert child is not None
        assert child.id == "cat_a"

    def test_get_child_not_found(self, simple_hierarchy):
        child = simple_hierarchy.root.get_child("nonexistent")
        assert child is None


class TestLabelHierarchy:
    def test_get_node_empty_path(self, simple_hierarchy):
        node = simple_hierarchy.get_node([])
        assert node is simple_hierarchy.root

    def test_get_node_valid_path(self, simple_hierarchy):
        node = simple_hierarchy.get_node(["cat_a", "leaf_a1"])
        assert node is not None
        assert node.id == "leaf_a1"

    def test_get_node_invalid_path(self, simple_hierarchy):
        node = simple_hierarchy.get_node(["cat_a", "nonexistent"])
        assert node is None

    def test_get_top_level_labels(self, simple_hierarchy):
        labels = simple_hierarchy.get_top_level_labels()
        assert len(labels) == 2
        assert labels[0].id == "cat_a"
        assert labels[1].id == "cat_b"

    def test_validate_valid(self, simple_hierarchy):
        errors = simple_hierarchy.validate_structure()
        assert errors == []

    def test_validate_duplicate_ids(self):
        hierarchy = LabelHierarchy(
            root=LabelNode(
                id="__root__",
                children=[
                    LabelNode(
                        id="cat_a",
                        children=[
                            LabelNode(id="duplicate"),
                            LabelNode(id="duplicate"),
                        ],
                    ),
                ],
            )
        )
        errors = hierarchy.validate_structure()
        assert len(errors) == 1
        assert "Duplicate ID" in errors[0]
