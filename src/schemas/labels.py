from __future__ import annotations

"""Pydantic models for representing hierarchical label structures.

Provides LabelNode for individual nodes in the tree and LabelHierarchy
for the complete label tree with navigation and validation utilities.
"""

from pydantic import BaseModel, PrivateAttr


class LabelNode(BaseModel):
    id: str
    description: str = ""
    children: list[LabelNode] = []
    _child_map: dict[str, LabelNode] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _) -> None:
        self._child_map = {child.id: child for child in self.children}

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_child_ids(self) -> list[str]:
        return [child.id for child in self.children]

    def get_child_descriptions(self) -> dict[str, str]:
        return {child.id: child.description for child in self.children}

    def get_child(self, child_id: str) -> LabelNode | None:
        return self._child_map.get(child_id)


class LabelHierarchy(BaseModel):
    root: LabelNode

    def get_node(self, path: list[str]) -> LabelNode | None:
        """Get node at given path. Empty path returns root."""
        current = self.root
        for label_id in path:
            child = current.get_child(label_id)
            if child is None:
                return None
            current = child
        return current

    def get_top_level_labels(self) -> list[LabelNode]:
        return self.root.children

    def validate_structure(self) -> list[str]:
        """Validate hierarchy, return list of error messages."""
        errors = []
        seen_ids: set[str] = set()

        def check_duplicates(node: LabelNode, path: list[str]) -> None:
            if node.id in seen_ids:
                errors.append(f"Duplicate ID '{node.id}' at path {path}")
            seen_ids.add(node.id)
            for child in node.children:
                check_duplicates(child, path + [child.id])

        if self.root.id != "__root__":
            errors.append("Root node must have id '__root__'")

        for child in self.root.children:
            check_duplicates(child, [child.id])

        return errors

    def get_all_leaf_paths(self) -> list[list[str]]:
        """Return paths to all leaf nodes in the hierarchy."""
        paths = []

        def traverse(node: LabelNode, current_path: list[str]) -> None:
            if node.is_leaf():
                paths.append(current_path)
            else:
                for child in node.children:
                    traverse(child, current_path + [child.id])

        for child in self.root.children:
            traverse(child, [child.id])

        return paths
