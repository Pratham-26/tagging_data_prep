from __future__ import annotations

"""Pydantic models for representing hierarchical label structures.

Provides LabelNode for individual nodes in the tree and LabelHierarchy
for the complete label tree with navigation and validation utilities.
"""
from pydantic import BaseModel, model_validator
from typing import Optional


class LabelNode(BaseModel):
    id: str
    description: str = ""
    children: list[LabelNode] = []

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_child_ids(self) -> list[str]:
        return [child.id for child in self.children]

    def get_child_descriptions(self) -> dict[str, str]:
        return {child.id: child.description for child in self.children}

    def get_child(self, child_id: str) -> Optional[LabelNode]:
        for child in self.children:
            if child.id == child_id:
                return child
        return None


class LabelHierarchy(BaseModel):
    root: LabelNode

    def get_node(self, path: list[str]) -> Optional[LabelNode]:
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
