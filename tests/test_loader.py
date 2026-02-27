from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.loader import load_hierarchy


class TestLoadHierarchy:
    def test_load_with_labels_key(self):
        json_content = '{"labels": [{"id": "cat_a", "children": []}]}'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            hierarchy = load_hierarchy(temp_path)
            assert hierarchy.root.id == "__root__"
            assert len(hierarchy.root.children) == 1
            assert hierarchy.root.children[0].id == "cat_a"
        finally:
            Path(temp_path).unlink()

    def test_load_without_labels_key(self):
        json_content = '{"id": "cat_a", "children": []}'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            hierarchy = load_hierarchy(temp_path)
            assert hierarchy.root.id == "__root__"
            assert len(hierarchy.root.children) == 1
            assert hierarchy.root.children[0].id == "cat_a"
        finally:
            Path(temp_path).unlink()

    def test_load_valid_file(self, simple_hierarchy, tmp_path):
        json_path = tmp_path / "hierarchy.json"
        json_path.write_text(
            '{"labels": [{"id": "cat_a", "description": "Category A", "children": []}]}'
        )

        hierarchy = load_hierarchy(json_path)
        assert hierarchy.root.id == "__root__"
        assert hierarchy.root.children[0].id == "cat_a"

    def test_load_invalid_json(self):
        json_content = "{ invalid json }"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            with pytest.raises(Exception):
                load_hierarchy(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_invalid_structure(self):
        json_content = '{"labels": [{"id": "dup"}, {"id": "dup"}]}'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid label hierarchy"):
                load_hierarchy(temp_path)
        finally:
            Path(temp_path).unlink()
