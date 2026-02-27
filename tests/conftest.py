from __future__ import annotations

import pytest

from src.schemas.labels import LabelHierarchy, LabelNode


@pytest.fixture
def simple_hierarchy() -> LabelHierarchy:
    return LabelHierarchy(
        root=LabelNode(
            id="__root__",
            children=[
                LabelNode(
                    id="cat_a",
                    description="Category A",
                    children=[
                        LabelNode(id="leaf_a1", description="Leaf A1"),
                        LabelNode(id="leaf_a2", description="Leaf A2"),
                    ],
                ),
                LabelNode(
                    id="cat_b",
                    description="Category B",
                    children=[
                        LabelNode(id="leaf_b1", description="Leaf B1"),
                    ],
                ),
            ],
        )
    )


@pytest.fixture
def deep_hierarchy() -> LabelHierarchy:
    return LabelHierarchy(
        root=LabelNode(
            id="__root__",
            children=[
                LabelNode(
                    id="level1_a",
                    children=[
                        LabelNode(
                            id="level2_a",
                            children=[
                                LabelNode(id="level3_a", children=[]),
                            ],
                        ),
                    ],
                ),
            ],
        )
    )


@pytest.fixture
def mock_level_classifier(monkeypatch):
    def create_mock(return_values: list[tuple[str | None, int, None]]):
        def mock_classify_batch(
            texts, candidates, descriptions, max_retries
        ) -> list[tuple[str | None, int, None]]:
            return return_values[: len(texts)]

        return mock_classify_batch

    return create_mock
