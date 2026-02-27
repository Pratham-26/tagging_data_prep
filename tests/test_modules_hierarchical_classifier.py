from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from src.modules.hierarchical_classifier import HierarchicalClassifier
from src.modules.level_classifier import ClassificationError
from src.schemas.classification import ClassificationStatus


class TestHierarchicalClassifier:
    def test_single_text_to_leaf(self, simple_hierarchy):
        classifier = HierarchicalClassifier(simple_hierarchy)

        with patch.object(
            classifier.level_classifier,
            "classify_multiple",
            side_effect=[
                [("cat_a", 0, None)],
                [("leaf_a1", 0, None)],
            ],
        ):
            results = classifier.classify_batch(["sample text"])

        assert len(results) == 1
        assert results[0].text == "sample text"
        assert results[0].path == ["cat_a", "leaf_a1"]
        assert results[0].status == ClassificationStatus.SUCCESS

    def test_single_text_multi_level(self, deep_hierarchy):
        classifier = HierarchicalClassifier(deep_hierarchy)

        with patch.object(
            classifier.level_classifier,
            "classify_multiple",
            side_effect=[
                [("level1_a", 0, None)],
                [("level2_a", 0, None)],
                [("level3_a", 0, None)],
            ],
        ):
            results = classifier.classify_batch(["sample text"])

        assert len(results) == 1
        assert results[0].path == ["level1_a", "level2_a", "level3_a"]
        assert results[0].status == ClassificationStatus.SUCCESS

    def test_multiple_texts(self, simple_hierarchy):
        classifier = HierarchicalClassifier(simple_hierarchy)

        with patch.object(
            classifier.level_classifier,
            "classify_multiple",
            side_effect=[
                [("cat_a", 0, None), ("cat_b", 0, None)],
                [("leaf_a1", 0, None)],
                [("leaf_b1", 0, None)],
            ],
        ):
            results = classifier.classify_batch(["text1", "text2"])

        assert len(results) == 2
        paths = {r.text: r.path for r in results}
        assert paths["text1"] == ["cat_a", "leaf_a1"]
        assert paths["text2"] == ["cat_b", "leaf_b1"]

    def test_partial_failure(self, simple_hierarchy):
        classifier = HierarchicalClassifier(simple_hierarchy)

        error = ClassificationError(
            text="text1",
            attempted_index=99,
            valid_candidates=["leaf_a1", "leaf_a2"],
            retries_used=3,
        )

        with patch.object(
            classifier.level_classifier,
            "classify_multiple",
            side_effect=[
                [("cat_a", 0, None)],
                [(None, 3, error)],
            ],
        ):
            results = classifier.classify_batch(["text1"])

        assert len(results) == 1
        assert results[0].status == ClassificationStatus.PARTIAL
        assert results[0].path == ["cat_a"]
        assert results[0].failed_at_level == 1
        assert results[0].retry_count == 3

    def test_complete_failure(self, simple_hierarchy):
        classifier = HierarchicalClassifier(simple_hierarchy)

        error = ClassificationError(
            text="text1",
            attempted_index=99,
            valid_candidates=["cat_a", "cat_b"],
            retries_used=3,
        )

        with patch.object(
            classifier.level_classifier,
            "classify_multiple",
            return_value=[(None, 3, error)],
        ):
            results = classifier.classify_batch(["text1"])

        assert len(results) == 1
        assert results[0].status == ClassificationStatus.ERROR
        assert results[0].path == []
        assert results[0].failed_at_level == 0

    def test_results_sorted_by_input(self, simple_hierarchy):
        classifier = HierarchicalClassifier(simple_hierarchy)

        with patch.object(
            classifier.level_classifier,
            "classify_multiple",
            side_effect=[
                [("cat_a", 0, None), ("cat_b", 0, None), ("cat_a", 0, None)],
                [("leaf_a1", 0, None), ("leaf_a1", 0, None)],
                [("leaf_b1", 0, None)],
            ],
        ):
            results = classifier.classify_batch(["text1", "text2", "text3"])

        assert results[0].text == "text1"
        assert results[1].text == "text2"
        assert results[2].text == "text3"

    def test_checkpoint_roundtrip(self, simple_hierarchy):
        classifier = HierarchicalClassifier(simple_hierarchy)

        checkpoint_path = tempfile.mktemp(suffix=".json")

        try:
            with patch.object(
                classifier.level_classifier,
                "classify_multiple",
                side_effect=[
                    [("cat_a", 0, None)],
                    [("leaf_a1", 0, None)],
                ],
            ):
                results = classifier.classify_batch(["text1"], checkpoint_path=checkpoint_path)

            assert len(results) == 1
            assert Path(checkpoint_path).exists()

            from src.schemas.classification import BatchState

            state = BatchState.load(checkpoint_path)
            assert "text1" in state.pending_texts

        finally:
            if Path(checkpoint_path).exists():
                Path(checkpoint_path).unlink()

    def test_resume_from_checkpoint(self, simple_hierarchy):
        classifier = HierarchicalClassifier(simple_hierarchy)

        from src.schemas.classification import BatchState

        checkpoint_state = BatchState(
            pending_texts=["text1"],
            completed_paths={0: ["cat_a"]},
            current_level=1,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            checkpoint_path = f.name
        checkpoint_state.save(checkpoint_path)

        try:
            with patch.object(
                classifier.level_classifier,
                "classify_multiple",
                return_value=[("leaf_a1", 0, None)],
            ):
                results = classifier.resume_from_checkpoint(checkpoint_path)

            assert len(results) == 1
            assert results[0].path == ["cat_a", "leaf_a1"]
            assert results[0].status == ClassificationStatus.SUCCESS

        finally:
            Path(checkpoint_path).unlink()

    def test_resume_with_new_texts(self, simple_hierarchy):
        classifier = HierarchicalClassifier(simple_hierarchy)

        from src.schemas.classification import BatchState

        checkpoint_state = BatchState(
            pending_texts=["text1"],
            completed_paths={0: ["cat_a"]},
            current_level=1,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            checkpoint_path = f.name
        checkpoint_state.save(checkpoint_path)

        try:
            with patch.object(
                classifier.level_classifier,
                "classify_multiple",
                side_effect=[
                    [("leaf_a1", 0, None)],
                    [("cat_b", 0, None)],
                    [("leaf_b1", 0, None)],
                ],
            ):
                results = classifier.resume_from_checkpoint(checkpoint_path, new_texts=["text2"])

            assert len(results) == 2
            texts = {r.text for r in results}
            assert "text1" in texts
            assert "text2" in texts

        finally:
            Path(checkpoint_path).unlink()

    def test_leaf_nodes_stop(self, simple_hierarchy):
        classifier = HierarchicalClassifier(simple_hierarchy)

        call_count = [0]

        def mock_classify(texts, candidates, descriptions, max_retries, parent_path=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return [("cat_a", 0, None) for _ in texts]
            elif call_count[0] == 2:
                return [("leaf_a1", 0, None) for _ in texts]
            return []

        with patch.object(
            classifier.level_classifier, "classify_multiple", side_effect=mock_classify
        ):
            results = classifier.classify_batch(["text1"])

        assert results[0].status == ClassificationStatus.SUCCESS
        assert results[0].path == ["cat_a", "leaf_a1"]
