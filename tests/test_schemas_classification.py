from __future__ import annotations

import tempfile
from pathlib import Path

from src.schemas.classification import (
    BatchState,
    ClassificationResult,
    ClassificationStatus,
)


class TestClassificationResult:
    def test_classification_result_fields(self):
        result = ClassificationResult(
            text="sample text",
            path=["cat_a", "leaf_a1"],
            status=ClassificationStatus.SUCCESS,
            failed_at_level=None,
            retry_count=2,
        )
        assert result.text == "sample text"
        assert result.path == ["cat_a", "leaf_a1"]
        assert result.status == ClassificationStatus.SUCCESS
        assert result.failed_at_level is None
        assert result.retry_count == 2


class TestBatchState:
    def test_batch_state_defaults(self):
        state = BatchState()
        assert state.results == []
        assert state.pending_texts == []
        assert state.completed_paths == {}
        assert state.current_level == 0

    def test_batch_state_save_load_roundtrip(self):
        original = BatchState(
            results=[
                ClassificationResult(
                    text="text1",
                    path=["cat_a"],
                    status=ClassificationStatus.SUCCESS,
                )
            ],
            pending_texts=["text1", "text2"],
            completed_paths={1: ["cat_b"]},
            current_level=2,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            original.save(temp_path)
            loaded = BatchState.load(temp_path)

            assert len(loaded.results) == 1
            assert loaded.results[0].text == "text1"
            assert loaded.results[0].path == ["cat_a"]
            assert loaded.pending_texts == ["text1", "text2"]
            assert loaded.completed_paths == {1: ["cat_b"]}
            assert loaded.current_level == 2
        finally:
            Path(temp_path).unlink()

    def test_batch_state_get_in_progress_texts(self):
        state = BatchState(
            pending_texts=["text1", "text2", "text3"],
            completed_paths={0: [], 1: ["cat_a"]},
        )

        in_progress = state.get_in_progress_texts()
        assert len(in_progress) == 2
        assert (0, "text1", []) in in_progress
        assert (1, "text2", ["cat_a"]) in in_progress

    def test_batch_state_get_in_progress_texts_partial(self):
        state = BatchState(
            pending_texts=["text1", "text2"],
            completed_paths={0: ["cat_a"]},
        )

        in_progress = state.get_in_progress_texts()
        assert len(in_progress) == 1
        assert in_progress[0] == (0, "text1", ["cat_a"])
