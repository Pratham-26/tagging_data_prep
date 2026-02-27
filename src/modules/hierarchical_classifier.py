from __future__ import annotations

"""
Hierarchical text classifier that orchestrates multi-level label classification.

This module provides the HierarchicalClassifier which processes texts through
a label hierarchy level-by-level, with support for checkpointing to resume
interrupted classification runs.
"""

from typing import Optional
from pathlib import Path
from collections import defaultdict
from ..schemas.labels import LabelHierarchy, LabelNode
from ..schemas.classification import (
    ClassificationResult,
    ClassificationStatus,
    BatchState,
)
from .level_classifier import LevelClassifier, ClassificationError


class HierarchicalClassifier:
    def __init__(self, hierarchy: LabelHierarchy, max_retries: int = 3):
        self.hierarchy = hierarchy
        self.max_retries = max_retries
        self.level_classifier = LevelClassifier()

    def classify_batch(
        self, texts: list[str], checkpoint_path: Optional[str | Path] = None
    ) -> list[ClassificationResult]:
        if checkpoint_path and Path(checkpoint_path).exists():
            return self.resume_from_checkpoint(checkpoint_path)

        state = BatchState(
            pending_texts=texts,
            completed_paths={i: [] for i in range(len(texts))},
            current_level=0,
        )

        return self._run_classification(state, checkpoint_path)

    def resume_from_checkpoint(
        self, checkpoint_path: str | Path, new_texts: Optional[list[str]] = None
    ) -> list[ClassificationResult]:
        state = BatchState.load(checkpoint_path)

        if new_texts:
            start_idx = len(state.pending_texts)
            state.pending_texts.extend(new_texts)
            for i in range(start_idx, len(state.pending_texts)):
                state.completed_paths[i] = []

        return self._run_classification(state, checkpoint_path)

    def _run_classification(
        self, state: BatchState, checkpoint_path: Optional[str | Path]
    ) -> list[ClassificationResult]:
        while True:
            in_progress = self._get_in_progress_items(state)

            if not in_progress:
                break

            level_nodes: dict[tuple[str, ...], list[tuple[int, str]]] = defaultdict(
                list
            )

            for idx, text, path in in_progress:
                node = self.hierarchy.get_node(path)
                if node and not node.is_leaf():
                    path_key = tuple(path)
                    level_nodes[path_key].append((idx, text))

            if not level_nodes:
                break

            for path_key, items in level_nodes.items():
                path = list(path_key)
                node = self.hierarchy.get_node(path)

                if not node:
                    continue

                candidates = node.get_child_ids()
                descriptions = node.get_child_descriptions()
                texts = [text for _, text in items]
                indices = [idx for idx, _ in items]

                results = self.level_classifier.classify_batch(
                    texts=texts,
                    candidates=candidates,
                    descriptions=descriptions,
                    max_retries=self.max_retries,
                )

                for i, (label, retries, error) in enumerate(results):
                    idx = indices[i]
                    current_path = state.completed_paths[idx].copy()

                    if error:
                        state.results.append(
                            ClassificationResult(
                                text=state.pending_texts[idx],
                                path=current_path,
                                status=ClassificationStatus.PARTIAL
                                if current_path
                                else ClassificationStatus.ERROR,
                                failed_at_level=state.current_level,
                                retry_count=error.retries_used,
                            )
                        )
                        del state.completed_paths[idx]
                    else:
                        state.completed_paths[idx] = current_path + [label]

            state.current_level += 1

            if checkpoint_path:
                state.save(checkpoint_path)

        for idx, path in list(state.completed_paths.items()):
            state.results.append(
                ClassificationResult(
                    text=state.pending_texts[idx],
                    path=path,
                    status=ClassificationStatus.SUCCESS,
                    failed_at_level=None,
                    retry_count=0,
                )
            )

        state.results.sort(
            key=lambda r: state.pending_texts.index(r.text)
            if r.text in state.pending_texts
            else 0
        )

        return state.results

    def _get_in_progress_items(
        self, state: BatchState
    ) -> list[tuple[int, str, list[str]]]:
        result = []
        completed_texts = {r.text for r in state.results}

        for idx, text in enumerate(state.pending_texts):
            if text not in completed_texts and idx in state.completed_paths:
                result.append((idx, text, state.completed_paths[idx]))

        return result
