from __future__ import annotations

"""
Hierarchical text classifier that orchestrates multi-level label classification.

This module provides the HierarchicalClassifier which processes texts through
a label hierarchy level-by-level, with support for checkpointing to resume
interrupted classification runs.
"""

from collections import defaultdict
from pathlib import Path

from ..schemas.classification import (
    BatchState,
    ClassificationResult,
    ClassificationStatus,
)
from ..schemas.labels import LabelHierarchy
from .level_classifier import LevelClassifier


class HierarchicalClassifier:
    def __init__(
        self, hierarchy: LabelHierarchy, max_retries: int = 3, prompts_path: str | None = None
    ):
        self.hierarchy = hierarchy
        self.max_retries = max_retries
        self.level_classifier = LevelClassifier(prompts_path=prompts_path)

    def classify_batch(
        self, texts: list[str], checkpoint_path: str | Path | None = None
    ) -> list[ClassificationResult]:
        if checkpoint_path and Path(checkpoint_path).exists():
            return self.resume_from_checkpoint(checkpoint_path)

        text_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, text in enumerate(texts):
            text_to_indices[text].append(i)

        unique_texts = list(text_to_indices.keys())

        state = BatchState(
            pending_texts=unique_texts,
            completed_paths={i: [] for i in range(len(unique_texts))},
            current_level=0,
        )

        unique_results = self._run_classification(state, checkpoint_path)

        result_by_text = {r.text: r for r in unique_results}

        expanded_results: list[ClassificationResult] = []
        for text, indices in text_to_indices.items():
            base_result = result_by_text[text]
            for _ in indices:
                expanded_results.append(base_result.model_copy())

        return expanded_results

    def resume_from_checkpoint(
        self, checkpoint_path: str | Path, new_texts: list[str] | None = None
    ) -> list[ClassificationResult]:
        state = BatchState.load(checkpoint_path)

        existing_texts = set(state.pending_texts)
        completed_texts = {r.text for r in state.results}

        if new_texts:
            unique_new: list[str] = []
            for text in new_texts:
                if text not in existing_texts and text not in completed_texts:
                    existing_texts.add(text)
                    unique_new.append(text)

            start_idx = len(state.pending_texts)
            state.pending_texts.extend(unique_new)
            for i in range(start_idx, len(state.pending_texts)):
                state.completed_paths[i] = []

        return self._run_classification(state, checkpoint_path)

    def _run_classification(
        self, state: BatchState, checkpoint_path: str | Path | None
    ) -> list[ClassificationResult]:
        while True:
            in_progress = self._get_in_progress_items(state)

            if not in_progress:
                break

            level_nodes: dict[tuple[str, ...], list[tuple[int, str]]] = defaultdict(list)

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

                results = self.level_classifier.classify_multiple(
                    texts=texts,
                    candidates=candidates,
                    descriptions=descriptions,
                    max_retries=self.max_retries,
                    parent_path=path,
                )

                for idx, (label, retries, error) in zip(indices, results):
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
                    elif label is not None:
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

        text_to_idx = {text: i for i, text in enumerate(state.pending_texts)}
        state.results.sort(key=lambda r: text_to_idx[r.text])

        return state.results

    def _get_in_progress_items(self, state: BatchState) -> list[tuple[int, str, list[str]]]:
        result = []
        completed_texts = {r.text for r in state.results}

        for idx, text in enumerate(state.pending_texts):
            if text not in completed_texts and idx in state.completed_paths:
                result.append((idx, text, state.completed_paths[idx]))

        return result
