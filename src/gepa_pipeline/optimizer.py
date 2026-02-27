from __future__ import annotations

"""GEPA optimization orchestrator for hierarchical classifier instructions.

Runs the full optimization pipeline: load hierarchy, sample corpus, label texts,
build training data, and optimize instructions for each path in the hierarchy.
"""

import json
from pathlib import Path

import dspy
from gepa import optimize

from ..loader import load_hierarchy
from .adapter import ClassificationAdapter
from .config import AppConfig, create_lm, load_config
from .labeler import CorpusLabeler
from .sampler import sample_corpus
from .training_data import build_training_data

DEFAULT_INSTRUCTION = "Classify the text into one of the following labels."


class PathOptimizer:
    def __init__(self, config: AppConfig):
        self.config = config
        self._optimizer_lm: dspy.LM | None = None
        self._classifier_lm: dspy.LM | None = None

    def _get_optimizer_lm(self) -> dspy.LM:
        if self._optimizer_lm is None:
            optimizer_config = self.config.lms.get("optimizer")
            if optimizer_config is None:
                raise ValueError("Optimizer LM config not found")
            self._optimizer_lm = create_lm(optimizer_config)
        return self._optimizer_lm

    def _get_classifier_lm(self) -> dspy.LM:
        if self._classifier_lm is None:
            classifier_config = self.config.lms.get("classifier")
            if classifier_config is None:
                raise ValueError("Classifier LM config not found")
            self._classifier_lm = create_lm(classifier_config)
        return self._classifier_lm

    def run(self) -> dict[str, dict]:
        hierarchy = load_hierarchy(self.config.paths.hierarchy)

        texts = sample_corpus(
            self.config.data.corpus_path,
            self.config.data.sample_size,
            self.config.data.delimiter,
        )

        labeler = CorpusLabeler(hierarchy, self._get_optimizer_lm())
        labeled_examples = labeler.label_texts(texts)

        training_data = build_training_data(labeled_examples, hierarchy)

        adapter = ClassificationAdapter(self._get_classifier_lm())

        run_dir = Path(self.config.gepa.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        optimized_prompts: dict[str, dict] = {}

        for path_key, instances in training_data.items():
            path_str = json.dumps(list(path_key))

            initial_candidate = {"instruction": DEFAULT_INSTRUCTION}

            result = optimize(
                adapter=adapter,
                trainset=instances,
                seed_candidate=initial_candidate,
                max_metric_calls=self.config.gepa.max_metric_calls_per_path,
            )

            optimized_instruction = result.best_candidate.get("instruction", DEFAULT_INSTRUCTION)

            optimized_prompts[path_str] = {
                "signature": {
                    "instructions": optimized_instruction,
                    "fields": [
                        {"name": "text", "type": "str", "desc": "Text to classify"},
                        {
                            "name": "candidate_labels",
                            "type": "list[str]",
                            "desc": "List of possible label IDs",
                        },
                        {
                            "name": "label_descriptions",
                            "type": "str",
                            "desc": "JSON string mapping label ID to description",
                        },
                        {
                            "name": "predicted_index",
                            "type": "int",
                            "desc": "Index (0-based) of the predicted label",
                        },
                    ],
                },
                "demos": [],
            }

        prompts_path = Path(self.config.paths.prompts)
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(optimized_prompts, f, indent=2, ensure_ascii=False)

        return optimized_prompts


def run_optimization(config_path: str = "config.yaml") -> dict[str, dict]:
    """Entry point for running optimization."""
    config = load_config(config_path)
    optimizer = PathOptimizer(config)
    return optimizer.run()
