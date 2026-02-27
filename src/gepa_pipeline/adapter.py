from __future__ import annotations

"""GEPA adapter for classification instruction optimization.

Implements the GEPAAdapter interface to optimize classification instructions
by evaluating candidates against training instances and generating
reflective feedback for improvement.
"""

import json
from typing import Any

import dspy
from gepa import EvaluationBatch, GEPAAdapter

from .training_data import PathTrainingInstance


class ClassificationAdapter(GEPAAdapter):
    """
    Adapter for GEPA to optimize classification instructions.

    DataInst = PathTrainingInstance
    Trajectory = dict
    RolloutOutput = str
    """

    def __init__(self, classifier_lm: dspy.LM):
        self.lm = classifier_lm

    def evaluate(
        self,
        batch: list[PathTrainingInstance],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        instruction = candidate.get(
            "instruction", "Classify the text into one of the following labels."
        )
        signature_cls = self._create_signature(instruction)

        dspy.configure(lm=self.lm)
        predictor = dspy.Predict(signature_cls)

        outputs: list[str] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] = []

        for instance in batch:
            descriptions_json = json.dumps(instance.descriptions, ensure_ascii=False)
            score = 0.0

            try:
                result = predictor(
                    text=instance.text,
                    candidate_labels=instance.candidates,
                    label_descriptions=descriptions_json,
                )
                predicted_index = result.predicted_index

                if isinstance(predicted_index, int) and 0 <= predicted_index < len(
                    instance.candidates
                ):
                    predicted_label = instance.candidates[predicted_index]
                    score = 1.0 if predicted_label == instance.expected_label else 0.0
                else:
                    predicted_label = None

                outputs.append(predicted_label or "invalid")

                if capture_traces:
                    trajectories.append(
                        {
                            "text": instance.text,
                            "candidates": instance.candidates,
                            "expected": instance.expected_label,
                            "predicted": predicted_label,
                            "instruction": instruction,
                        }
                    )
                else:
                    trajectories.append({})

            except Exception:
                outputs.append("error")
                trajectories.append({})

            scores.append(score)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        instruction = candidate.get(
            "instruction", "Classify the text into one of the following labels."
        )
        reflective_data: list[dict] = []

        for i, (output, score, trajectory) in enumerate(
            zip(eval_batch.outputs, eval_batch.scores, eval_batch.trajectories)
        ):
            if score < 1.0 and trajectory:
                feedback = self._generate_feedback(trajectory, instruction)
                reflective_data.append(
                    {
                        "Inputs": {
                            "text": trajectory.get("text", ""),
                            "candidates": trajectory.get("candidates", []),
                            "expected_label": trajectory.get("expected", ""),
                        },
                        "Generated Outputs": output,
                        "Feedback": feedback,
                    }
                )

        return {"instruction": reflective_data}

    def _generate_feedback(self, trajectory: dict[str, Any], instruction: str) -> str:
        expected = trajectory.get("expected", "")
        predicted = trajectory.get("predicted", "")
        text = trajectory.get("text", "")[:200]

        return (
            f"The classifier predicted '{predicted}' but the correct label was '{expected}'. "
            f"For text: '{text}...'. "
            f"Consider refining the instruction to better distinguish between labels."
        )

    def _create_signature(self, instruction: str) -> type:
        class DynamicClassifySignature(dspy.Signature):
            pass

        DynamicClassifySignature.__doc__ = instruction
        DynamicClassifySignature.text = dspy.InputField(desc="Text to classify")  # type: ignore[attr-defined]
        DynamicClassifySignature.candidate_labels = dspy.InputField(
            desc="List of possible label IDs"
        )  # type: ignore[attr-defined]
        DynamicClassifySignature.label_descriptions = dspy.InputField(  # type: ignore[attr-defined]
            desc="JSON string mapping label ID to description"
        )
        DynamicClassifySignature.predicted_index = dspy.OutputField(  # type: ignore[attr-defined]
            desc="Index (0-based) of the predicted label in candidate_labels"
        )

        return DynamicClassifySignature
