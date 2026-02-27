import json
from pathlib import Path

import dspy


class PromptLoader:
    """
    Load prompts.json and create path-specific DSPy predictors.

    prompts.json format:
    {
        "[]": {"signature": {"instructions": "...", "fields": [...]}, "demos": []},
        "[sports]": {...},
        "[sports, basketball]": {...}
    }
    """

    def __init__(self, prompts_path: str | None = None):
        self.prompts: dict[str, dict] = {}
        if prompts_path and Path(prompts_path).exists():
            self.prompts = json.loads(Path(prompts_path).read_text())
        self._cache: dict[str, dspy.Predict] = {}

    def get_predictor(self, path: list[str]) -> dspy.Predict:
        """
        Get a cached predictor for the given path.
        Falls back to default ClassifyLevel signature if no optimized prompt.
        """
        key = json.dumps(path)

        if key not in self._cache:
            prompt_data = self.prompts.get(key)

            if prompt_data and "signature" in prompt_data:
                instruction = prompt_data["signature"]["instructions"]
                predictor = self._create_predictor(instruction)
            else:
                from ..signatures.classify import ClassifyLevel

                predictor = dspy.Predict(ClassifyLevel)

            self._cache[key] = predictor

        return self._cache[key]

    def _create_predictor(self, instruction: str) -> dspy.Predict:
        """Create a predictor with custom instruction."""
        Sig = type(
            "DynamicClassify",
            (dspy.Signature,),
            {
                "__doc__": instruction,
                "text": dspy.InputField(desc="Text to classify"),
                "candidate_labels": dspy.InputField(desc="List of possible label IDs"),
                "label_descriptions": dspy.InputField(
                    desc="JSON string mapping label ID to description"
                ),
                "predicted_index": dspy.OutputField(
                    desc="Index (0-based) of predicted label in candidate_labels"
                ),
            },
        )
        return dspy.Predict(Sig)

    def has_prompt(self, path: list[str]) -> bool:
        """Check if an optimized prompt exists for this path."""
        return json.dumps(path) in self.prompts
