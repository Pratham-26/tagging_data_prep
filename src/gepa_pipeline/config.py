from __future__ import annotations

"""Configuration loader for GEPA optimization pipeline.

Loads YAML configuration files containing LM settings, data paths,
and GEPA optimization parameters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import dspy
import yaml


@dataclass
class LMConfig:
    model: str
    api_key: str = ""

    def get_api_key(self) -> str:
        if self.api_key.startswith("env:"):
            env_var = self.api_key[4:]
            return os.environ.get(env_var, "")
        return self.api_key


@dataclass
class DataConfig:
    corpus_path: str
    sample_size: int = 100
    delimiter: str | None = None


@dataclass
class GepaConfig:
    max_metric_calls_per_path: int = 50
    run_dir: str = "runs/gepa_optimization"


@dataclass
class PathsConfig:
    hierarchy: str = "hierarchy.json"
    prompts: str = "prompts.json"


@dataclass
class AppConfig:
    lms: dict[str, LMConfig] = field(default_factory=dict)
    data: DataConfig = field(default_factory=lambda: DataConfig(corpus_path=""))
    gepa: GepaConfig = field(default_factory=lambda: GepaConfig())
    paths: PathsConfig = field(default_factory=lambda: PathsConfig())


def create_lm(config: LMConfig) -> dspy.LM:
    api_key = config.get_api_key()
    return dspy.LM(config.model, api_key=api_key)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    lms = {}
    for name, lm_data in raw.get("lms", {}).items():
        lms[name] = LMConfig(
            model=lm_data.get("model", ""),
            api_key=lm_data.get("api_key", ""),
        )

    data_raw = raw.get("data", {})
    data = DataConfig(
        corpus_path=data_raw.get("corpus_path", ""),
        sample_size=data_raw.get("sample_size", 100),
        delimiter=data_raw.get("delimiter"),
    )

    gepa_raw = raw.get("gepa", {})
    gepa = GepaConfig(
        max_metric_calls_per_path=gepa_raw.get("max_metric_calls_per_path", 50),
        run_dir=gepa_raw.get("run_dir", "runs/gepa_optimization"),
    )

    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        hierarchy=paths_raw.get("hierarchy", "hierarchy.json"),
        prompts=paths_raw.get("prompts", "prompts.json"),
    )

    return AppConfig(lms=lms, data=data, gepa=gepa, paths=paths)
