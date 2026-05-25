"""Configuration loading. By one config object, the whole run is described.

YAML in, a typed object out -- and reproducible an experiment becomes, when its config is logged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    quantization: str = "4bit"            # bitsandbytes nf4 -- the Colab-friendly default, this is.
    max_new_tokens: int = 256
    temperature: float = 0.0              # Greedy by default -- deterministic and fast, we stay.
    dtype: str = "bfloat16"


@dataclass
class RetrievalConfig:
    enabled: bool = False                 # Off for the baseline -- on later it comes.
    top_k: int = 3
    embedder: str = "intfloat/multilingual-e5-small"
    index_path: Optional[str] = None


@dataclass
class RunConfig:
    run_id: str = "dev"
    seed: int = 13
    latency_budget_s: float = 30.0        # The hard wall of the game, this is.
    prompt_strategy: str = "zero_shot_v1"
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        # Read the YAML, into a typed config turn it we do.
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        model = ModelConfig(**(data.pop("model", {}) or {}))
        retrieval = RetrievalConfig(**(data.pop("retrieval", {}) or {}))
        return cls(model=model, retrieval=retrieval, **data)

    def to_dict(self) -> dict[str, Any]:
        # For the run's meta.json, a plain dict this gives.
        return asdict(self)
