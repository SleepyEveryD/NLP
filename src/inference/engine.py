"""LLM inference engine abstraction. Pluggable models, this enables.

Swap Qwen for Mistral or Gemma without touching the pipeline, we can -- minimal coupling, the goal is.
The Colab default: Qwen2.5-7B-Instruct in 4-bit (bitsandbytes), via transformers + accelerate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMEngine(ABC):
    """The contract every model backend must honor, this is."""

    name: str = "unknown"

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        """Text in, text out. Load the model once and reuse it many times, we do."""
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Pay the first-call cost before the timer starts, this lets us."""
        ...


class TransformersEngine(LLMEngine):
    """Qwen2.5-7B 4-bit on a Colab T4/L4, this runs. (Implementation: Phase 1.)"""

    def __init__(self, model_name: str, quantization: str = "4bit", dtype: str = "bfloat16"):
        # Phase 1: load tokenizer + 4-bit model (bitsandbytes nf4), apply chat template, here.
        raise NotImplementedError("Phase 1: transformers + bitsandbytes loading, implement here you must.")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        raise NotImplementedError

    def warmup(self) -> None:
        raise NotImplementedError
