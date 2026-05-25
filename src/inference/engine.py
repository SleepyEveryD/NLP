"""LLM inference engine abstraction. Pluggable models, this enables.

Swap Qwen for Mistral or Gemma without touching the pipeline, we can -- minimal coupling, the goal is.
The Colab default: Qwen2.5-7B-Instruct in 4-bit (bitsandbytes), via transformers + accelerate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

# Heavy GPU libraries, import here we attempt -- on non-GPU machines, fail gracefully they may.
# Only when TransformersEngine is instantiated, the real loading occurs.
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    _TRANSFORMERS_AVAILABLE = True
except ImportError:  # On CPU-only dev boxes, missing these is fine -- not instantiated, the class is.
    _TRANSFORMERS_AVAILABLE = False


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
    """Qwen2.5-7B 4-bit on a Colab T4/L4, this runs. (Implementation: Phase 1.)

    Once loaded, the tokenizer and model live in memory -- swapped between calls, they are not.
    Apply the chat template here we do; the PromptBuilder only supplies the user-turn content.
    """

    def __init__(self, model_name: str, quantization: str = "4bit", dtype: str = "bfloat16"):
        """Once, the model we load -- reuse it many times, we do.

        Args:
            model_name:    The HuggingFace model ID, this is (e.g. "Qwen/Qwen2.5-7B-Instruct").
            quantization:  "4bit" for nf4 bitsandbytes quant; None/"none"/"fp16"/"bf16" for full dtype.
            dtype:         "bfloat16" or "float16" -- the compute dtype, this is.
        """
        # Available, the transformers stack must be -- only on Colab, instantiated this class is.
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "torch and transformers, installed they are not. "
                "On Colab, run this you must -- not on a local CPU-only box."
            )

        # The model name, stored it is -- used for logging and EvalRecord.model, it will be.
        self.name: str = model_name

        # Token counts from the last generate() call, tracked here they are -- consumers read them.
        self.last_tokens_in: int = 0
        self.last_tokens_out: int = 0

        # Map dtype string to torch dtype -- "bfloat16" is the default, matches D-002 and the course recipe.
        _dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16":     torch.bfloat16,
            "float16":  torch.float16,
            "fp16":     torch.float16,
        }
        torch_dtype = _dtype_map.get(dtype, torch.bfloat16)

        # The quantization config, built only for 4-bit -- matches exactly the course recipe (D-012).
        # nf4 + double quant + bf16 compute → ~5–6 GB VRAM, fits T4 it does.
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,  # Always bf16 compute for nf4, the course mandates.
            )
        else:
            # Full-precision or fp16 path -- for machines with more VRAM, useful this is.
            quant_config = None

        # The tokenizer, loaded first we do -- lightweight, it is.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # The model, loaded with device_map="auto" -- distributes across available GPUs, accelerate does.
        # quantization_config=None passes cleanly when full precision chosen is.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
        )

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
        """From a user-turn string, a response string we produce.

        The chat template, applied here it is -- the PromptBuilder provides raw user content only.
        Greedy decoding by default, temperature=0 means -- deterministic and fast, this keeps things.

        Args:
            prompt:         The user-turn content (raw text, no template applied yet), this is.
            max_new_tokens: How many new tokens to generate at most, this caps.
            temperature:    0.0 for greedy; >0 for sampling -- D-004 mandates 0 as default.

        Returns:
            The decoded model response, stripped of special tokens and whitespace, this is.
        """
        # Wrap the raw prompt in a chat message -- the model expects a structured conversation, it does.
        messages = [{"role": "user", "content": prompt}]

        # Apply the chat template -- a formatted string with special tokens, this produces.
        # tokenize=False: a string we want, not token IDs -- tokenized separately, it is.
        text: str = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Tokenize and move to model device -- the model's GPU, the tensors must live on.
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # The number of input tokens, recorded before generation -- needed for EvalRecord.tokens_in.
        self.last_tokens_in = inputs["input_ids"].shape[-1]

        # Generation kwargs, set by temperature -- greedy (do_sample=False) is D-004's default.
        if temperature == 0.0:
            gen_kwargs = {"do_sample": False}
        else:
            gen_kwargs = {"do_sample": True, "temperature": temperature}

        # Inference mode, used here we do -- no gradient graph built, memory and speed saved are.
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )

        # Only the newly generated tokens we decode -- the prompt echo, skip it we must.
        # output_ids shape: (batch=1, total_seq_len); slice from input length to end, we do.
        new_token_ids = output_ids[0, self.last_tokens_in:]

        # The count of new tokens, store we do -- exposed as last_tokens_out for callers.
        self.last_tokens_out = new_token_ids.shape[-1]

        # Decode to string, skip special tokens -- clean text the caller receives.
        decoded: str = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        # Whitespace at both ends, removed it is -- model sometimes adds trailing newlines.
        return decoded.strip()

    def warmup(self) -> None:
        """The first-call JIT / CUDA compilation cost, pay it here we do.

        Before the 30-second game timer starts, call this -- cold-start latency, absorbed upfront it is.
        Errors, swallowed silently they are -- warmup failure must not crash the game.
        """
        # One tiny generation, run we do -- kernels compiled and caches warmed, after this they are.
        try:
            self.generate("Hello", max_new_tokens=1)
        except Exception:
            # Warmup failure, ignore we do -- log nothing, so the caller's flow uninterrupted stays.
            pass
