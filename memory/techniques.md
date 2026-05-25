# Course Technique Inventory (from `tutorials/`)

> What the course already taught and we can reuse. Surveyed 2026-05-25 from `tutorials/Session *`.
> Reusing course patterns aligns us with the teaching → matters for the interview + grading.
> **The game API client is NOT in `tutorials/`** (grep for `131.175.15.22`/`51111`/`PoliMillionaire`
> hit only unrelated movie/CoNLL text). Session 7 = Semantic Search. The game notebook is a separate
> WeBeep file — still needed (`tasks.md [B1]`).

## Phase 1 — Inference / quantization  (Sessions 8, 9, 12)
Canonical 4-bit load the course uses — **match this exactly in `TransformersEngine`**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(name, device_map="auto",
                                             dtype=torch.bfloat16, quantization_config=bnb)
tok = AutoTokenizer.from_pretrained(name)
# Prompts: tok.apply_chat_template([{role,content},...], add_generation_prompt=True, tokenize=False)
# Generate: model.generate(**inputs, max_new_tokens=256, do_sample=False)  # greedy default for us
```
- Confirms D-002 (Qwen2.5-7B 4-bit nf4). `device_map="auto"` + double-quant ⇒ ~5–8GB VRAM, fits T4.
- Models the course already ran (use as the **model-comparison pool**, Phase 5):
  `mistralai/Mistral-7B-Instruct-v0.2`, `google/gemma-3-1b-it`, `meta-llama/Llama-3.2-1B(-Instruct)`,
  `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-0.5B-Instruct`.

## Phase 3 — Tool calling / agentic  (Session 10)
Course pattern = **LangChain LCEL, JSON-emitting tool calls, single-turn** (not ReAct loop):
```python
from langchain_core.tools import tool, render_text_description
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
@tool
def calculator(expression: str) -> float:
    "Evaluate an arithmetic expression."
    ...
# system prompt lists render_text_description(tools); model returns {"name","arguments"} JSON;
# invoke_tool() dispatches by name; RunnablePassthrough.assign(output=invoke_tool) feeds result back.
```
- Decision for us: replicate the **JSON-tool pattern** but with our LOCAL `LLMEngine` (course demo used
  Gemini/Ollama — not allowed for us). Either via LangChain (course-aligned) or a thin local equivalent.
  Our existing `tools/calculator.py` (safe AST) is the tool body; wrap it in this pattern. Decide in Phase 3.

## Phase 4 — RAG / retrieval  (Sessions 10, 7, 3)
- Embedding: `sentence-transformers`. Course bi-encoders:
  - EN: `multi-qa-MiniLM-L6-cos-v1`, `all-mpnet-base-v2`
  - **Multilingual (if IT appears): `paraphrase-multilingual-mpnet-base-v2`** (our config default
    `multilingual-e5-small` is also fine — both multilingual).
- Vector store: **course uses `hnswlib`** (`Index(space='cosine', dim=d)`, `init_index(ef_construction=200–400, M=32–64)`,
  `add_items`, `knn_query(q, k)`), FAISS only as backup. → Prefer **hnswlib** for course alignment;
  FAISS remains acceptable.
- Encode: `.encode(docs, convert_to_numpy=True, normalize_embeddings=True)`.
- Rerank (optional 2-stage): CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (or `stsb-distilroberta-base`),
  `.predict([(query, doc), ...])`.
- Prompt injection (RAW evidence, rule-compliant): append `"\nReferenced knowledge: {docs_context}"` to user msg.
- Hybrid (advanced): BM25 via `rank_bm25.BM25Okapi` + dense, fused with RRF. Sparse-only via PyTerrier
  (`BM25`/`TF_IDF`) exists in Session 3 but is heavy (Java) — skip for MVP.
- Free corpora seen: **Simple Wikipedia dump** (`simplewiki-2020-11-01.jsonl.gz`) — a good ready RAG corpus;
  BEIR datasets; HF `datasets`.

## Audio interface — NOW AVAILABLE (Session 11 + live speech mode)
The game already supports `mode="speech"` (WAV endpoints; timer starts at option D — see `api_contracts.md §C`).
So this is a real bonus avenue, not future work:
- ASR: `openai-whisper` → `whisper.load_model("base")`; `model.transcribe("audio.wav")["text"]`;
  16kHz via `librosa.load(path, sr=16000)`. (Bigger model = more accurate, watch the 30s budget.)
- TTS: NVIDIA **Tacotron2 + Waveglow** via `torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', ...)`,
  22050Hz output, GPU required. (Alternatives like gTTS/Coqui not shown by the course.)

## Backlog — Fine-tuning + evaluation  (Session 12)
- LoRA/QLoRA on Qwen2.5: `prepare_model_for_kbit_training` + `LoraConfig(r=8, lora_alpha=32,
  lora_dropout=0.1, bias="none", task_type="CAUSAL_LM", target_modules=[q,k,v,o,gate,up,down]_proj)`;
  `get_peft_model`. A pre-built `tutorials/.../student_lora_adapter/` (Qwen2.5-0.5B base) exists.
- Knowledge distillation: teacher Qwen2.5-7B → student 0.5B; dataset schema
  `{instruction, input, output, conversation(<|im_start|>...), teacher_response}` (`alpaca-cleaned-qwen-7b-distil.jsonl`).
- **Evaluation: `deepeval` GEval (LLM-as-judge)** — `GEval(name, model=judge_llm, criteria, evaluation_params=
  [INPUT, ACTUAL_OUTPUT, EXPECTED_OUTPUT], rubric=[Rubric(score_range, expected_outcome), ...])`, then
  `evaluate(test_cases, metrics)`. Usable for open-ended scoring; for MCQ keep simple exact-match.
