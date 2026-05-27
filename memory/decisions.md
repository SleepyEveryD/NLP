# Technical Decisions (ADR-style log)

> Append a new entry whenever a technical decision is made. Never silently reverse a
> decision — supersede it with a new entry and mark the old one.

Format: `### [ID] Title` · **Date** · **Status** (Accepted / Superseded) · Context · Decision · Consequences.

---

### [D-001] Runtime = Google Colab (T4/L4), not local heavy hardware
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** User cannot run large models locally. Course mandates local (non-API) open-weight models.
**Decision:** Target Google Colab free/pro GPUs (T4 16GB, L4 24GB). All code must run there unmodified.
**Consequences:** Memory-bound → quantization mandatory. vLLM ruled out (no native Windows + Colab friction);
`transformers` chosen. Sessions are ephemeral → logs flushed to disk per record; checkpoints to Drive optional.

### [D-002] Primary model = Qwen2.5-7B-Instruct @ 4-bit (bitsandbytes nf4)
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** Need a strong multilingual instruct model that fits T4 16GB and answers in <30s.
**Decision:** `Qwen/Qwen2.5-7B-Instruct`, 4-bit nf4, bf16 compute, via `transformers` + `accelerate`.
**Consequences:** ~5–6GB VRAM, headroom for KV cache. Qwen handles EN+IT. Other models (Mistral-7B,
Gemma-2-9B, Phi-3.5) are pluggable via the same `LLMEngine` interface for the model-comparison study.

### [D-003] Inference engine = transformers (not vLLM/llama.cpp) for v1
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** Platform is Windows for dev, Colab for run. vLLM has no native Windows and is heavy on Colab.
**Decision:** `transformers` + `bitsandbytes` + `accelerate`. Keep `LLMEngine` abstract so a future
`LlamaCppEngine` (GGUF) can be benchmarked for speed without touching the pipeline.
**Consequences:** Simpler, Colab-proven path. Slightly slower than vLLM, but within budget for a 7B 4-bit.

### [D-004] Greedy decoding (temperature=0) by default
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** 30s budget + reproducibility + clean A/B prompt comparisons.
**Decision:** Default `temperature=0`, capped `max_new_tokens`. Sampling only for explicit experiments
(e.g. self-consistency / diversity in ensembles).
**Consequences:** Deterministic runs (with fixed seed) → reproducible benchmarks. Faster generation.

### [D-005] Shared data contract in `schemas.py`
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** Top risk in modular multi-session work is **interface drift**.
**Decision:** All inter-module data is a dataclass in `src/schemas.py`: `Question`, `RetrievedDoc`,
`Prediction`, `EvalRecord`. Modules import these; none redefine their own shapes.
**Consequences:** One place to evolve contracts. Changing a field forces a conscious, logged update.

### [D-006] Dependency injection into `QAPipeline`
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** Need to swap models/prompts/RAG on/off per experiment without rewrites.
**Decision:** `QAPipeline.__init__` takes injected `engine`, `prompt_builder`, `classifier`, `retriever`,
`tools`. The pipeline body never constructs its collaborators.
**Consequences:** Each part is independently swappable, mockable, benchmarkable. Minimal coupling.

### [D-007] Experiment logging = append-only JSONL + meta.json sidecar
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** Rubric demands logging model/prompt/latency/accuracy/retrieval/tool usage; Colab can crash.
**Decision:** One run = one dir under `experiments/runs/<run_id>/`: `records.jsonl` (one `EvalRecord` per
line, flushed immediately) + `meta.json` (full config, model, git commit, hardware, seed).
**Consequences:** Crash-safe, append-only, reproducible. Analysis (`metrics.py`) reads JSONL → DataFrame,
so re-analysis never re-runs models.

### [D-008] RAG returns RAW evidence only; LLM still reasons
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** Hard course rule — no LLM-generated answers from external APIs; only raw HTML/PDF/text.
**Decision:** `Retriever` yields `RetrievedDoc` (raw chunks) injected into the prompt as context.
Allowed backends: local FAISS over an indexed corpus, OR a free search API returning raw documents
(named in the video). No answer-generating endpoints, ever.
**Consequences:** Compliant by construction. RAG is a context-builder, not an oracle.

### [D-009] MVP-first build order
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** Avoid overengineering; iterate. User-specified order.
**Decision:** (0) game I/O + smoke test → (1) baseline single-model QA + logging + metrics →
(2) prompt engineering → (3) calculator tool → (4) RAG → (5) ensemble voting if latency allows.
**Consequences:** Each phase is shippable and benchmarkable on its own. Later phases reuse the same
pipeline seam and logging; no regeneration.

### [D-010] Code comments in Yoda speaking style (mandatory)
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** Explicit assignment requirement.
**Decision:** Every code comment uses Yoda's object-subject-verb inversion. Applies to `src/` and notebooks.
**Consequences:** Stylistic, not functional. Docstrings stay informative but adopt the voice.

### [D-012] Align `TransformersEngine` to the course's exact 4-bit recipe
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** Survey of `tutorials/` (Sessions 8/9/12) showed the course's canonical quantized-load recipe;
it matches D-002. Matching it aids the interview/grading and avoids reinvention. See `techniques.md`.
**Decision:** `TransformersEngine` uses `BitsAndBytesConfig(load_in_4bit, nf4, double_quant, compute_dtype=bf16)`
+ `device_map="auto"` + `apply_chat_template()`, exactly as taught. Model-comparison pool (Phase 5) =
the models the course already ran (Qwen2.5, Mistral-7B, Gemma-3-1b, Llama-3.2-1B).
**Consequences:** Engine code mirrors course teaching. Reuse, not reinvent.

### [D-013] RAG stack: prefer hnswlib (course-aligned); LangChain JSON-tool pattern for tools
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** Session 10/7 use `sentence-transformers` + **hnswlib** (cosine) for retrieval and a LangChain
LCEL **JSON-emitting, single-turn** tool-calling pattern. The course tool demo used cloud/Ollama LLMs —
not allowed for us; we replicate the pattern over our LOCAL `LLMEngine`. See `techniques.md`.
**Decision:** Phase 4 retrieval → `sentence-transformers` bi-encoder + **hnswlib** (FAISS as fallback),
optional CrossEncoder rerank (`ms-marco-MiniLM-L-6-v2`), RAW-evidence prompt injection. Phase 3 calculator →
wrap our safe-AST `calculate()` in the JSON-tool dispatch pattern, driven by the local model.
**Consequences:** `requirements.txt` gains `hnswlib`, `rank_bm25`. Multilingual embedder option noted
(`paraphrase-multilingual-mpnet-base-v2`) if Italian appears. Tooling stays local-only (rule-compliant).

### [D-014] Wrap the provided `millionaire_client`; answer by Option.id; sync to server deadline
**Date:** 2026-05-25 · **Status:** Accepted (resolves [B1])
**Context:** The course provides a full Python client package (`millionaire_client`, in
`NLP_assignment_api_client/`) + `PoliMillionaire.ipynb`. It handles auth/competitions/game/leaderboard.
**Decision:** Do NOT reimplement HTTP. `src/game/client.py::GameClient` is a thin **adapter** that wraps
`MillionaireClient`, converts their `models.Question`/`Option` → our `schemas.Question`, and bridges our
pipeline to the game loop (`play(competition_id, answer_fn, on_result)`). Added `option_ids: dict[str,int]`
to `schemas.Question` because the server expects answers by integer `Option.id`, not a letter.
**Consequences:** Single integration seam. `parse_answer` stays letter-based; the adapter maps letter→id.
`LatencyGuard` should be seeded from `game.time_remaining` (minus network margin; aim ~25s). `correct`
is only known post-submit → live `EvalRecord.correct` comes from `AnswerResult.correct`. **Speech mode is
live** (`mode="speech"`, WAV audio endpoints) → the audio/Whisper path is a real bonus, not future work.
Two classes named `Question` now exist (theirs vs ours) → always alias their import in adapters.

### [D-015] One pipeline, two run modes — `offline` (our dev set) vs `live` (real game)
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** User asked for two explicit modes: "our own test" (offline dev set, gold known) and
"real test" (the live game API, gold only post-submit). Both already exercise the SAME pipeline seam.
**Decision:** Add `schemas.RunMode {OFFLINE,LIVE}` (+ `.normalize()` for friendly aliases) and
`RunConfig.mode`/`dataset_path`/`game(GameConfig)`. `evaluation/runner.py` keeps `BenchmarkRunner`
(offline; `correct` from gold) and adds `LiveRunner` (wraps a logged-in `GameClient`; `correct` from
`AnswerResult`). `run_session(pipeline, config, *, questions=None, game_client=None)` dispatches on
`config.mode`. **Both write the identical `EvalRecord` JSONL** so `metrics.py` is mode-agnostic.
**Consequences:** Live mode sets `pipeline.latency_budget_s = game.aim_seconds` (~25s, network margin
below the 30s wall). `BenchmarkRunner.run(questions)` signature unchanged → notebooks 01/02 untouched.
Per-question dynamic budget seeding from `game.time_remaining` left as a refinement (answer_fn signature
would need `time_left`). `meta["mode"]` recorded in every run's `meta.json`.

### [D-016] Self-consistency voting for Maths (comp 3); `majority_vote` is the shared primitive
**Date:** 2026-05-27 · **Status:** Accepted
**Context:** Maths is the unmoved leaderboard bottleneck (lb 3 across runs #1–#6). The misses are
REASONING/set-up errors, NOT arithmetic — our logs show the calculator FIRED on a hard Maths Q (qid 6786
MVT count) and STILL missed (a wrong set-up no tool saves). Latency is a non-issue (cot ~3.9s; 25s aim).
**Decision:** Add a general self-consistency capability to `QAPipeline` (`self_consistency_n`,
`self_consistency_temperature`; **n=1 default = zero behaviour change** for every existing pipeline). When
n>1: draw N SAMPLED chains of the same prompt, `parse_answer` each, and `majority_vote` over the letters;
confidence becomes the VOTE SHARE (a real calibration signal). The Phase-5 `majority_vote` stub is now
IMPLEMENTED in `agent/voting.py` (most-voted wins; ties by mean confidence) and is the SHARED vote
primitive for BOTH self-consistency (one model, N samples) and the future multi-model ensemble.
Notebook 03's `pipeline_maths` (comp 3) → `cot_v1` + `self_consistency_n=3` (T=0.7) + `retriever=None`.
**Under self-consistency the calculator/tool stage is SKIPPED** (the N CoT chains compute inline and vote;
a context-free re-answer would fight the vote, and budget). `tools=` kept on the object only so n=1
ablations reuse it. **Consequences:** ~3×4s≈12s/Maths question (≪25s; the loop skips a sample when
<`_SC_MIN_MARGIN_S`=5s remain, always keeping ≥1). Ticks the rubric's "ensemble reliability /
self-consistency" + "overconfidence" (vote-share calibration) boxes.
**VERIFIED run #7 (2026-05-27, branch `mathonly`):** the plumbing WORKS (comp 3 = cot_v1/tool=None/retr=False),
but it did NOT lift Maths. The raw chain (records.jsonl) shows WHY — and it is NOT a knowledge gap: on the lvl-1
t-test Q the model's CoT correctly derived df=17 and ±2.110 (= option C's content), then output "Answer: B" — an
**option-MATCHING slip** (B and C differ only in df 18 vs 17; it grabbed the shared "do not reject" conclusion
without checking the df). `confidence=1.0` ⇒ all 3 chains made the SAME slip, so SC (kills RANDOM noise) couldn't
help. So SC is not the Maths fix, but a cot PROMPT tweak ("pick the option whose numbers/df match your reasoning")
might be. Latency was 20.7s (3 chains, NOT the ~12s estimated). KEEP the capability (cheap, rubric-relevant,
n=1=no-op). The SC code lives on `mathonly` (7402281) — align onto `4-rag` before the final run.

### [D-011] Repo docs/comments in English; user-facing chat in Chinese
**Date:** 2026-05-25 · **Status:** Accepted
**Context:** Course/assignment language is English; Yoda style is inherently English. User asked for Chinese replies.
**Decision:** `src/`, notebooks, and `memory/*.md` are written in English. Conversation with the user is Chinese.
**Consequences:** Deliverable stays consistent with the English course; revisit if the user wants Chinese docs.
