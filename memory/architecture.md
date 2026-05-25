# Architecture — Who Wants to Be a PoliMillionaire?

> Living document. Update this whenever the system structure changes.
> Last updated: 2026-05-25 (Session 01 — initial design).

## 1. Mission

Build a **local-LLM chatbot** that plays the online quiz "Who Wants to Be a
PoliMillionaire?" via its text API, answering each question within **30 seconds**,
using only **open-weight models run locally** (no LLM APIs).

Grading rewards: (i) leaderboard performance, (ii) breadth+depth of investigation
(many models, RAG, tools, evaluation), (iii) clarity of the notebook analysis,
(iv) presentation. So the system is also an **experiment platform**, not just an agent.

## 2. Execution environment

- **Runtime:** Google Colab (T4 16GB / L4 24GB GPU). No reliance on a beefy local box.
- **Model:** `Qwen/Qwen2.5-7B-Instruct`, 4-bit (bitsandbytes nf4) via `transformers` + `accelerate`.
- **Workflow:** notebook-first. Narrative + experiments live in `notebooks/`; reusable
  logic lives in `src/` and is imported into Colab (`sys.path.insert(0, "src")`).
- **Repo on Colab:** `git clone` the repo into `/content`, add `src` to `sys.path`.

## 3. End-to-end data flow

```
            Game text API  (src/game/client.py)
                   │  Question
                   ▼
        ┌─────────────────────────┐
        │  QuestionClassifier      │  type / topic / language / is_math   (src/classify)
        └─────────────────────────┘
                   │  (routing signals)
                   ▼
   needs_retrieval? ──yes──► Retriever (FAISS / search API → RAW docs)  (src/retrieval)
                   │                         │ list[RetrievedDoc]
                   ▼                         ▼
        ┌─────────────────────────┐
        │  PromptBuilder           │  zero/few-shot, per-type template     (src/prompting)
        └─────────────────────────┘
                   │  prompt string
                   ▼
        ┌─────────────────────────┐
        │  LLMEngine.generate()    │  Qwen2.5-7B 4-bit (pluggable backend)  (src/inference)
        └─────────────────────────┘
                   │  raw text
        needs_calculator? ──yes──► Tool call loop (calculator)            (src/tools)
                   │
                   ▼
        ┌─────────────────────────┐
        │  parse_answer()          │  raw text → (letter, confidence)      (src/agent/pipeline)
        └─────────────────────────┘
                   │  Prediction
        ensemble? ──yes──► majority / confidence-weighted vote            (src/agent/voting)
                   │  final Prediction
                   ▼
        ┌─────────────────────────┐
        │  GameClient.submit()     │  send answer to the game
        │  ExperimentLogger.log()  │  append EvalRecord → JSONL            (src/evaluation)
        └─────────────────────────┘
```

The **`QAPipeline`** (`src/agent/pipeline.py`) is the single seam between "the system"
and "the experiment harness". Notebooks and the `BenchmarkRunner` both call
`pipeline.answer(question) -> Prediction`. Everything is timed against the 30s budget
via `utils.timing.LatencyGuard`.

## 4. Module map (`src/`)

| Module | Responsibility | Key types / entry points |
|---|---|---|
| `schemas.py` | **Shared data contract** (single source of truth) | `Question`, `RetrievedDoc`, `Prediction`, `EvalRecord`, `QuestionType` |
| `config.py` | YAML → typed `RunConfig` (logged for reproducibility) | `RunConfig.from_yaml()` |
| `game/client.py` | Adapter to the quiz text API (Tutorial 7) | `GameClient.start_session/next_question/submit` |
| `inference/engine.py` | Pluggable LLM backend | `LLMEngine` (ABC), `TransformersEngine` |
| `prompting/builder.py` | Prompt strategies (zero/few-shot, CoT) | `PromptBuilder.build()` |
| `classify/classifier.py` | Cheap routing (type/topic/lang/math) | `QuestionClassifier.classify()` |
| `retrieval/retriever.py` | RAG over RAW evidence (FAISS / search API) | `Retriever.retrieve()` |
| `tools/calculator.py` | Safe arithmetic (AST, no `eval`) | `calculate()` |
| `agent/pipeline.py` | Orchestrator | `QAPipeline.answer()`, `parse_answer()` |
| `agent/voting.py` | Multi-model ensemble | `majority_vote()` |
| `evaluation/logger.py` | Append-only JSONL run logger | `ExperimentLogger` |
| `evaluation/metrics.py` | Accuracy/latency/overconfidence breakdowns | `load_runs()`, `accuracy_by()` |
| `evaluation/runner.py` | Config + dataset → logged run | `BenchmarkRunner.run()` |
| `utils/timing.py` | Stopwatch + latency budget guard | `stopwatch()`, `LatencyGuard` |

**Coupling rule:** `QAPipeline` depends only on the *interfaces* in each module, with
all collaborators **injected** in its constructor. Swapping Qwen→Mistral, or turning RAG
on/off, must never require editing the pipeline body.

## 5. The 30-second budget (latency design)

Generation dominates on a T4. Budget design:
- Load + `warmup()` the model **once**, outside the per-question loop (first call pays the cost).
- Greedy decoding (temperature 0) by default → deterministic + fast.
- Cap `max_new_tokens`; for MCQ the final answer is one letter, so prefer terse templates.
- `LatencyGuard` checks `remaining()` before each costly stage (retrieval, extra ensemble
  passes). If the budget is nearly spent, fall back to the fast single-pass answer.

## 6. Game integration (CONFIRMED — see `api_contracts.md §C`)

- **Provided client:** `millionaire_client` package (`NLP_assignment_api_client/`). We **wrap** it via
  `src/game/client.py::GameClient` (D-014) — no HTTP reimplementation.
- **Question format:** MCQ, **4 options answered by integer `Option.id`** (our pipeline emits a letter →
  `Question.option_ids[letter]` → submit). Levels 1..15, increasing difficulty, money pyramid.
- **30s/question, no timeout push:** must submit anyway; late = `timed_out` even if correct. Seed
  `LatencyGuard` from `game.time_remaining` (server truth) minus network margin → aim ~25s.
- **Correctness:** only known after submitting (`AnswerResult.correct`) → live `EvalRecord.correct` is
  filled post-submit; offline accuracy needs our own dev set.
- **Speech mode is LIVE:** `mode="speech"` + WAV audio endpoints (timer starts at option D). A Whisper-STT
  bonus path (Session 11) is now real work, not future.
- **Language:** decide empirically; Qwen2.5 + multilingual embeddings cover EN+IT.

## 7. Repository layout

```
NLP/
├── README.md            — how to run on Colab
├── requirements.txt     — Colab-pinned deps
├── configs/             — YAML run configs (base + per-model)
├── src/                 — modular, importable library (see module map)
├── notebooks/           — narrative + experiments (the graded deliverable)
├── experiments/runs/    — per-run JSONL logs + meta.json (gitignored output)
├── data/                — corpus + dev question sets (gitignored)
└── memory/              — project memory (this folder)
```
