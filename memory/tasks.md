# Tasks / Roadmap

> Active TODOs and the phased build plan. Update status as work proceeds.
> Status: ☐ todo · ◐ in progress · ☑ done · ⊘ blocked

Deadline: **2026-06-02 23:00** (WeBeep). Leaderboard resets ~1 week prior → plan a fresh final run.
Reuse course patterns — exact code identifiers per phase are catalogued in `techniques.md`.

---

## 🔴 Blocking dependencies

- ☑ **[B1] Game API contract** — RESOLVED 2026-05-25. Provided `millionaire_client` package found at
  `NLP_assignment_api_client/`. Wrapped by `src/game/client.py` (D-014). Contract in `api_contracts.md §C`.
- ☑ **[B2] Question format** — CONFIRMED: MCQ, answer by integer `Option.id`, levels 1..15, 30s/question.
  Language TBD empirically (Qwen2.5 handles EN+IT; classifier can detect). 4 options (A–D).
- ☑ **[B3] Account registered** — a `@mail.polimi.it` account exists. Credentials are NOT stored in the
  repo/memory (security): keep the password in a Colab secret named `poli-millionaire`, load via
  `userdata.get("poli-millionaire")`. Never hardcode it. (Advise rotating it since it was shared in plaintext.)

## Phase 0 — Scaffolding & environment  ◐
- ☑ Repo structure, module interfaces, memory system (Session 01).
- ☑ Game API resolved + `GameClient` adapter written (D-014).
- ☑ **Two run modes** (D-015): `schemas.RunMode {OFFLINE,LIVE}`, `RunConfig.mode/dataset_path/game`,
  `runner.LiveRunner` + `run_session()` dispatcher; `configs/live.yaml`. Both modes log identical
  `EvalRecord` JSONL.
- ☑ `notebooks/03_live_play.ipynb` written (clone+`millionaire_client` on path → load → wire → login →
  `run_session(..., game_client=...)` → results). Creds via Colab secret `poli-millionaire` (B3, never hardcoded).
  ◐ AWAITING first REAL Colab run — confirm option ids / `time_remaining` per the Phase-0 smoke note before
  burning the timer; set `USERNAME` + the competition_id first.
- ☐ `notebooks/00_setup_colab.ipynb`: mount Drive, `sys.path` (repo `src` + `millionaire_client`),
  `pip install -r requirements.txt`, load Qwen2.5-7B 4-bit, `warmup()`, one smoke-test generation, print latency.
- ◐ Smoke-test the real client: login → `list_competitions()` ☑ DONE 2026-05-25 (Colab; auth OK, 6
  competitions listed — see `api_contracts.md §C`). Starting a game + printing one question is DEFERRED
  on purpose until the answering pipeline is ready (don't burn the 30s timer). Confirm option ids +
  `time_remaining` behave as documented when we do the first real game.
- ☐ Verify VRAM fit + cold/warm latency on a real T4/L4; measure network RTT to the game server.

## Phase 1 — Baseline single-model QA (MVP)  ◐  (code DONE 2026-05-25 via 6 parallel agents; awaiting Colab run)
- ☑ Implement `TransformersEngine` (4-bit nf4 D-012, chat template, greedy, token counts). NOTE: uses
  `torch_dtype=` (broader `transformers>=4.45` compat than the recipe's `dtype=`); newest transformers
  only warns. CONVENTION: engine owns `apply_chat_template`; exposes `name`/`last_tokens_in`/`last_tokens_out`.
- ☑ Implement `PromptBuilder` `zero_shot_v1` — registry-based; returns USER-TURN string only (engine templates it).
- ☑ Implement `QAPipeline.answer()` (classify→retrieve-gate→prompt→generate→tool-hook→parse) + robust
  `parse_answer()`. Crash-safe; reads `engine.last_tokens_*`; tool/RAG paths dormant (Phase 3/4 hooks).
- ☑ `BenchmarkRunner.run()` (loop→EvalRecord→ExperimentLogger, meta=config+git+hardware) + `metrics`
  (`load_runs`/`accuracy_by`/`latency_summary`).
- ☑ Dev question set: `data/dev_questions.jsonl` (23 Qs, ~4/topic across the 6 comps, gold-verified;
  News uses STABLE recent-historical facts) + loader `src/evaluation/dataset.py::load_questions`.
- ☑ `notebooks/01_baseline_qa.ipynb` (now git-clone workflow: clones `SleepyEveryD/NLP`, no Drive;
  load→warmup→demo→benchmark→accuracy/latency plots).
- ☑ RAN on Colab 2026-05-25 → **87.0% (20/23)**, latency median 0.91s (0 budget violations). Logged in
  `experiments.md`. Weak spots: Maths 0.50, Science 0.75. ☐ NEXT: inspect the 3 misses (failure modes +
  overconfidence) — user chose this before picking Phase 2 vs Phase 3.

## Phase 2 — Prompt engineering  ◐  (code DONE 2026-05-25; awaiting Colab run)
- ☑ Add strategies `few_shot_v1` (3 exemplars, no dev-set leakage) + `cot_v1` ("think briefly → Answer: X")
  to `PromptBuilder` registry + `_render_mcq` helper. Documented in `prompts.md`.
- ☑ `notebooks/02_prompt_engineering.ipynb`: loads model ONCE, benchmarks all 3 strategies on the dev set,
  compares overall + topic×strategy + **Maths-in-focus** + latency/tokens. ☑ RAN + recorded (run `prompt_eng`).
- ⊘ KEY read-off SUSPENDED: the cot 61% was a PARSER BUG (parse_answer `_EXPLICIT` regex can't match
  "Answer:", falls to pattern-7 which grabs the article "a"→"A"). cot's reasoning is mostly CORRECT and it
  even solved 17×13 & 2^10. Re-judge "does CoT fix maths / best prompt" AFTER a parser fix + re-parse.
- 🔴 [P2-bug] FIX `parse_answer` then RE-PARSE saved records.jsonl (no model re-run) → true zero/few/cot
  numbers. Latent landmine for RAG/tool prose paths → do this BEFORE Phase 3. Details: experiments.md CORRECTION.
- ☐ `concise_v1` + difficulty-adaptive prompting experiment (simple vs harder rungs). (backlog)
- ☐ Prompt-sensitivity study across ≥2 models. → `experiments.md`, `prompts.md`. (later, with model pool)

## Phase 3 — Tools (calculator)  ☐
- ☐ `QuestionClassifier.needs_calculator` (regex/number heuristics).
- ☐ Tool-call loop in pipeline (detect → call `calculate()` → feed result → re-answer).
- ☐ Ablation: maths-question accuracy with vs without calculator. → `experiments.md`.

## Phase 4 — RAG (raw evidence only)  ☐
- ☐ Build/clean a corpus (Wikipedia/PDF/HTML) OR pick a free raw-content search API (name it in video).
- ☐ `Retriever` with multilingual-e5 + FAISS; chunking strategy.
- ☐ `needs_retrieval` gating + context injection in `PromptBuilder`.
- ☐ Ablation: accuracy with vs without RAG, latency impact. → `experiments.md`.

## Phase 5 — Ensemble voting (if latency allows)  ☐
- ☐ Compare ≥2–3 models (Qwen, Mistral, Gemma/Phi) via the same `LLMEngine`.
- ☐ `majority_vote` / confidence-weighted vote.
- ☐ Latency check: does N forward passes fit in 30s? If not, document the tradeoff. → `experiments.md`.

## Phase 6 — Polish & deliverables  ☐
- ☐ Final submission notebook (self-explanatory, group names/emails, video link, coding-assistant statement).
- ☐ Export `.ipynb` + `.html`.
- ☐ Record 5-min screen-capture walkthrough; optional demo-run video.
- ☐ Fresh leaderboard run after the reset.

## Investigation questions to answer in the notebook (from the PDF — the "depth" rubric)
30s feasibility · model A/B · size vs quality · vs-human · best prompt (zero/few/CoT) · prompt
sensitivity · hard-question failure modes · per-topic strengths · overconfidence · adaptive prompting ·
RAG lift · calculator lift · thinking vs non-thinking · ensemble reliability · (optional) fine-tuning ·
(later) audio STT/TTS effect on accuracy & latency · web vs text interface.

## Backlog / optional
- ☐ `LlamaCppEngine` (GGUF) speed comparison vs transformers.
- ☐ Fine-tuning (LoRA) feasibility note + candidate training data.
- ☐ Audio bonus (speech mode is LIVE): `mode="speech"` → fetch WAV → Whisper STT → pipeline → answer.
  Timer starts at option D. Measure transcription accuracy + latency hit vs the text path.
