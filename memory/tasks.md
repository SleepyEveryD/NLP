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
- ◐ [P2-bug] FIX `parse_answer` ☑ DONE 2026-05-25: `_EXPLICIT` regex demanded a SPACE before the colon,
  so cot's own "Answer: B" it MISSED → fell to pattern-7, the article "a" as "A" grabbing. Tightened to two
  branches (`answer:` straight, or `answer is`); "answer a question" no longer a false-positive (verified).
  RE-PARSE tool written → `src/evaluation/reparse.py` (the saved `raw_output` through the fixed parser it
  replays, no model re-run). ☐ NEXT: run it on Colab over the `prompt_eng` records → true zero/few/cot
  numbers (which strategy is truly best, this may change). Then safe for RAG/tool prose paths, we are.
- ☐ `concise_v1` + difficulty-adaptive prompting experiment (simple vs harder rungs). (backlog)
- ☐ Prompt-sensitivity study across ≥2 models. → `experiments.md`, `prompts.md`. (later, with model pool)

## Phase 3 — Tools (calculator)  ◐  (code DONE 2026-05-25; ablation awaits a Colab run)
- ☑ `QuestionClassifier.needs_calculator` (regex/number heuristics) — already present; verified True on
  "17 * 13", False on non-maths.
- ☑ Tool-call loop in pipeline (D-013 JSON single-turn): `QAPipeline._run_calculator_tool` + `tools.default_tools()`
  (`{"calculator": calculate}`). Model emits `{"name","arguments":{"expression"}}` JSON → safe-AST `calculate()`
  → result fed back → re-answer for the letter. Crash-safe (bad JSON / forbidden expr / calc error → the plain
  answer stands); skipped when <4s budget left. `_extract_first_json` brace-counts (tolerates chatter). 6
  mock-engine tests pass (no GPU). Live notebook 03 now wires `tools=default_tools()` (a no-op off the maths comp).
- ☐ Ablation: maths-question accuracy with vs without calculator (toggle `tools=` in a benchmark on Colab). → `experiments.md`.

## Phase 4 — RAG (raw evidence only)  ◐  (Wikipedia DONE 2026-05-26; ROUTED web+FAISS DONE 2026-05-26 on `4-rag`; Colab run + index build await)
- ☑ Data source: **routed, three RAW-content backends** (all rule-compliant; NAME ALL IN THE VIDEO):
  **DuckDuckGo** (live web, post-cutoff News) · **Wikipedia Action API** (knowledge fallback) ·
  **Simple-Wikipedia + FAISS** (local dense corpus). User chose "both" (2026-05-26) after the sweep showed
  Wikipedia-only left **News at 2/7** — it cannot hold breaking 2026 facts. Web search closes that gap.
- ☑ `src/retrieval/retriever.py` REWRITTEN (was a `NotImplementedError` stub — so RAG had NEVER truly run
  before the sweep used `WikipediaRetriever` directly): `WebSearchRetriever` (DDG HTML, keyless, `search_fn`
  override hook), `FaissRetriever` (multilingual-e5 + FAISS, lazy load, e5 query/passage prefixes), and
  `Retriever` facade that ROUTES per question — News→web(+wiki fallback), else→FAISS(+wiki fallback). All
  crash-safe (`[]` on any failure). `build_retriever(RetrievalConfig)` factory. `WikipediaRetriever` reused.
- ☑ `src/retrieval/build_index.py`: corpus JSONL → `<out>/index.faiss` + `docs.jsonl` (e5 "passage:" prefix,
  chunking, normalized IP=cosine). Run on Colab: `python -m retrieval.build_index --corpus simplewiki.jsonl
  --out data/corpus/simplewiki --text-field text --id-field title --source-field title`.
- ☑ `needs_retrieval` now also fires on the **News recency signature** (ISO date / "according to..article" /
  "published on") — live play leaves `topic` unset, so the text must trigger it (else News skips retrieval).
  Same signal the router reads → gate and route stay in step.
- ☑ Config: `live.yaml` now `source: "routed"`, `index_path: "data/corpus/simplewiki"`. Notebook 03 wire cell
  uses `build_retriever(config.retrieval)`. `index_path` null / missing → graceful Wikipedia fallback.
- ◐ **VERIFY ON COLAB** (no local Python): build the FAISS index, then re-run the sweep with `source="routed"`
  and confirm (a) News retrieval_used=True + web snippets land, (b) News accuracy lifts above 2/7,
  (c) DDG not blocked on Colab IP (if 429/blocked → swap `WebSearchRetriever(search_fn=...)` for a news RSS/API).
  ◐ PARTIAL (run #3, 2026-05-26): routed sweep ran → **News reached_level 1→3** (web path lifts News, (b) ✅ partial).
  BUT (a)/(c) UNVERIFIED — the wrong-dump lacked the tool/retrieval trace, so for qid 11194 we can't tell gate-miss
  vs DDG-blocked vs snippet-miss. ☑ FIXED: notebook 03 dump now prints/saves `tool_used`/`retrieval_used`/
  `retrieved_doc_ids` per wrong Q + a per-comp usage table + per-question detail for ALL comps (was Maths-only).
  ☐ NEXT: re-run instrumented sweep, read the News trace; if `retrieval_used=True`+`docs_landed=0` on News → DDG blocked.
- ☑ Data source (legacy note): live Wikipedia Action API — kept as the knowledge-topic backend + News fallback.
- ☑ `WikipediaRetriever` (`src/retrieval/wikipedia.py`): search → fetch intro extracts (`explaintext`) →
  `RetrievedDoc(title, text[:700], wiki-URL, score)`. ENTITY-FIRST query (proper nouns / quoted titles first,
  then the full question) so abstract phrasings still hit ('M3GAN', 'Marriage Story' now resolve). Crash-safe:
  ANY error → `[]` (the live turn never sinks). 429-aware: ONE short capped retry (<=2s), then graceful degrade.
- ☑ `needs_retrieval` gating (already in classifier) + context injection (`_build_context_block`, already in
  every prompt builder) → wired via DI. Config: `retrieval.enabled/source` (+ `RetrievalConfig.source`);
  `live.yaml` enabled=true, source=wikipedia. Notebook 03 wire cell builds it from config (flag = the ablation).
- ☑ Verified end-to-end (FakeEngine): factual Q → context injected ("Referenced knowledge: …"), and the raw
  extract CONTAINED the answer (e.g. West End Blues → "...recording...June 11, 1928" = the gold option).
- ⚠️ HONEST FINDINGS (good rubric material): helps entity/fact recall (Eminem, Louis Armstrong, Roman weddings);
  does NOT reliably solve very-recent **News** (qid 11239 2026-05-17 event absent from Wikipedia → near-miss
  context); returns NOISE on pure-reasoning Qs (4902 sci-method) so gating matters; sweep hammering Wikipedia
  triggers **429** (mitigated by retry + graceful degrade, but RAG silently degrades under heavy bursts).
- ☐ Ablation: accuracy with vs without RAG (flip `retrieval.enabled`), + latency impact + per-topic where it
  helps/hurts (watch over-firing on reasoning Qs). → `experiments.md`. RUN ON COLAB.

## Phase 5 — Ensemble voting (if latency allows)  ◐  (self-consistency DONE 2026-05-27 on `4-rag`; multi-model still ☐)
- ☑ `majority_vote` IMPLEMENTED (`agent/voting.py`): most-voted wins, ties by mean confidence; builds a
  fresh Prediction with confidence = VOTE SHARE. The SHARED vote primitive (self-consistency + multi-model).
- ☑ **Self-consistency** wired (D-016): `QAPipeline` gains `self_consistency_n`/`self_consistency_temperature`
  (n=1 default = no change). n>1 → N sampled chains, parse each, `majority_vote`; tool stage skipped under SC.
  Notebook 03 `pipeline_maths` (comp 3) → cot_v1 + `self_consistency_n=3` (T=0.7). ☐ VERIFY ON COLAB: clean
  sweep → does Maths climb past lb 3? Read comp 3 detail: `strat=cot_v1`, `tool=None`, `retr=False`, conf=vote-share.
- ☐ Compare ≥2–3 models (Qwen, Mistral, Gemma/Phi) via the same `LLMEngine`, then `majority_vote` across them.
- ☐ Latency check: does N forward passes fit in 30s? (self-consistency n=3 ≈ 12s, confirmed in budget.) → `experiments.md`.

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
