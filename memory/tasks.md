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
- ☐ `notebooks/00_setup_colab.ipynb`: mount Drive, `sys.path` (repo `src` + `millionaire_client`),
  `pip install -r requirements.txt`, load Qwen2.5-7B 4-bit, `warmup()`, one smoke-test generation, print latency.
- ◐ Smoke-test the real client: login → `list_competitions()` ☑ DONE 2026-05-25 (Colab; auth OK, 6
  competitions listed — see `api_contracts.md §C`). Starting a game + printing one question is DEFERRED
  on purpose until the answering pipeline is ready (don't burn the 30s timer). Confirm option ids +
  `time_remaining` behave as documented when we do the first real game.
- ☐ Verify VRAM fit + cold/warm latency on a real T4/L4; measure network RTT to the game server.

## Phase 1 — Baseline single-model QA (MVP)  ☐
- ☐ Implement `TransformersEngine` (4-bit load, chat template, greedy generate, token counts).
  Use the course's exact `BitsAndBytesConfig` nf4 recipe (`techniques.md` Phase 1; D-012).
- ☐ Implement `PromptBuilder` `zero_shot_v1` (MCQ → ask for a single letter).
- ☐ Implement `QAPipeline.answer()` (classify→prompt→generate→parse) + `parse_answer()`.
- ☐ Implement `ExperimentLogger` wiring + `BenchmarkRunner.run()`.
- ☐ Implement `metrics.load_runs/accuracy_by/latency_summary`.
- ☐ Build a small **dev question set** (raw, hand-checked) to measure accuracy offline.
- ☐ `notebooks/01_baseline_qa.ipynb`: run baseline, report accuracy + latency. → log to `experiments.md`.

## Phase 2 — Prompt engineering  ☐
- ☐ Add strategies: `few_shot_v1`, `cot_v1` ("think briefly, then answer"), `concise_v1`.
- ☐ Difficulty-adaptive prompting experiment (simple vs harder rungs).
- ☐ Prompt-sensitivity study across ≥2 models. → `experiments.md`, `prompts.md`.

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
