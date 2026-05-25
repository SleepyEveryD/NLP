# Prompt Library

> The human-readable record of every prompt strategy. The code form lives in
> `src/prompting/builder.py` — keep the two in sync (same strategy name = same prompt).

Naming: `<style>_v<N>` (e.g. `zero_shot_v1`). Bump the version on any wording change so logged
runs stay reproducible (the `prompt_strategy` field in `EvalRecord` points here).

---

## zero_shot_v1  (Phase 1 baseline — MCQ)  ✅ IMPLEMENTED (`src/prompting/builder.py`)
**Goal:** force a single-letter answer, minimal tokens (latency-friendly).
**Convention:** `build()` returns the **user-turn content only** — NO system message, NO chat template.
`TransformersEngine.generate()` wraps it via `apply_chat_template([{user, content}], add_generation_prompt=True)`.
**Exact text (MCQ, no context):**
```
Question: {text}
A) {A}
B) {B}
C) {C}
D) {D}
Reply with ONLY the letter of the correct option (A, B, C, or D). No explanation, no punctuation -- the letter alone.
```
**With RAG context (D-008, raw evidence only):** a `Referenced knowledge:` block of numbered raw doc
texts is prepended before `Question:`. **Open question:** `Answer briefly in one or two sentences.`
**Parse (`QAPipeline.parse_answer`):** priority regexes — `answer is X` / `(X)` / `Option X` / `X)` /
letter+option-text / standalone-line letter / any isolated A–D; restricted to letters present in
`options`; confidence 1.0 clean → 0.5 ambiguous → 0.0 fallback to first option (never empty for MCQ).

## few_shot_v1  (Phase 2 — planned)
Prepend 2–3 solved MCQ exemplars (diverse topics) before the target question.

## cot_v1  (Phase 2 — planned)
"Think step by step briefly (≤2 sentences), then give 'Answer: X'." Parse the letter after "Answer:".
Watch the token cap — CoT costs latency.

## concise_v1 / difficulty-adaptive  (Phase 2 — planned)
Short prompt for easy rungs; richer/CoT prompt for hard rungs. Tests the "same prompt for all?" question.

---

## Observations on prompt sensitivity
_(append findings: which models are robust vs brittle, which phrasing helps, etc.)_
