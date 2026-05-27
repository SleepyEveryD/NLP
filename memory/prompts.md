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

## few_shot_v1  (Phase 2)  ✅ IMPLEMENTED (`src/prompting/builder.py`)
Prepend **3 solved MCQ exemplars** (capital-of-France / photosynthesis gas / 6×7), each ending `Answer: X`,
then the target question + `Answer with ONLY the letter (A, B, C, or D).` Exemplars are from OUTSIDE the
dev set (no leakage). One includes arithmetic to prime the format. RAG-context block prepended if present.

## cot_v1  (Phase 2)  ✅ IMPLEMENTED (`src/prompting/builder.py`)
MCQ block + `Think step by step briefly (one or two short sentences). Then on a new line, write your
final choice as 'Answer: X', where X is one of A, B, C, or D.` The parser's top-priority regex keys on the
`Answer: X` marker. Token cap 256 default covers brief CoT; latency has ~30x headroom so cost is a non-issue.

## cot_v2  (Phase 2/5 — option-matching CoT)  ✅ IMPLEMENTED (`src/prompting/builder.py`)
cot_v1 + an explicit **option-matching check**. MCQ block + `Think step by step briefly (one or two short
sentences). Before deciding, CHECK the options against your reasoning: when two options state the same
conclusion, the correct one must ALSO match every detail you computed -- numbers, degrees of freedom, signs.
Pick the option that matches your work in FULL, not just the conclusion. Then on a new line, write your final
choice as 'Answer: X', where X is one of A, B, C, or D.` Open questions: identical to cot_v1 (no options).
**Why (run #7, qid 6702):** on a t-test MCQ the model reasoned correctly (df=17, ±2.110 = option C) but wrote
`Answer: B`, whose only flaw was `df=18` -- B and C shared the conclusion and it matched the conclusion alone.
cot_v2 targets these adversarial near-identical distractors. Used by the live Maths pipeline (comp 3) under
self-consistency (n=3); **cot_v1 stays as the control** for the cot_v1-vs-cot_v2 ablation. ☐ Measure the lift.

## concise_v1 / difficulty-adaptive  (Phase 2 — planned)
Short prompt for easy rungs; richer/CoT prompt for hard rungs. Tests the "same prompt for all?" question.

---

## Observations on prompt sensitivity
_(append findings: which models are robust vs brittle, which phrasing helps, etc.)_
