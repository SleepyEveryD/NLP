# Prompt Library

> The human-readable record of every prompt strategy. The code form lives in
> `src/prompting/builder.py` — keep the two in sync (same strategy name = same prompt).

Naming: `<style>_v<N>` (e.g. `zero_shot_v1`). Bump the version on any wording change so logged
runs stay reproducible (the `prompt_strategy` field in `EvalRecord` points here).

---

## zero_shot_v1  (Phase 1 baseline — MCQ)
**Goal:** force a single-letter answer, minimal tokens (latency-friendly).
**Sketch:**
```
System: You are an expert quiz contestant. Answer accurately.
User:   Question: {text}
        Options:
        A) {A}
        B) {B}
        C) {C}
        D) {D}
        Reply with ONLY the letter of the correct option.
```
**Parse:** first standalone A/B/C/D in the output. Confidence: heuristic (1.0 if clean single letter).

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
