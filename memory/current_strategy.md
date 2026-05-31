# Current Strategy — Live-play configuration (post-revert)

> Snapshot of how `notebooks/03_live_play.ipynb` answers each competition.
> Last updated: 2026-05-30 (after the Maths revert to the 5bcf593 known-good config).

---

## 1. The shape of it

One model (`Qwen/Qwen2.5-7B-Instruct` 4-bit nf4, greedy), **two pipelines**, **routed by competition_id**:

```
              competition_id
                    │
       ┌─────── cid == 3 ───────┐
       │                        │
   pipeline_maths           pipeline (default)
   (Maths only)             (all 5 other competitions)
```

The router lives in the sweep cell:

```python
pipeline_for=lambda cid: pipeline_maths if cid == 3 else pipeline
```

Both pipelines share the same `LLMEngine`, `QuestionClassifier`, and 30s latency budget.
They differ on **prompt strategy**, **retrieval**, and **tools**.

---

## 2. Default pipeline (Entertainment, Ancient History, Science, Philosophy, News)

| Knob | Setting | Why |
|---|---|---|
| `prompt_strategy` | `few_shot_v1` (from `configs/live.yaml`) | Offline sweep winner (0.870 on dev set), generalises across the 5 mixed-topic comps. |
| `retriever` | `build_retriever(config.retrieval)` — `source="routed"` | Per-question router: News → live web (DuckDuckGo), else → FAISS Simple-Wikipedia (`data/corpus/simplewiki`). `needs_retrieval()` gates each call. |
| `tools` | `default_tools()` — safe-AST calculator | Fires only when `needs_calculator()` matches arithmetic cues. On non-arithmetic comps it's a no-op. |
| `self_consistency_n` | 1 (default) | Single greedy pass; SC adds latency without measurable accuracy lift here. |
| `latency_budget_s` | 30.0 | Hard wall of the game. |

**News specifically**: the retriever routes News questions to live web (DuckDuckGo) because post-cutoff facts (2026) aren't in Wikipedia. ISO date `2026-MM-DD` in the question is the tell — see `classifier._RETRIEVAL_NEWS_RE`.

**Calculator specifically (default pipeline)**: when triggered, it runs as a **match-gated verifier** (added in commit `403b989`, see D-017 below). The model emits a single arithmetic JSON call; the result must uniquely match one option's numeric value to override the chain — otherwise the original answer stands. The match gate exists to avoid the run-#8 t-test clobber (calc returning "5" overriding a correct conclusion-style answer).

---

## 3. Maths pipeline (comp 3) — reverted to 5bcf593

| Knob | Setting | Why |
|---|---|---|
| `prompt_strategy` | `cot_v2` | Step-by-step reasoning + option-matching directive + hard brevity cap (≤3 steps, no LaTeX). Designed for Maths concept questions (run #7/#9 fixes). |
| `retriever` | `None` | Wikipedia hits distract on formal-maths questions (group theory, topology). |
| `tools` | `None` | **NO calculator** — at n=1, the calculator can clobber `cot_v2`'s reasoning on stats/group-theory questions whose answers aren't numbers but conclusions using numbers. |
| `self_consistency_n` | 1 | SC at n≥3 with `cot_v2` (~20s/chain) breaks the 30s wall — run #8 timed out on a question it had answered correctly. |

**Console signature on startup**:
```
Maths pipeline (comp 3): cot_v2 + single-pass (n=1) + NO retrieval + NO calculator
```

---

## 4. What we tried after 5bcf593 and reverted (lessons captured)

These two changes shipped between `5bcf593` (Maths 9/10 reach=9) and the current revert. Both showed mixed results and neither beat the 5bcf593 baseline reliably; rolled back until we have stronger evidence.

### 4.1 Calculator as match-gated verifier (commit `403b989`)

- **What changed**: replaced the 2-turn "compute → re-answer" pattern with a 1-turn match-gated verifier. The calc result must uniquely map to an option's numeric value to override `cot_v2`'s answer.
- **Why we tried it**: the 2-turn re-answer was clobbering correct chains on questions where the answer is a *conclusion using a number*, not the number itself (e.g. t-test: `p < 0.05` → "reject H0", not "5").
- **Why it didn't help Maths**: introduced new failure modes — e.g. **Q6767** (group theory, `|(Z_4 × Z_12) / (⟨2⟩ × ⟨2⟩)|`), the model emitted `4*12/(2*2) = 12` mapping to option C, but `|⟨2⟩ in Z_12| = 6` not 2, so the correct answer is `48 / (2·6) = 4` (option A). The calculator amplified a knowledge gap into a confident wrong answer.
- **Status for default pipeline**: KEPT (still active on the 5 non-Maths comps where the match-gate is a net win).
- **Status for Maths**: REMOVED (`tools=None`).

### 4.2 `cot_maths_v1` prompt strategy (commit `5130bac`)

- **What changed**: new prompt strategy that prepends two worked Maths exemplars (ratio with no numbers; rectangle area) to `cot_v2`'s directives, plus a "introduce variables when no numbers are present" instruction.
- **Why we tried it**: **Q6777** — "speed/price ratio is what percent of …" — `cot_v2` alone never set up variables because the question carries no numbers, so the chain skipped straight to a wrong percentage.
- **Partial success**: on the next Q6777 encounter the chain DID introduce variables (`Let s = speed, p = price`) and DID write the relationships (`Five years ago: s/2, 2p`) — the exemplar anchored the setup.
- **Why it didn't finish the job**: the chain then slipped on the *algebra* step (`Ratio then: s/(2p)` — dropped the `/2` on speed) and the *option mapping* (computed 200%, picked B=32 which doesn't even appear in its own work). Symbolic algebra under the brevity cap is too fragile.
- **Status**: registered in `_REGISTRY` but unused. `cot_v2` is back in the Maths slot.

---

## 5. Confirmation checklist for a Maths run

After `Restart runtime → Run all` (or pull + restart), the wire cell must print:

```
Maths pipeline (comp 3): cot_v2 + single-pass (n=1) + NO retrieval + NO calculator
```

And each record in the run's JSONL should show:

```json
"prompt_strategy": "cot_v2",
"retrieval_used": false,
"retrieved_doc_ids": [],
"tool_used": null
```

If `prompt_strategy` is `cot_maths_v1` or `tool_used` is `"calculator"`, the wire cell wasn't re-executed after pull (Colab notebook autosave can overwrite local changes — `Runtime → Restart` is the only reliable fix).

---

## 6. Pointers

- Wire cell: `notebooks/03_live_play.ipynb` cell id `code-wire`
- Sweep router: same notebook, the cell calling `run_all_competitions(..., pipeline_for=...)`
- Prompt strategies: `src/prompting/builder.py` (`_REGISTRY`)
- Calculator: `src/tools/calculator.py` + `src/agent/pipeline.py::_run_calculator_tool`
- Retrieval routing: `src/retrieval/retriever.py`
- Classifier gates: `src/classify/classifier.py` (`needs_calculator`, `needs_retrieval`)
- Per-run records: `experiments/runs/live_comp{0..5}/records.jsonl`
- Decisions referenced: D-006 (DI), D-008 (raw evidence only), D-013 (tool loop), D-017 (calculator verifier), D-NEWS (web for News)
