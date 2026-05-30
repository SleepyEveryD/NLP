# Adaptive Prompt Routing for a Small Reasoning Model — Findings

> A study of whether routing each question to a prompt chosen for its *reasoning shape* beats forcing
> one universal prompt on a 7B model (Qwen2.5-7B-Instruct, 4-bit), and of **when explicit reasoning
> helps and when it hurts**.

---

## 1. Motivation (a real, on-the-leaderboard failure)

The live Maths run capped at **level 9 / 16,000** when it died at level 6 on an *interval-counting*
question (`temp/records (4).jsonl`, qid 6712):

> *"A clock chimes the hour-number every hour and once per 15-minute mark; how many chimes between
> 5:10 and 7:35 P.M.?"* — correct **C (21)**.

The model wrote `"Step 2: Count additional chimes..."` and then jumped straight to **`Answer: B (18)`**
without ever counting (`tokens_out = 54`). The cause was not a knowledge gap — it was the prompt:
`cot_v2` carries a hard **"Solve in AT MOST 3 very short steps"** brevity cap, introduced to stop a
*different* failure (a t-test question that blew its token budget writing LaTeX and got truncated
before the answer line). That cap helps concept/stats questions and **actively breaks enumeration**:
it orders the model to commit to an answer before it has finished listing the cases.

One prompt was being asked to serve two opposite regimes. That tension is the hypothesis this
experiment isolates and tests.

## 2. Research questions

- **RQ1.** Does **adaptive** prompt routing beat a single **universal** prompt for a small model?
- **RQ2.** **When** does explicit reasoning (CoT, enumeration, checklists) help, and when does it hurt?

## 3. Hypotheses

- **H1.** No single prompt is best across reasoning shapes — so an adaptive router that picks the
  per-shape winner beats any fixed prompt.
- **H2.** *Structured enumeration* rescues interval-counting / temporal / discrete-enumeration
  questions (it forces the listing the brevity cap suppressed).
- **H3.** The same enumeration prompt **hurts** factual recall and commonsense — explicit reasoning
  invites hallucinated justification and drift where a direct answer would have been right
  ("overthinking").
- **H4.** Checklist-style verification helps logical-reasoning and multi-hop questions (it forces the
  model to test each option/hop instead of committing early).

## 4. Method

**Design.** One independent variable — the **prompt strategy**. To keep it clean, the experiment
harness (`src/experiments/adaptive_routing.py`) deliberately runs **no retrieval and no calculator**
(both of which the live `QAPipeline` uses); every condition sees the identical question with only the
prompt changed.

**Four conditions (ablation):**

| Condition | Prompt | Role |
|---|---|---|
| `A_universal` | `few_shot_v1` (the production single prompt) | the "one prompt for all" baseline |
| `B_generic_cot` | `generic_cot` ("think step by step") | the vanilla-CoT control |
| `C_structured` | `structured_enumeration_cot` | always-enumerate control |
| `D_adaptive` | `ReasoningRouter` picks per question | the treatment |

**Prompt strategies** (`src/prompting/builder.py`):
- `direct_answer` — letter only, no reasoning.
- `generic_cot` — plain step-by-step, no cap, no exemplars.
- `structured_enumeration_cot` — list **every** case/event in order, **boundary-check** the endpoints,
  count **only after** listing. (The clock-chime cure.)
- `checklist_cot` — restate → assumptions → evaluate **each** option/hop → validate the pick.

**Reasoning classifier + router** (`src/classify/reasoning_router.py`) — rule-based (regex + keyword
precedence), transparent, well under the latency budget. Eight categories and the routing policy:

| Reasoning category | Routed strategy | Why |
|---|---|---|
| `interval_counting` | `structured_enumeration_cot` | enumerate events over a clock range |
| `temporal_reasoning` | `structured_enumeration_cot` | lay out the timeline |
| `discrete_enumeration` | `structured_enumeration_cot` | list cases before counting |
| `arithmetic` | `generic_cot` | compute stepwise, don't over-enumerate |
| `factual_qa` | `direct_answer` | recall; reasoning causes drift |
| `commonsense` | `direct_answer` | quick judgement; overthinking hurts |
| `logical_reasoning` | `checklist_cot` | test each option/assumption |
| `multi_hop` | `checklist_cot` | decompose and validate each hop |

**Dataset** (`data/reasoning_eval.jsonl`) — 40 MCQs, 5 per category, each with a hand-verified gold
answer **and** a gold reasoning-category label. The clock-chime question is `ic-001`.

**Logged per (condition, question)** — strategy used, router verdict + the cue that fired,
predicted/gold answer, correctness, confidence, latency, tokens in/out, reasoning length (chars and
lines), raw output. Everything re-analysable from one JSONL without re-running the model.

**Metrics & figures** (`src/experiments/analysis.py`) — strategy-comparison table, category×condition
accuracy heatmap, latency-vs-accuracy scatter, routing-accuracy + confusion, failure taxonomy.

## 5. Results

### 5a. Harness validation (deterministic fixture — NOT the model)

Run locally with `SimulatedReasoningEngine`, a deterministic fixture whose correctness comes from a
skill table that *encodes the hypotheses above*. **These numbers validate the pipeline (routing,
logging, metrics, figures), not the model.** They are reported only to show the harness produces the
right shapes; the real numbers come from §5b.

Strategy comparison (fixture):

| condition | accuracy | mean latency (s) | mean tokens-out | mean reasoning lines |
|---|---|---|---|---|
| A_universal | 0.525 | 0.7 | 8 | 1.0 |
| B_generic_cot | 0.600 | 5.0 | 55 | 3.0 |
| C_structured | 0.650 | 12.7 | 140 | 7.0 |
| **D_adaptive** | **0.800** | 8.1 | 89 | 4.5 |

Routing accuracy on the labelled set: **1.0** (the classifier rules were developed against this set —
see Limitations). The category heatmap shows the predicted pattern: `structured` is strong on
counting/temporal rows and weak on factual; `direct` (via A) is strong on factual and weak on counting;
`D_adaptive` tracks the per-row best. The latency–accuracy scatter shows `C_structured` paying the most
seconds for less accuracy than `D_adaptive`, which buys high accuracy at moderate latency.

Figures: `fig_condition_accuracy.png`, `fig_category_heatmap.png`, `fig_latency_accuracy.png`.

### 5b. Real model (Colab GPU) — **to be filled**

Set `USE_REAL_MODEL = True` in `notebooks/04_adaptive_routing.ipynb` on a T4/L4 runtime and re-run.
Paste the resulting `strategy_comparison_table` and the heatmap here. The hypotheses (H1–H4) are
confirmed iff: D ≥ max(A,B,C) overall; `structured` leads on the counting rows and trails on
`factual_qa`; `direct` leads `factual_qa`/`commonsense`; `checklist` leads `logical`/`multi_hop`.

## 6. Failure taxonomy (what the analysis labels)

From the raw output, each wrong answer is bucketed:
- **overthinking** — a recall/commonsense question answered with a long chain (reasoning where none
  helped). Expected concentrated in `C_structured` / `B_generic_cot` on `factual_qa`.
- **boundary_error** — a counting question wrong despite enumerating (off-by-one at an endpoint).
- **skipped_case** — a counting/logic question wrong with a short chain (cases never listed). Expected
  concentrated in `A_universal`.
- **arithmetic_drift** — arithmetic wrong despite shown work.
- **no_answer_parsed** — the generation never reached an `Answer:` line (truncation / format miss) —
  the original `cot_v2` LaTeX-blowup mode.
- **hallucinated/other** — a confident wrong fact.

In the fixture run the dominant pattern is exactly as predicted: `A_universal`'s errors are almost all
**skipped_case** (no reasoning → counting questions fail), while the always-CoT arms accumulate
**overthinking** errors on recall. `D_adaptive` carries the fewest, and zero overthinking.

## 7. Recommendation

1. **Adopt adaptive routing on the Maths pipeline first.** In `notebooks/03_live_play.ipynb`, replace
   the fixed Maths strategy with a per-question router: send `interval_counting` / `temporal_reasoning`
   / `discrete_enumeration` to `structured_enumeration_cot`, keep concept/stats arithmetic on the
   current chain. This directly fixes the clock-chime death without re-introducing the LaTeX-blowup
   that the brevity cap exists to prevent — because counting questions and stats questions are now
   handled by *different* prompts.
2. **Never make `structured_enumeration_cot` the universal default** — it over-reasons recall
   questions. The other five competitions keep `few_shot_v1` / `direct_answer`.
3. **Watch the latency budget.** Structured enumeration is the most verbose strategy
   (~140 tokens ≈ ~13 s at the 11 tok/s 4-bit rate). It fits the 30 s wall for a single counting
   question but must not be combined with self-consistency (n≥3) there.

## 8. Limitations

- **Small dataset (40 Q)** — directional, not significance-grade. Per-cell accuracy is 5-question
  granular; treat the heatmap as qualitative.
- **Router validated on its own dev set** — the 1.0 routing accuracy reflects rules tuned against
  these 40 questions. On unseen questions expect lower, especially **multi_hop**, which surface-overlaps
  with single-fact recall ("the capital of the country that…"); a learned router is the upgrade path.
- **Isolation is deliberate** — no retrieval/calculator here, so absolute accuracies differ from live
  play. The comparison *between conditions* is what is valid.
- **§5a is a fixture** — only §5b (the Colab run) yields real model findings.

## 9. Reproduce

```bash
# Local (fixture — validates the harness, fast, no GPU):
PYTHONPATH=src python3 -c "
from experiments.adaptive_routing import load_reasoning_eval, AdaptiveRoutingExperiment
from inference.engine import SimulatedReasoningEngine
from experiments import analysis
q,c = load_reasoning_eval()
AdaptiveRoutingExperiment(SimulatedReasoningEngine(q,c)).run(q,c)
print(analysis.run_full_analysis())"

# Real (Colab GPU): open notebooks/04_adaptive_routing.ipynb, set USE_REAL_MODEL=True, Run all.
```

Artifacts land in `experiments/adaptive_routing/`: `records.jsonl`, `meta.json`, and the three PNGs.
