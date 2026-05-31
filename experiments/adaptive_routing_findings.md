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

> This is the offline-experiment default (`DEFAULT_ROUTING_POLICY`). The **live Maths** policy
> (`MATHS_LIVE_POLICY`) is more conservative and differs deliberately: only counting/temporal/enumeration
> are rerouted (to structured); `logical_reasoning` stays on `cot_v2`, because §5c showed checklist
> *regresses* logic on the real model.

**Dataset** (`data/reasoning_eval.jsonl`) — 40 core MCQs (5 per category) + 13 logic questions for the
§5c focused A/B, each with a hand-verified gold answer **and** a gold reasoning-category label. The
clock-chime question is `ic-001`; the live induction death is `log-ind-001` (= qid 6737).

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

### 5b. Real model (Qwen2.5-7B-Instruct, Colab GPU, 512-token cap)

Strategy comparison (40-question set, real model):

| condition | accuracy | mean latency (s) | mean tokens-out |
|---|---|---|---|
| A_universal (few_shot) | 0.750 | 1.4 | 2 |
| B_generic_cot | 0.875 | 17.8 | 207 |
| **C_structured** | **0.950** | 13.0 | 130 |
| **D_adaptive** | **0.950** | 14.2 | 158 |

**Truncation was the dominant hidden factor.** At the engine-default 256-token cap, the verbose chains
never reached the `Answer:` line (the `no_answer_parsed` mode): B_generic_cot scored only 0.700 and
D_adaptive 0.775. Raising the cap to 512 (free under the game's 130s/question budget) lifted them to
0.875 and 0.950 — the single biggest swing in the study. **Lesson: give the chain room to finish before
blaming the prompt.**

**On the Maths question types specifically** (arithmetic / temporal / interval / discrete, 20 Q),
adaptive is the clear winner — because no single prompt is best across them:

| | arithmetic | temporal | interval_counting | discrete | mean |
|---|---|---|---|---|---|
| B_generic_cot | 1.0 | 0.8 | **0.4** | 1.0 | 0.80 |
| C_structured | 0.8 | 0.8 | **1.0** | 1.0 | 0.90 |
| **D_adaptive** | **1.0** | 0.8 | **1.0** | 1.0 | **0.95** |

`interval_counting` (the clock-chime family that capped live Maths at level 9) goes **0.4 → 1.0** with
structured enumeration; arithmetic prefers `generic_cot` (1.0 vs structured 0.8). Adaptive routes each to
its winner.

**Hypothesis scorecard:** H1 (adaptive ≥ universal) **holds** — D ties the best fixed prompt overall and
wins on Maths types. H2 (structured rescues counting) **holds** decisively. H3 (structured hurts factual
via overthinking) **does NOT hold** — structured kept `factual_qa`/`commonsense` at 1.0 on this model.
H4 (checklist helps logic/multi-hop) **does NOT hold** — see §5c.

**Live confirmation (notebook 03, real game):** with the conservative Maths routing deployed
(counting/temporal/enumeration → structured, else → cot_v2, 512 tokens), Maths reached **level 11**
(11/12 correct), up from the prior best of 9 — score 16,000 → 64,000. The one miss was a strong-induction
question (qid 6737), a different failure type (see §5c), not a counting miss.

### 5c. Focused A/B on logic questions — checklist refuted, an induction ceiling found

After the counting deaths were fixed, the live Maths run died at level 11 on a strong-induction /
contrapositive question (qid 6737 = `log-ind-001`): under `cot_v2` the model emitted a 24-token wrong
one-liner ("n0 is a counterexample, so all higher values are also counterexamples" → C; correct is D).
The brevity cap looked like the culprit, so we A/B'd four prompts over 13 logic questions
(`LOGIC_FOCUS_CONDITIONS`, `experiments/adaptive_routing_logic/`):

| arm | accuracy (13 logic Q) |
|---|---|
| cot_v2 (current fallback) | **0.846** |
| generic_cot (no cap) | 0.846 |
| checklist_cot | **0.769** (regressed) |
| checklist_cot + self-consistency n=5 | 0.846 |

Per-question outcome (real model):
- **qid 6737 is wrong under ALL FOUR prompts** — removing the brevity cap, switching to a checklist, and
  voting over 5 chains all fail. This is a **capability ceiling of the 7B model** on contrapositive
  direction, not a prompt-format problem. (Same for `lr-001`, a quantifier-fallacy trap.)
- **`checklist_cot` is a net LOSS** (0.769 < 0.846): it did not fix the induction questions and it broke
  one (`log-ind-003`) that `cot_v2` got right. So routing Maths logic → checklist would *regress* the
  live game — the offline A/B prevented a bad deploy.
- **Self-consistency only recovers stochastic errors, not systematic ones.** SC voted `log-ind-003` back
  to correct (a noisy miss) but could not touch 6737 — the model is *confidently* wrong there, so all 5
  chains agree on the wrong answer. A useful general limit of SC.
- All three stats look-alikes (`log-stat-*`) are correct under every prompt — robust, not at risk.

**Decision:** keep `cot_v2` as the Maths fallback for logic (tied-best, and checklist regresses). Do not
reroute logic. 6737-type questions are accepted as a model ceiling, not chased with prompt engineering.

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

## 7. Recommendation (as deployed)

1. **Maths: route counting to structured enumeration, keep everything else on `cot_v2`.** Deployed in
   `notebooks/03_live_play.ipynb` via `RoutingPromptBuilder(policy=MATHS_LIVE_POLICY)`:
   `interval_counting` / `temporal_reasoning` / `discrete_enumeration` → `structured_enumeration_cot`;
   arithmetic, logic and concept/stats → the `cot_v2` fallback. This fixed the clock-chime death and
   took live Maths from level 9 → 11 (§5b). It is deliberately conservative — §5c showed rerouting
   *logic* to checklist would **regress** the game, so logic stays on `cot_v2`.
2. **Set `max_new_tokens ≥ 512` wherever a CoT/structured chain runs.** The 256 default truncates the
   chain before the answer line — the biggest single accuracy leak (§5b). The 130s/question budget makes
   512 free. `QAPipeline` now takes an optional `max_new_tokens`; the Maths pipeline sets 512.
3. **Never make `structured_enumeration_cot` the universal default** — and don't reroute logic to
   checklist. Structured ≈ best on Maths types but checklist *lost* on logic. The other five
   competitions keep `few_shot_v1` (unchanged).
4. **Don't chase the level-11 induction question with prompting.** §5c shows it is a 7B capability
   ceiling — no prompt or self-consistency vote moves it. The free lever for climbing past it is
   live-run variance (re-run; leaderboard keeps best).

## 8. Limitations

- **Small dataset (40 + 13 logic Q)** — directional, not significance-grade. Per-cell accuracy is
  ~5-question granular; treat per-category numbers as qualitative.
- **Router validated on its own dev set** — the 1.0 routing accuracy reflects rules tuned against these
  questions (including a mid-study fix: induction questions were leaking to `arithmetic` via an
  incidental `+1`, and stats MCQs via hyphens reading as the minus operator — logic now checks before
  arithmetic). On unseen questions expect lower, especially **multi_hop**, which surface-overlaps with
  single-fact recall. A learned router is the upgrade path.
- **Self-consistency helps only stochastic errors** — it recovered a noisy logic miss but not the
  systematic induction error (§5c). Budget it for genuinely uncertain questions, not confident-wrong ones.
- **Isolation is deliberate** — the offline harness uses no retrieval/calculator, so absolute accuracies
  differ from live play. The comparison *between conditions* is what is valid.
- **§5a is a fixture** — only §5b/§5c (the Colab runs) yield real model findings.

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
