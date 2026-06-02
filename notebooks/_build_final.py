"""Merge the 5 narrative notebooks into one self-explanatory deliverable.

Strategy (user-chosen): PRESERVE OUTPUTS. We concatenate the curated analysis
cells from each notebook in story order, keep exactly ONE clone/install/paths
setup (from 05, already BRANCH='science'), drop every other duplicate setup
cell, and bracket each experiment with a section divider. Outputs are kept as-is
(offline experiments are NOT re-run); only the live leaderboard run (§8) is
refreshed on Colab.
"""
import json, copy, os

HERE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)

NB = {
    "00": load("00_setup_colab.ipynb"),
    "01": load("01_baseline_qa.ipynb"),
    "02": load("02_prompt_engineering.ipynb"),
    "03": load("03_live_play.ipynb"),
    "04": load("04_adaptive_routing.ipynb"),
    "05": load("05_news_test.ipynb"),
}

def cell(src, src_idx):
    """Deep-copy a cell from a source notebook, force BRANCH='science'."""
    c = copy.deepcopy(NB[src]["cells"][src_idx])
    if c["cell_type"] == "code":
        c["source"] = [
            ("BRANCH = 'science'\n" if l.lstrip().startswith("BRANCH =") else l)
            for l in c["source"]
        ]
    return c

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

out = []

# ---------------------------------------------------------------- §0 Cover
out.append(md("""# Who Wants to Be a PoliMillionaire? — NLP 2025/26 Group Assignment

## Group members
- Jiaxin Yang — jiaxin.yang@mail.polimi.it
- Runjie (Simone) Dai — runjie.dai@mail.polimi.it

## Video
- Presentation video (≤5 min): **TODO_LINK**  <!-- ⚠️ 提交前唯一待填 -->

## Statement on coding assistants
We used a coding assistant (Claude Code) to help write, refactor and document parts of the
implementation. The system design, the experiments, and the prompt / RAG / routing strategies and
all analysis are our own; the assignment was **not** handed to an LLM to complete.

---

## What this notebook is
A self-explanatory consolidation of our whole pipeline and study, in narrative order:

| § | Section | Rubric question it answers |
|---|---|---|
| 1 | Setup | — (clone branch `science`, install, paths, game client) |
| 2 | Baseline single-model QA | 30s feasibility · per-topic strengths · overconfidence · failure modes |
| 3 | Prompt engineering | best prompt (zero / few / CoT) · prompt sensitivity |
| 4 | Adaptive prompt routing | adaptive prompting · *when does reasoning help vs hurt* · failure taxonomy |
| 5 | Tools + ensemble voting | calculator lift · self-consistency / ensemble reliability (agentic AI) |
| 6 | RAG (live web, raw content only) | RAG lift · RAG-vs-no-RAG ablation |
| 7 | Model comparison | model A/B · size vs quality |
| 8 | Live play — the real game | the leaderboard result (the actual test) |
| 9 | Conclusions | every investigation question → our finding |

> **How to read / reproduce.** Outputs are **preserved** from each experiment's own Colab run — the
> offline studies (§2–§7) are not meant to be re-run end-to-end in one pass (each loads the 7B model
> with its own config). To reproduce a single section: run **§1 Setup**, then that section's cells.
> The **live leaderboard run (§8)** is the only section we refresh for the final submission.
>
> Rules respected throughout: open-weight model run **locally** on Colab (`Qwen/Qwen2.5-7B-Instruct`,
> 4-bit), ≤30 s/question, and RAG uses **raw retrieved content only** (never generated answers).
> Code comments are in Yoda style (an assignment requirement)."""))

# ---------------------------------------------------------------- §1 Setup
out.append(md("""---
# §1 · Setup — clone (branch `science`), install deps, paths, game client

One canonical setup for the whole notebook. Pulls the code from GitHub (branch **`science`**), installs
the inference stack + headless Chromium (for the live-News body fetch), and puts `src` + the provided
`millionaire_client` on the path. Run this once."""))
out += [cell("05", 3), cell("05", 4), cell("05", 5)]

# ---------------------------------------------------------------- §2 Baseline
out.append(md("""---
# §2 · Baseline single-model QA  *(Phase 1 — the MVP)*

The first answering pipeline: load Qwen2.5-7B 4-bit once, wire the `QAPipeline`, and benchmark zero-shot
over the dev set. This section establishes **30 s feasibility**, **per-topic strengths**,
**overconfidence**, and the dominant **failure modes** — the questions the rubric asks first."""))
out += [cell("01", i) for i in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)]

# ---------------------------------------------------------------- §3 Prompt engineering
out.append(md("""---
# §3 · Prompt engineering  *(Phase 2)*

Three prompt strategies head-to-head on the dev set — **zero-shot**, **few-shot**, **chain-of-thought** —
plus `cot_v2`, a CoT variant driven by two real failure cases (option-matching slips on Maths). Which
prompt is truly best, and how sensitive is the model to phrasing?"""))
out += [cell("02", i) for i in (1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)]

# ---------------------------------------------------------------- §4 Adaptive routing
out.append(md("""---
# §4 · Adaptive prompt routing — a small-LLM reasoning study  *(the depth piece)*

The deeper question behind §3: for a small model, does routing each question to a prompt chosen for its
*reasoning shape* beat forcing one universal prompt — and **when does explicit reasoning help vs hurt?**
Four conditions (universal / generic-CoT / structured / adaptive) over a labelled 8-category reasoning
set, with a category×condition heatmap, a latency-vs-accuracy view, a routing-accuracy check, and a
failure taxonomy."""))
out += [cell("04", i) for i in range(3, 42) if i not in (4, 5, 6)]

# ---------------------------------------------------------------- §5 Tools + ensemble
out.append(md("""---
# §5 · Tools + ensemble voting  *(Phase 3 & 5 — agentic AI)*

The assignment encourages agentic techniques: tool calls + multi-model / self-consistency voting. Below
are our two core components and what we learned. **Calculator:** a safe-AST evaluator gated to maths-shaped
questions. **Ensemble:** a shared `majority_vote` primitive powering both self-consistency and multi-model
voting. (Honest finding: on live Maths, single-pass `cot_v2` beat self-consistency once latency was
accounted for — SC's 3 chains timed out near 41 s. See §4 / §8.)"""))
out += [cell("03", 13)]

# ---------------------------------------------------------------- §6 RAG
out.append(md("""---
# §6 · RAG — live web retrieval, raw content only  *(Phase 4)*

News (and the knowledge races) sit behind a routed retriever. **Rule-compliant**: only free sources that
return **raw, non-generated** content — Google News RSS + the Guardian Open Platform API + headless-Chromium
article bodies, with a query cascade, date re-ranking, and per-doc relevance filtering. Below: the core
implementation, a live stress-test (N rounds, each logged), and a **RAG-vs-no-RAG ablation** on the
retrieval races.

> The `TARGET` switch picks the race; outputs preserved here are from our stress-test run. Set
> `TARGET='news'` and re-run §6 to reproduce the News numbers."""))
out += [cell("05", i) for i in (12, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19)]

# ---------------------------------------------------------------- §7 Model comparison
out.append(md("""---
# §7 · Model comparison — base 7B vs a math-specialised model  *(model A/B · size vs quality)*

Does a math-specialised open model beat the general base 7B on the Maths race, and at what latency cost?
⚠️ This cell loads a **second** model (frees the first), so run it on its own."""))
out += [cell("05", i) for i in (22, 23)]

# ---------------------------------------------------------------- §8 Live play
out.append(md("""---
# §8 · Live play — the REAL test  *(the game API + full sweep)*

The actual game, not the dev set. Load the live config, wire the same pipeline, log in with the Colab
secret, then play. The **full sweep** plays one live game in each of the 6 competitions and reports the
reached level + every wrong question with its tool/retrieval/strategy trace. **This is the fresh final
run for the leaderboard.**"""))
out += [cell("03", i) for i in (7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22)]

# ---------------------------------------------------------------- §9 Conclusions
out.append(md("""---
# §9 · Conclusions — every investigation question → our finding

| Investigation question | Where | Our finding |
|---|---|---|
| **30 s feasibility** | §2 | Yes. Baseline median ~0.91 s, p95 ~1.0 s, 0 budget violations on a T4. Even the heaviest live turn (Maths `cot_v2`, ~20 s) stays under the 30 s wall. |
| **Per-topic strengths** | §2 | Strong on factual recall (History / Entertainment / News / Philosophy ≈ 100% at baseline); weak on **Maths (50%)** and **Science (75%)** — reasoning/calculation-bound, not recall-bound. |
| **Overconfidence** | §2 | Yes — poorly calibrated: confidently wrong answers reported `confidence = 1.0` (e.g. the moons question). |
| **Failure modes** | §2, §4 | Mostly genuine reasoning/knowledge gaps, not parser bugs. §4's taxonomy adds *overthinking* (recall Qs over-reasoned) vs *option-matching slips* (right derivation, wrong letter). |
| **Best prompt (zero / few / CoT)** | §3 | No single winner — few-shot helps factual recall, CoT helps reasoning; `cot_v2` fixes the Maths option-matching slip. This motivates §4. |
| **Prompt sensitivity / adaptive prompting** | §4 | **Adaptive routing beats every fixed prompt** — no one prompt is best across categories. Structured enumeration *cures* interval-counting/temporal Qs but *hurts* factual recall; checklists win on logic/multi-hop; direct answering wins on factual/commonsense. |
| **RAG lift** | §6 | Net positive, especially News (the model's prior is often wrong on post-cutoff facts). Raw-content-only sources; gated so it doesn't fire on pure-reasoning Qs. Ablation in §6. |
| **Calculator lift** | §5 | Safe-AST tool helps arithmetic-shaped Qs, but on live Maths `cot_v2` reasoning beat the tool, so it is gated off there to avoid clobbering correct chains. |
| **Ensemble reliability** | §4, §5 | `majority_vote` works, but self-consistency (n=3) blows the 30 s wall on Maths (~41 s) and all chains shared the same slip, so it could not self-correct. Single-pass `cot_v2` is the live choice. |
| **Model A/B · size vs quality** | §7 | Base 7B vs a math-specialised open model on the Maths race (accuracy vs latency trade-off). |
| **Live leaderboard result** | §8 | _Fill in after the fresh final sweep:_ best level per competition. |

### Honest limitations
- The ~7B model has a knowledge ceiling (~55–65%) on the hardest knowledge rungs; RAG narrows but does not close it.
- Heavy retrieval bursts can trigger source rate-limits; we degrade gracefully (empty context, never a crashed turn).
- Self-consistency is unused live purely for latency; it remains valuable offline (see §4).

_(Full write-ups: `experiments/adaptive_routing_findings.md` and the per-phase notes in `memory/`.)_"""))

# ---------------------------------------------------------------- assemble
final = {
    "cells": out,
    "metadata": NB["05"].get("metadata", {}),
    "nbformat": 4,
    "nbformat_minor": NB["05"].get("nbformat_minor", 5),
}
dst = os.path.join(HERE, "99_final_submission.ipynb")
with open(dst, "w") as f:
    json.dump(final, f, indent=1, ensure_ascii=False)

print("wrote", dst)
print("total cells:", len(out),
      "| md:", sum(c["cell_type"] == "markdown" for c in out),
      "| code:", sum(c["cell_type"] == "code" for c in out))
