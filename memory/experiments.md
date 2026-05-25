# Experiments Log

> Append one block per benchmark run. Pull numbers from `experiments/runs/<run_id>/`.
> Never edit past results; add a new entry if a run is repeated.

Required fields per run (the rubric): **model · prompt · latency · accuracy · retrieval usage · tool usage**.

## Template

```
### <run_id> — <one-line purpose>
- Date / commit:
- Model:                 (name, quantization)
- Prompt strategy:
- Retrieval:             (off | corpus/API, top_k)
- Tools:                 (none | calculator)
- Ensemble:              (no | models, vote rule)
- Dataset:               (dev set name, N questions)
- Hardware:              (T4 / L4)
- Accuracy:              overall = __%   (by topic / level: …)
- Latency:               median __s, p95 __s, max __s   (budget 30s; violations: __)
- Overconfidence:        (mean confidence vs accuracy)
- Notes / failure modes:
```

---

## Results

### baseline — Phase 1 first baseline (zero-shot, single model)
- Date / commit:         2026-05-25 · post "git-clone workflow" commit
- Model:                 Qwen/Qwen2.5-7B-Instruct, 4-bit nf4 (bitsandbytes)
- Prompt strategy:       zero_shot_v1 (single letter)
- Retrieval:             off
- Tools:                 none
- Ensemble:              no
- Dataset:               dev_questions.jsonl, N=23 (6 topics)
- Hardware:              Colab GPU (T4/L4 — see run meta.json)
- Accuracy:              overall = **87.0%** (20/23)
  - by topic: Ancient History 1.00(4) · Entertainment 1.00(4) · News 1.00(4) · Philosophy 1.00(3) ·
    **Science 0.75(4)** · **Maths 0.50(4)** ← the two weak spots
  - by level: clean until rung 5–6 (lvl5=0.80/5, lvl6=0.67/3); small-N so noisy
- Latency:               median **0.91s**, p95 1.01s, max 1.05s (budget 30s; violations: 0)
- Overconfidence:        **CONFIRMED — all 3 misses at confidence = 1.0** (confidently wrong). Under
  zero_shot the parser assigns 1.0 to every clean single letter, so confidence carries NO calibration
  signal → it is USELESS as a routing trigger. Design implication: gate RAG/tools on **content**
  (classifier `needs_calculator`/`needs_retrieval`), never on self-reported confidence. Validates D-006 routing.
- Notes / failure modes (3 misses INSPECTED + confirmed):
  - **Latency is a non-issue** — ~1s/question leaves HUGE 30s headroom for CoT, RAG, and 3-model ensembles.
    Answers the "30s feasibility" rubric question decisively for a single 7B.
  - **dev-001 (Science): "planet with most moons" → pred A (Jupiter), gold B (Saturn).** CONFIRMED
    knowledge-cutoff trap (Saturn overtook Jupiter in 2023). The model knows the pre-2023 fact. → Phase-4 **RAG**.
    (Caveat: such "facts" drift — a real risk for the live Science competition.)
  - **dev-005 (Maths): 17×13 → pred D (201), gold B (221).** Plain arithmetic error (picked the near-miss distractor).
  - **dev-008 (Maths): 2^10 → pred A (512=2^9), gold C (1024).** Off-by-one-power error. → both → Phase-3 **calculator**.
  - Open question for Phase 2: does **CoT** ("think step by step") recover the arithmetic on its own, or is a
    deterministic **calculator tool** required? A clean CoT-vs-tool comparison this sets up (good rubric depth).

### prompt_eng — Phase 2: three prompt strategies head-to-head
- Date / commit:         2026-05-25 · branch `prompt`, commit d830d68 (notebooks/02_prompt_engineering.ipynb)
- Model:                 Qwen/Qwen2.5-7B-Instruct, 4-bit nf4 (bitsandbytes), loaded ONCE
- Prompt strategy:       zero_shot_v1 · few_shot_v1 (3 exemplars) · cot_v1 ("think briefly → Answer: X")
- Retrieval:             off
- Tools:                 none
- Ensemble:              no
- Dataset:               dev_questions.jsonl, N=23 (6 topics)
- Hardware:              Colab GPU (same session as baseline)
- Generation:            pipeline default max_new_tokens=256 (NOT overridden) → CoT not truncated
- Accuracy (overall):    zero_shot **20/23 (87.0%)** · few_shot **20/23 (87.0%)** · cot **14/23 (60.9%)**
- Accuracy (Maths, n=4): zero_shot **2/4 (0.50)** · few_shot **3/4 (0.75)** · cot **3/4 (0.75)**
  - topic×strategy (cot / few / zero):
    Ancient 0.75/1.00/1.00 · Entertainment 0.50/1.00/1.00 · Maths 0.75/0.75/0.50 ·
    News 0.50/0.75/1.00 · Philosophy 0.67/1.00/1.00 · Science 0.50/0.75/0.75
- Latency:               mean — zero 0.87s · few 1.40s · cot 3.86s (all ≪ 30s budget; 0 violations)
- Tokens out (mean):     zero 2.0 · few 2.0 · cot 35.9
- ⚠️ THE cot ACCURACY NUMBERS ABOVE ARE A PARSER ARTIFACT — see CORRECTION below. zero/few unaffected.
- BEST PROMPT (corrected): cot's 60.9% is bogus; per-question raw_output shows cot's reasoning is mostly
  CORRECT. zero_shot/few_shot tie at 87% (real); cot likely ≥91% once re-parsed → probably the BEST.
- READ-OFF (does CoT fix Maths 0.50?): under the bug it read "no". CORRECTED: cot is ABSENT from the maths
  misses for dev-005 (17×13) and dev-008 (2^10) — i.e. cot SOLVED both arithmetic Qs zero/few got wrong.
  So CoT very likely DOES lift Maths toward ~1.00. The calculator (Phase 3) value-add must be re-judged
  against a correctly-parsed CoT baseline, NOT against zero_shot. PENDING re-parse to confirm.

  ### CORRECTION (same session, 2026-05-25) — CoT collapse is a PARSER BUG, not overthinking
  Per-question raw_output (user-supplied) is decisive. EVERY cot prediction came back "A" @ conf 0.5, yet
  the model's own "Answer: X" line was CORRECT in 7 of the 9 shown cot misses (dev-004 D, dev-007 D,
  dev-011 B, dev-013 D, dev-014 C, dev-017 C, dev-021 D); only dev-001 (A, 2023 cutoff) & dev-023 (D vs B)
  are genuine model errors. ROOT CAUSE in `pipeline.py::parse_answer`:
    1. Pattern-1 `_EXPLICIT` regex is `answer\s+(?:is\s*:?\s*|:\s*)([A-Da-d])` — the mandatory `\s+` after
       "answer" means it CANNOT match the model's actual format "Answer:" (colon, no space). Never fires.
    2. Falls through to pattern-7 `_ISOLATED`, which returns the FIRST standalone A–D letter — almost
       always the English article **"a"** ("a vacuum", "a rectangle", "a dialogue", …) → "A" @ conf 0.5.
  LATENT RISK: zero/few-shot only dodge this because they emit a BARE letter (pattern-6). Any prose answer
  (CoT, RAG context echoes, tool re-answers) will hit the same bug → fix BEFORE Phase 3/4.
  FIX (not yet applied — user asked analyse-only): relax `_EXPLICIT` so `\s+` is optional / "Answer:" matches;
  and stop pattern-7 from grabbing lone "a"/"i" (prefer the LAST isolated letter, or anchor on the final
  "Answer:" line). Then RE-PARSE the saved records.jsonl for all 3 runs (NO model re-run needed —
  raw_output is persisted) to get the true accuracies.

## Leaderboard runs (live game)
_(track each official run here: date, config used, final prize/level reached, observations)_

### live_comp0..5 — first ALL-competitions live sweep (few_shot + calculator)
- Date / commit:         2026-05-25 · branch `phase-3`
- Model:                 Qwen/Qwen2.5-7B-Instruct, 4-bit nf4
- Prompt strategy:       few_shot_v1
- Retrieval:             off
- Tools:                 calculator (`default_tools`; Phase 3 — auto-fires on Maths only)
- Mode:                  LIVE (text), one game per competition (0–5), via `run_all_competitions` sweep
- Accuracy:              **overall 86.7% (39/45)** — tracks the offline 87% closely
  - per competition: Philosophy **1.00 (15/15)** · Entertainment 0.88 (15/17) · Ancient History 0.80 (4/5) ·
    Maths 0.75 (3/4) · Science 0.50 (1/2) · News 0.50 (1/2)   [Science/News tiny-N]
  - reached_level = 0 for ALL; "answered" count varies a lot (Ent 17 vs Sci/News 2). GAME-MECHANIC
    UNRESOLVED: a wrong answer seems to END some games (Sci/News/Maths/Hist each stopped right after their
    lone wrong) yet Entertainment continued past 2 wrongs to 17. Investigate the level/termination contract.
- Latency:               all turns ~0.8–1.8s (≪ 30s; 0 violations) — the sweep reconfirms latency is a non-issue
- Notes / failure modes (6 wrong, saved to `experiments/wrong_questions.jsonl`) — THREE buckets:
  - **Pure recall** → Phase-4 RAG: qid 140 (Johnny Cash crossover, arguably ambiguous A vs B); 1232 (Roman
    tunica recta); 772 (M3GAN — picked the generic "self-preservation" AI trope over the film-specific
    "emotional attachment").
  - **Reasoning/set-up errors the model SHOULD get** → CoT candidate (parser now fixed, so cot parses):
    - 4902 (Science method): to repeat a freezing experiment you need "volume of water" (C); model picked
      "temperature of the ice" (A).
    - 6854 (Maths): paint ∝ surface area, gold 13/2; model picked 13 (=√169, dropped the proportionality
      constant). NOTE: the calculator CANNOT save a wrong set-up — only CoT-style reasoning can.
  - **Beyond knowledge cutoff** → RAG-only: 11239 (News, assassination on 2026-05-17; model cutoff Jan-2026
    cannot know it). Confirms News = the canonical RAG competition.
- Pipeline status: end-to-end LIVE works — few_shot + calculator + parser fix + options logging + 6-comp sweep.
- NEXT: (B) settle strategy (re-parse / re-run for true cot numbers; cot may lift 4902/6854) → then (A) RAG (Phase 4).
