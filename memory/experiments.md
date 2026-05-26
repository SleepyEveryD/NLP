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

### live_comp0..5 (run #2) — bigger sweep, Wikipedia RAG actually ON (the 26-wrong dump)
- Date / commit:         2026-05-26 · branch `4-rag` (commits 6a3a86c "Questions answered" + 6b05d0e + bf7dee1)
- Config:                few_shot_v1 + calculator + **RAG ON = `WikipediaRetriever`** (notebook printed
  `RAG: ON (wikipedia)`; Maths run shows `retrieval_used Counter({False:5, True:4})`). ⚠️ The section-8
  markdown said "RAG off" — STALE/WRONG; Wikipedia WAS on. (Now fixed in the notebook.)
- Accuracy:              **overall 83.4% (131/157)** — 26 wrong (the dump the user analysed).
  - per competition (answered/correct): Entertainment 15/9 (0.60) · Ancient 30/26 (0.87) · Science 31/28
    (0.90) · Maths 9/4 (0.44) · Philosophy 65/62 (0.95) · **News 7/2 (0.29)**.
- reached_level (the REAL leaderboard metric, from `my_reached_levels`, commit 6b05d0e):
  Entertainment **15** · Philosophy **15** · Science **13** · Ancient **12** · **Maths 3** · **News 1**.
  → The two bottlenecks, unambiguous they are: **Maths** and **News**.
- KEY READ-OFF: News at **2/7 EVEN WITH Wikipedia RAG ON** — Wikipedia cannot hold the post-cutoff 2026
  events the News questions cite ("article published on 2026-05-15", a Guardian byline). The fix is a LIVE
  **web** source for News, not Wikipedia. Maths misses are mostly bad SET-UP (S_5 order, Z_5 char-p
  derivative, ODE, area=perimeter) → CoT, not the calculator (which a wrong set-up cannot save).
- ACTION TAKEN (2026-05-26, this session): Phase-4 retriever REWRITTEN to a routed `Retriever` —
  News → `WebSearchRetriever` (DuckDuckGo, RAW snippets) with Wikipedia fallback; else → `FaissRetriever`
  (Simple-Wikipedia) with Wikipedia fallback. `needs_retrieval` now fires on the News recency signature too.
  `live.yaml` source="routed". PENDING Colab: build the FAISS index + re-run the sweep → measure News lift.

### live_comp0..5 (run #3) — first ROUTED RAG sweep (web+FAISS), the News-lift check
- Date / commit:         2026-05-26 · branch `4-rag` (post-`91461f3` "WebSearchRetriever, WikipediaRetriever, FaissRetriever")
- Config:                few_shot_v1 + calculator + **RAG `source="routed"`** (News → `WebSearchRetriever`
  DuckDuckGo, else → `FaissRetriever` / Wikipedia fallback). The retriever that run #2's ACTION TAKEN rewrote — this is its FIRST live measurement.
- Accuracy:              **overall 80.0% (28/35)** — 7 wrong. ⚠️ **small-N**, much shorter sweep than run #2's 157
  (Philosophy & News only 2 graded each → their per-comp accuracy is noise, do not read it).
  - per competition (answered/correct/acc): Entertainment 12/10 (0.83) · Ancient 8/7 (0.88) · Science 7/6 (0.86) ·
    Maths 4/3 (0.75) · Philosophy 2/1 (0.50) · News 2/1 (0.50).
- reached_level (REAL leaderboard metric, `my_reached_levels` — CUMULATIVE best, not this-session-only):
  Entertainment **15** · Philosophy **15** · Science **13** · Ancient **12** · **Maths 3** · **News 3**.
  → **HEADLINE: News 1 → 3** (+2, the only mover) — the routed **web** path lifted News, exactly the run-#2
  hypothesis. Maths still **stuck at 3** — the other bottleneck unmoved.
- KEY READ-OFF (Maths): qid 6786 (MVT count) shows `tool=calculator correct=False` — the calculator FIRED and
  still missed. Reconfirms (run #1, #2): **a wrong set-up the calculator cannot save** → Maths needs **CoT**, not the tool.
  (Maths-comp trace: `tool_used Counter({None:2,'calculator':2})`, `retrieval_used Counter({False:3,True:1})`.)
- 7 wrong (qids 183, 566, 1121, 4129, 6786, 8071, 11194) — buckets:
  - **CoT-fixable reasoning/terminology**: 4129 (Science "constructive force of glacier" = DEPOSITION/moraine D;
    we picked the EROSIONAL "valleys carved" A) · 6786 (MVT, above).
  - **Recall / RAG candidates**: 183 (Welles' Nuremberg "Cathedral of Light" = lighting C, not back-projection) ·
    8071 (Zeno USED actual infinity A; potential infinity is Aristotle's REPLY).
  - **Suspect / source-dependent gold — do NOT tune to these**: 566 (Pulp Fiction "fabricated biblical passage" →
    Jules IS the canonical answer, yet marked WRONG ⇒ likely a bad gold) · 1121 (Egyptian-blue decline → "recipe lost"
    is the common reason = the option we PICKED; gold favoured another ⇒ RAG would not flip us).
  - **News / web-path target**: 11194 (Washington Sq Park each May = NYU graduations B; we picked political rally D).
- ⚠️ DIAGNOSTIC GAP (now closed): this run's wrong-dump carried NO per-question `tool_used`/`retrieval_used`, and the
  detail cell was Maths-only — so whether the News web path FIRED for 11194 (and whether snippets landed) is UNKNOWN.
  Three distinct failure modes are indistinguishable: gate didn't fire · DDG 429/blocked on Colab IP → empty · snippets
  landed but missed. ACTION TAKEN (this session): notebook 03 dump cells now print + save `tool_used` / `retrieval_used` /
  `retrieved_doc_ids` (count) for EVERY wrong question + a per-competition RAG/tool usage table + per-question detail
  across ALL comps (was Maths-only). `retrieval_used=True` + `docs_landed=0` = "fired but empty" (the News bug to watch).
- NEXT: re-run the sweep with the instrumented dump → read the News trace for 11194; route Maths → cot_v1.

### live_comp0..5 (run #4) — instrumented dump READ: the News killer is a calculator-clobber bug
- Date / commit:         2026-05-26 · branch `4-rag` (post-`f5a6804`; the 26/33 instrumented dump the user analysed)
- Config:                few_shot_v1 + calculator + **RAG `source="routed"`** (same as run #3), now with the
  per-question `tool_used`/`retrieval_used`/`docs_landed`埋点 that run #3 added.
- Accuracy:              **overall 78.8% (26/33)** — 7 wrong. ⚠️ small-N (News/Maths only 1 graded each — noise).
  - per comp (answered/correct): Entertainment 3/1 · Ancient 11/10 · Science 10/9 · Maths 1/0 · Philosophy 7/6 · News 1/0.
  - reached_level (cumulative best): Entertainment 15 · Philosophy 15 · Science 13 · Ancient 12 · Maths 3 · News 3.
- ROOT CAUSE FOUND (the run-#3 diagnostic gap, now CLOSED): News qid=11425 trace was
  `tool=calculator · retrieval_used=True · docs_landed=3 ['ddg:0','ddg:1','ddg:2']` — i.e. NOT "gate didn't
  fire" and NOT "DDG empty". The web evidence LANDED, then was THROWN AWAY. Two-part bug, confirmed + reproduced:
    1. `classify/classifier.py::needs_calculator` returned **True** on a News question, because the ISO date
       **`2026-05-18`** trips it twice: its hyphens match the minus operator `[+\-*/×÷]` AND its three
       digit-groups `['2026','05','18']` satisfy the "operator + ≥2 numbers" rule (lines ~344-348).
    2. `agent/pipeline.py::_run_calculator_tool` builds the re-answer prompt from the **bare MCQ only** —
       NOT the retrieved docs. So once the calculator wrongly fires, the model re-answers BLIND to the snippets.
  - EVERY News question carries "article published on YYYY-MM-DD" → EVERY News question mis-fired the calculator
    and discarded its web evidence. THE reason News stayed bottom even with the routed web path working.
- FIX APPLIED (this session, branch `4-rag`):
    1. `classifier.py`: new `_DATE_LIKE_RE`; `needs_calculator` now strips ISO/slash dates BEFORE the
       operator/digit checks. Verified: qid=11425 → needs_calculator=False; 17×13 / 2^10 / "sum of" / "20%"
       still → True; bare year "1492" → False. (7/7 inline cases pass; no test suite in repo.)
    2. `pipeline.py`: tool stage now gated on `not retrieval_used` — a context-free re-answer must never
       override an evidence-grounded one (defensive 2nd layer; the calc path carries no docs today anyway).
- NEXT: re-run the sweep on Colab → confirm News qid=11425 now keeps the web answer (expect News lift);
  route Maths → cot_v1 (still the other unmoved bottleneck — 6657 is a concept Q the calculator can't save).

### live_comp0..5 (run #5) — News calculator-clobber FIX VERIFIED (the 37/43 dump)
- Date / commit:         2026-05-26 · branch `4-rag` (post-`7b05d0f`; the News/calc fix + pandas-pin + hard-sync notebook)
- Config:                few_shot_v1 + calculator + RAG `source="routed"`. Same as run #4, now WITH the fix:
  `needs_calculator` strips ISO/slash dates first; pipeline skips the tool stage when retrieval already fired.
- Accuracy:              **overall 86.0% (37/43)** — UP from run #4's 78.8% (26/33). 6 wrong.
  - per comp (answered/correct): Entertainment 2/0 · Ancient 15/15 · Science 5/4 · Maths 3/2 · Philosophy 15/14 · News 3/2.
- reached_level (cumulative leaderboard best): Entertainment 15 · **Ancient 15 (12→15, MAXED; score 128k→1024000)** ·
  Science 13 · Maths 3 · Philosophy 15 · News 3.
- ✅ FIX CONFIRMED: all 3 News questions now `tool=None` (run #4 had `tool=calculator` on the dated News q). The
  ISO date no longer mis-fires the calculator; the retrieved web/wiki evidence is PRESERVED in the prompt. News 2/3.
- NEW FINDING #1 — retrieval HURTS Maths: qid=6822 (group theory: factor-group / normal-subgroup transitivity)
  fired retrieval (the capitalised-word heuristic in `needs_retrieval`), the query distilled to "Statement", and it
  pulled GARBAGE Wikipedia pages `['Statement','Financial statement','The Statement']`. Wrong (picked D; gold A=False,False).
  → Maths should NOT retrieve (no FAISS index on Colab → degrades to Wikipedia with a junk query) AND needs CoT, not the tool.
  (qid=6649 `tool=calculator retr=False correct=True` shows the date-strip fix did NOT break legit calc on real maths.)
- NEW FINDING #2 — the News miss (11882) is retrieval QUALITY, not the calc bug: DDG came back empty → fell back to
  Wikipedia → generic Orbán pages (not the specific 2026-05-14 article), so the answer (C Peter Magyar) wasn't in context.
  The web path returning empty is the residual News risk (comp 1 also had 3× `retr=True docs=0` — harmless there).
- Other wrong (recall/concept, no retrieval fired): 800 (Jay-Z 2023 title) · 128 (Isla Nublar = Hawaii, Jurassic Park) ·
  4610 (a mutation making fur CONTRAST with the environment HARMS the mice — concept, CoT) · 9796 ('bro' neologism term — obscure/maybe bad gold).
- NEXT: route comp 3 Maths → `cot_v1` AND suppress retrieval for Maths — keyed on `competition_id` (the reliable live
  signal; topic is unset in live play). Optionally improve the News web path's empty-result rate (DDG fallback quality).

### live_comp0..5 (run #6) — Maths→cot routing INCONCLUSIVE; a logger-append CONTAMINATION bug found
- Date / commit:         2026-05-26 · branch `4-rag` (post-`86c137e`, the Maths-routing commit)
- Config:                few_shot_v1 + calculator + routed RAG; comp 3 → `pipeline_maths` (cot_v1 + no retrieval).
- ⚠️ RESULT UNRELIABLE — a logging bug surfaced. `ExperimentLogger` opened `records.jsonl` in APPEND ("a") mode,
  and the run dir `live_comp{id}` is a FIXED name (gitignored → force-sync never clears it). So every re-run of the
  sweep PILED its records onto the previous run's; `load_runs` read the UNION. Tells in the run-#6 dump: comp 0 = 22
  answered with 5 wrong (a single game can't), qids 345/156 appear TWICE; comp 1 = 36 with qid 1157 twice. The whole
  run-#6 dump is several sweeps merged → do NOT read its accuracy/counts.
- Maths routing verdict: INCONCLUSIVE. comp 3 showed `6822 retr=True docs=3`, which CANNOT come from `pipeline_maths`
  (retriever=None skips the retrieve stage) — almost certainly a leftover run-#5 record (the default pipeline). Whether
  cot_v1 + no-retrieval helps Maths is UNKNOWN until a clean run. Maths lb still 3; News lb still 3 (calc fix holds:
  all News `tool=None`, but 2 misses 11882/12105 are the web path returning generic/empty — retrieval QUALITY).
- FIX APPLIED (this session):
    1. `logger.py`: open mode "a" → "w" (truncate-per-run). `live_comp{id}` now reflects the LATEST run ONLY.
    2. notebook detail-dump: now prints `prompt_strategy` per question + a per-comp Counter → the next run VERIFIES
       routing at a glance (comp 3 should be `strat=cot_v1` + `retr=False`; every other comp `few_shot_v1`).
- NEXT: CLEAN re-run (the "w" fix truncates stale dirs on open; or `rm -rf experiments/runs/live_comp*` first to be sure).
  Then read comp 3: `strat=cot_v1` + `retr=False` confirms routing; does Maths finally climb past lb 3?
