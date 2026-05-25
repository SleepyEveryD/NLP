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

## Leaderboard runs (live game)
_(track each official run here: date, config used, final prize/level reached, observations)_
