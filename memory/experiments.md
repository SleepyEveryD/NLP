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

_(none yet — Phase 1 will produce the first baseline)_

## Leaderboard runs (live game)
_(track each official run here: date, config used, final prize/level reached, observations)_
