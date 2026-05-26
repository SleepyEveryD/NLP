# Errors & Debugging Insights

> Record failures, root causes, and fixes here so we don't re-debug the same thing.
> Especially Colab/quantization gotchas and game-API quirks.

Format: `### Symptom` · Context · Root cause · Fix · Date.

---

## Known risk areas to watch (pre-emptive notes)
- **Colab OOM on load:** 7B in fp16 won't fit T4 → must use 4-bit. If OOM persists, lower
  `max_new_tokens`/batch, or restart runtime to clear fragmented VRAM.
- **bitsandbytes/CUDA mismatch on Colab:** pin versions in `requirements.txt`; if load fails,
  check the installed CUDA vs the wheel.
- **Latency spikes from cold start:** always `warmup()` before timing; first call is not representative.
- **Answer parsing:** models add chatter around the letter → `parse_answer` must be robust
  (regex for a standalone A–D, ignore "Option", handle lowercase).
- **Game rate limiting:** rapid requests may get throttled → enforce `request_delay_s`, never loop hot.
- **Session ephemerality:** Colab disconnects lose in-memory state → logs flushed per record; consider Drive mount.

## Environment notes
- **No local Python** on the Windows dev box (verified 2026-05-25: `python`/`py` not found).
  All execution + verification happens on **Colab**. Don't expect to run `src/` locally; the
  Phase 0 notebook is the first place modules get imported/smoke-tested.

## Resolved issues

### Scoreboard `reached_level` always 0 (notebook 03)
- Context: the section-6 ("Highest level reached: 0") and section-8 SCORES-BY-COMPETITION table both showed
  `reached_level = 0` for every competition, while the standalone leaderboard cell printed the real levels
  (Entertainment 15, Maths 3, ...). Confusing — looked like we never climbed.
- Root cause: both cells computed `reached = int(df['level'].dropna().max())`. But `EvalRecord.level` is
  `question.level` — the per-turn rung the **server sends as 0** (`Question.from_dict` → `level=data.get("level", 0)`).
  The REAL climb lives in `EvalRecord.reached_level`/`current_level`, lifted from `AnswerResult.reachedLevel`/
  `currentLevel` (runner.py:221) — a DIFFERENT column the scoreboard never read.
- Fix (2026-05-26): both cells now use a `_run_reached(df)` helper = `max` of `reached_level` (fallback
  `current_level`, then `level`). The section-8 table also gained `lb_level`/`lb_score` columns pulled straight
  from the leaderboard API (`leaderboard.find_player`, crash-safe) — the authoritative all-time scored value,
  so even if per-turn telemetry is None the real number still shows. `run_reached` = this-sweep climb;
  `lb_level` = all-time best.
- Date: 2026-05-26.
