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
_(none yet)_
