# Cache Residency Probe

Date: 2026-06-06

## Goal

Test the missing cache question:

```text
If the agent switches from full history to a forgotten active prompt, does the
runtime keep the full-history prompt reusable in cache?
```

This is not the same as modular KV-cache recomposition. It is a smaller but
important requirement: whole-prompt cache residency across active-context
switches.

## Setup

Runtime:

```text
llama.cpp
Qwen3-4B-Q4_K_M.gguf
context size: 8192
prompt cache: enabled
threads: 10
```

Probe sequence:

```text
full_history_first
forgotten_active_first
full_history_return
forgotten_active_return
```

The full-history prompt includes the noisy QR, frontend, billing, and analytics
files. The forgotten active prompt uses `forget_critical_context_v1` and omits
those files from the active prompt.

## Result

```text
full_history_first:
  runtime prompt tokens: 3600
  cached prompt tokens: 0
  cache hit ratio: 0.0000
  latency: 228636.41 ms

forgotten_active_first:
  runtime prompt tokens: 436
  cached prompt tokens: 0
  cache hit ratio: 0.0000
  latency: 23402.08 ms

full_history_return:
  runtime prompt tokens: 3600
  cached prompt tokens: 3599
  cache hit ratio: 0.9997
  latency: 3700.47 ms

forgotten_active_return:
  runtime prompt tokens: 436
  cached prompt tokens: 435
  cache hit ratio: 0.9977
  latency: 2274.19 ms
```

## Conclusion

This proves a useful near-term claim for llama.cpp:

> The runtime can keep both the full-history prompt and the forgotten active
> prompt resident/reusable across switches.

So the system can work like this:

```text
full external trace remains available
full-history prompt can remain cached by the runtime
short forgotten prompt can be used for the active task
returning to the full prompt can hit the runtime prompt cache
```

This makes the active-context forgetting layer more useful than plain prompt
rewriting, because switching away from noise did not destroy the cached full
prompt in this probe.

## Limitation

This does **not** prove the original hard idea yet:

```text
arbitrary KV blocks can be removed, kept, and recombined into new prompt shapes
without recomputation
```

For a normal causal transformer, cached KV for later tokens depends on earlier
tokens and positions. If a prompt was:

```text
A + noise + B
```

then the cached KV for `B` is not generally valid for:

```text
A + B
```

The current result is whole-prompt cache residency, not KV surgery.

## Next Step

The next credible runtime experiment is a branch/prefix probe:

```text
A
A + noise
A + task
A + noise again
A + task again
```

That would measure whether the runtime can keep multiple branches under a
shared prefix hot at the same time.
