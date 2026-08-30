# llama-server Multi-Slot Baseline (Milestone 6 Re-Test)

Date: 2026-07-25
Model: Qwen3-4B-Q4_K_M (CPU, 10 threads, ctx 16384, batch 2048).
Script: `scripts/run_llama_server_switch_benchmark.py`.
Raw data: `benchmark-results/llama-server-switch-benchmark.json`.
Compared against: `2026-07-23-branch-switch-benchmark.md` (same fixture, same
schedule, same segment texts, 5 repeats each).

## Why this test

The native branch-switch benchmark compares resident KV branches against a
**single** prefix-cache slot. Its own docstring says so: "a single sequence
emulates a one-slot prefix cache (like a single llama.cpp server slot)."

But stock llama.cpp is not limited to one slot. `llama-server -np N` keeps N
independent slots, each with its own KV and its own prefix matching. That is
the natural deployment shape for "one resident conversation per subtask", and
it needs no custom KV code. If it matches `branch_cache`, the switch-latency
result of Milestone 6 measures a harness limitation rather than a property of
semantic branching.

## Setup

`llama-server -m Qwen3-4B-Q4_K_M.gguf -c 16384 -t 10 -b 2048 -np 4 --slots`.
The same `auth -> qr -> auth -> qr` schedule and the same segment texts are
replayed over `/completion` with `cache_prompt: true`, `n_predict: 1`, and
`id_slot` pinned per branch. Per-switch cost is `timings.prompt_ms`. Slots are
erased between repeats. Setup (root + branch prefill) is measured separately,
mirroring the native plan's setup stage.

## Result

Medians of 5 repeats, per switch:

```text
                          native runner        llama-server
branch_cache / slots:        1110 ms              1201 ms
prefix_slot  / single slot:  7002 ms              6760 ms
scratch:                    40597 ms                  n/a
```

Setup (one-time), medians of 5 repeats:

```text
native branch_cache:  43949 ms   (root evaluated once, then seq_cp per branch)
llama-server slots:   83993 ms   (root prefilled once per slot, no sharing)
```

Per-switch prefill mechanics are identical in kind:

```text
llama-server slots:       prompt_n 26 / switch,      cache_n 1511-1547
llama-server single slot: prompt_n 196-232 / switch, cache_n 1341 (root only)
```

## Interpretation

- **Switch latency: no contribution from semantic branching.** 1201 ms on
  stock `llama-server -np 2` versus 1110 ms on the native runner — 1.08x, well
  inside the repeat-to-repeat spread (1021-1273 ms across the 5 runs). Both
  reuse the resident branch and prefill only the 26 new turn tokens.
- **The single-slot baseline reproduces to within 4%** (6760 vs 7002 ms),
  which validates that the replay is equivalent to the native `prefix_slot`
  condition and that the gap above is a real comparison, not a mismatch.
- **The headline "6.3x faster than a prefix cache" therefore does not hold
  against the state of practice.** It holds against a one-slot cache, which is
  a configuration choice, not a constraint. `scratch` (36.6x) is weaker still:
  no real deployment re-prefills a conversation from scratch on every turn.
- **Branching does win setup: ~1.9x** (43.9 s vs 84.0 s). The root is
  evaluated once and copied per branch, where the server prefills it once per
  slot. This is a genuine, if one-time, advantage.

## Follow-up finding: seq_cp is copying, not sharing

`llama_kv_cache::seq_cp` is metadata-only — `cells.seq_add(i, seq_id_dst)`,
no data movement — **only when both sequences live in the same stream**.
Cross-stream copies move the actual buffer.

`native/semantic_kv_runner.cpp:581` builds its context from
`llama_context_default_params()` and never sets `kv_unified`, which defaults to
`false` (`llama-context.cpp:3375`). With `kv_unified = false`, each sequence
gets its own stream and `n_ctx_seq = n_ctx / n_seq_max`. So every branch today
holds a **physical copy** of the root KV.

Two consequences:

- The reported footprint (~2.9x a single slot) is real, not a measurement
  artifact — but it is a cost with no sharing benefit, and on this axis
  branching is currently no better than N server slots (which also do not
  share: the server ran at `n_ctx_seq = 4096` per slot).
- The one architectural advantage branching could hold over server slots —
  sharing the root's cells across branches — is **available but unused**.
  Setting `kv_unified = true` should make branch creation metadata-only
  (near-zero setup instead of 43.9 s) and make the root cost KV memory once
  instead of once per branch.

That is the untested experiment this re-test surfaces, and it is the only
remaining path by which Milestone 6 could claim something stock llama.cpp does
not already do.

## Caveats

Single machine, CPU-only, one model, 5 repeats. The llama-server path adds HTTP
and re-tokenization overhead the native runner does not pay, so the 1.08x gap
is an upper bound on any native advantage, not a measurement of one. The
`kv_unified = true` claim above is derived from reading llama.cpp, not measured.
