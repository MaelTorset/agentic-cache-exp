# Branch-Switch Benchmark: Resident KV Branches vs Prefix Cache vs Re-Prefill

Date: 2026-07-23
Model: Qwen3-4B-Q4_K_M (CPU, 10 threads, ctx 16384, 8 sequences)
Workload: fixture repo trace, root padded to ~1341 tokens, branches auth/qr
(~108 tokens each), switch schedule `auth -> qr -> auth -> qr`, ~30-token turn
per switch. 5 repeats (fresh process each), medians reported.
Runner: `scripts/run_branch_switch_benchmark.py` -> `semantic-kv-runner`.
Raw data: `benchmark-results/branch-switch-benchmark.json`.

## Headline result

Per-switch cost (median across 5 runs of the mean over 4 switches):

```text
branch_cache (resident KV branches):   1110 ms / switch
prefix_slot  (one-slot prefix cache):  7002 ms / switch   (6.3x slower)
scratch      (full re-prefill):       40597 ms / switch  (36.6x slower)
```

Setup cost (paid once):

```text
branch_cache: 43949 ms  (root eval + 2 seq copies + 2 branch evals)
prefix_slot:  33480 ms  (root eval)
scratch:          0 ms
```

Break-even: branch_cache's extra ~10.5 s of setup vs prefix_slot is repaid in
under 2 switches; vs scratch the whole setup is repaid in ~1.1 switches.

## Exactness

In the same plan, the final auth-branch state under branch_cache and the
scratch re-prefill of identical tokens produced:

```text
mean_abs_diff: 0, cosine_similarity: 1, top_k_overlap: 5/5
generation_text_match: true in 5/5 runs (24 greedy tokens)
```

This required a runner fix: `generate` now takes its first greedy token from
the sequence's captured logit snapshot instead of the context-global logits of
the last decode (which can belong to another sequence's generation).

## Memory cost

Resident tokens after the run (split KV cache, fp16):

```text
branch_cache: seqs 0+1+2 = 4504 tokens  (~633 MB at 147 KB/token for Qwen3-4B)
prefix_slot:  seq 3     = 1575 tokens  (~221 MB)
scratch:      transient (4 x ~1560 tokens during the run)
```

Branch residency buys a 6.3x switch speedup over the prefix-slot model for
~2.9x the resident KV memory (with two branches). `llama_memory_seq_cp` cost
itself is negligible (<1 ms).

## Interpretation

- On CPU consumer hardware, prefill dominates agent branch switching; keeping
  semantic branches resident turns a 40 s switch into a 1.1 s switch.
- The prefix_slot condition models the *best case* of a single-slot
  longest-prefix cache (root always retained); real single-slot servers do at
  least this much work when alternating branches.
- The per-switch gap grows with branch context size and turn history length;
  turns here were small (~30 tokens), so 1.1 s/switch is mostly turn prefill.

## Caveats

- Single machine, CPU-only, one model, small fixture branches.
- prefix_slot is an in-process emulation, not a llama-server measurement.
- Memory figures are estimates from token residency, not RSS measurements.

## Next

- Disk slot save/restore condition (llama.cpp `/slots/{id}/save|restore`) as a
  fourth point between resident-KV and re-prefill.
- Quantized KV (`--cache-type-k/v q8_0`) footprint axis.
- Shift-only splice probe (`scripts/run_kv_splice_probe.py`) as the 0%-repair
  baseline of a CacheBlend-style recompute curve.
