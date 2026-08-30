# Branch-Switch Benchmark, Large Context (Scaling Check)

Date: 2026-07-24
Model: Qwen3-4B-Q4_K_M (CPU, 10 threads, ctx 65536, batch 8192, 10 sequences)
Workload: root padded to 5229 tokens (~3.9x the small run), 6 switches
(`auth/qr` alternating), 3 repeats, medians.
Raw data: `benchmark-results/branch-switch-benchmark-large.json`.
Small-context companion: `2026-07-23-branch-switch-benchmark.md`.

## Result

```text
                          small (1341-tok root)   large (5229-tok root)
branch_cache / switch:        1110 ms                 5354 ms
prefix_slot  / switch:        7002 ms  (6.3x)        26887 ms  (5.0x)
scratch      / switch:       40597 ms (36.6x)       315313 ms (58.9x)
```

Exact greedy match branch-vs-scratch: 3/3 runs.

KV residency (fp16 estimate): branch_cache 16220 tokens (~2.3 GB) vs
prefix_slot 5489 tokens (~0.8 GB).

## Interpretation

- The advantage over full re-prefill **grows with context size** (36.6x ->
  58.9x): re-prefill cost scales with total context, resident-branch switch
  cost scales only with the new turn plus attention over the branch.
- The advantage over the single-slot prefix cache stays roughly constant
  (6.3x -> 5.0x): both conditions keep root resident; the gap is the branch +
  turn re-eval, which grew more slowly than root here.
- branch_cache per-switch cost grew 1.1 s -> 5.4 s despite identical turn
  sizes: per-token decode cost rises with attention window length. Resident
  branches remove re-prefill, not attention cost — carrying less context (see
  dead-reads benchmark) remains complementary.

## Reframing (harness discussion, 2026-07-24)

In an append-only hosted harness, task switching never re-prefills; the
scratch column instead models the cost of **forgetting** (any mid-context
removal invalidates the prefix cache from that point). The correct reading of
this benchmark for harness design: with a branch-structured KV layout,
dropping dead context is free, while an append-only layout pays the scratch
column (and, hosted, the corresponding cache-write cost) to achieve the same
forgetting. Companion experiment: `scripts/run_dead_reads_benchmark.py`.

## Caveats

Single machine, CPU-only, one model; 3 repeats; branch contexts still small
relative to root.
