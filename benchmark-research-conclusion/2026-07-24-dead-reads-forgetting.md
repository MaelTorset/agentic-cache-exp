# Dead-Reads Forgetting Benchmark (Harness Scenario)

Date: 2026-07-24
Model: Qwen3-4B-Q4_K_M (CPU). Script: `scripts/run_dead_reads_benchmark.py`.
Raw data: `benchmark-results/dead-reads-benchmark.json`. Single run.

## Scenario

Reframing after harness discussion (2026-07-24): in an append-only hosted
harness, task switching never re-prefills — the real cost is carrying dead
context and the cache invalidation paid to remove it. This benchmark models an
agent that read three large files (~1300 tokens each) of which only one
matters, then answers the auth-bug task under three layouts:

- `noisy`: append-only status quo (root + dead1 + useful + dead2 + task)
- `clean`: branch layout — each read on its own KV branch, dead ones dropped
  for free (root + useful + task)
- `spliced`: a-posteriori forgetting without branch structure — remove dead
  reads' KV from the noisy state, RoPE-shift survivors left

## Results

Cost of forgetting the two dead reads:

```text
branch layout:       0 tokens invalidated (drop the branches)
append-only layout:  4064 tokens invalidated (re-prefill + cache re-write)
context carried:     1474 tokens (clean) vs 4175 (noisy) = -65%
```

Pre-generation logits (vs clean reference):

```text
noisy_vs_clean:   cosine 0.979, top-1 match, top-20 overlap 17/20
spliced_vs_clean: cosine 0.984, top-1 match, top-20 overlap 18/20
```

Greedy answers (96 tokens, bug oracle, 1 run — indicative only):

```text
clean 1/4, noisy 1/4, spliced 3/4
```

All three name the correct file; none reproduces the exact greedy text of the
others (splice remains non-exact, consistent with the shift-only probe).

## Interpretation

- The economic argument for branch-structured KV in a harness is the
  forgetting cost: free vs ~4k invalidated tokens here, and the ratio scales
  with how much dead context precedes/follows the removal point (hosted:
  cache-write billing on the re-written suffix).
- Shift-only splice damage is mild in this configuration (closer to clean than
  noisy is), unlike the branch-adjacent splice probe — the removed material
  here is bulky filler-like file content, suggesting splice error depends on
  how strongly the surviving KV attended to the removed spans.
- Quality signal (spliced 3/4 > clean 1/4) is a single greedy sample with a
  keyword oracle — noise, not evidence. Needs repeats + better oracle.

## Caveats

One run, one model, synthetic padded files, keyword oracle. The clean-layout
score of 1/4 vs earlier routed-prompt results suggests oracle sensitivity to
formatting; do not read quality rankings from this run.
