# Shift-Only KV Splice Probe (0%-Repair Baseline)

Date: 2026-07-23
Model: Qwen3-4B-Q4_K_M (CPU). Script: `scripts/run_kv_splice_probe.py`.
Raw data: `benchmark-results/kv-splice-shift-only-probe.json`.

## Setup

Evaluate `A + noise + B` (A = fixture root, noise = qr branch, B = auth
branch), then on a copy: remove the noise KV (`llama_memory_seq_rm`) and shift
B's KV left by the noise length (`llama_memory_seq_add`, RoPE re-alignment).
Compare the resulting pre-generation state against a scratch `A + B` eval and
against the untouched noisy state. This is the "0% recompute" point of a
CacheBlend-style repair curve (see `docs/research-watch-2026-07.md`).

## Result

```text
spliced vs scratch: cosine 0.918, mean_abs_diff 1.05, top-1 match, top-20 overlap 18/20
spliced vs noisy:   cosine 0.975, mean_abs_diff 0.58, top-1 match, top-20 overlap 17/20
noisy   vs scratch: cosine 0.895, mean_abs_diff 1.26, top-1 match, top-20 overlap 17/20

greedy generation (48 tokens): spliced matches NEITHER scratch NOR noisy;
first divergence at character 21 in both cases.
```

## Interpretation

- Confirms theory: shift-only re-alignment is **not exact** — B's cached KV
  still encodes attention to the removed noise. The spliced state is closer to
  the noisy state (cos 0.975) than to the scratch target (0.918).
- But it is **not catastrophic** either: top-1 token identical across all
  three states, 18/20 top-k overlap vs scratch, and the spliced state is
  slightly *closer* to scratch than the noisy state is (0.918 vs 0.895).
  Positional repair alone recovers a measurable fraction of the gap.
- Greedy decoding diverges within ~21 characters, so exact-match workloads
  cannot use shift-only splicing; approximate/sampled workloads might.

## Next

Add a selective boundary recompute op to the native runner and trace the
recompute-ratio -> exact-match curve from this 0% point toward CacheBlend's
reported ~15% repair threshold.
