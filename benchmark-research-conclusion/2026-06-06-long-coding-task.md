# Long Coding Task Benchmark: Noise Reads + Forget

Date: 2026-06-06

## Status

This is a pilot result and has been superseded by:

```text
benchmark-research-conclusion/2026-06-06-corrected-matrix.md
```

The pilot query leaked expected answer facts into the prompt, so its quality
result is not valid as research evidence. Keep this note to document the
iteration history, but use the corrected matrix result for conclusions.

## Goal

Test the first useful product-shaped idea:

> Make both modes read the same useless files during a long coding task, then let the Forget mode remove irrelevant context from the active prompt while the industry baseline keeps everything.

This benchmark evaluates both:

- token efficiency;
- task answer quality.

## Setup

Hardware:

```text
CPU: Intel Xeon E-2176M, 6 cores / 12 threads
RAM: 62 GiB
GPU: NVIDIA Quadro P2000 Mobile
runtime: llama.cpp local GGUF, CPU-focused run
```

Model:

```text
Qwen3-4B-Q4_K_M.gguf
context size: 8192
prompt cache: enabled
threads: 10
```

Benchmark:

```text
script: scripts/run_long_coding_task_benchmark.py
noise files: 12
warmup: 1
runs: 1
max output tokens: 96
```

Task:

```text
Solve an authentication cookie bug.
Expected patch file: backend/src/auth/cookie.ts
Expected root cause: SameSite=None without Secure is rejected on local HTTP.
Expected fix: use SameSite=Lax locally; keep SameSite=None with Secure in production.
```

Noise read by both modes:

```text
QR files
frontend files
billing files
analytics files
```

## Compared Modes

```text
industry_raw_cached:
  full rolling history remains active
  llama.cpp prompt cache enabled

forget_routed_cached:
  same event log preserved externally
  irrelevant files omitted from active prompt
  llama.cpp prompt cache enabled
```

## Real Qwen3-4B Result

Warm-cache measured run:

```text
industry_raw_cached:
  runtime prompt tokens: 3623
  cached prompt tokens: 3622
  cache hit ratio: 0.9997
  latency: 31610.34 ms
  quality pass: false

forget_routed_cached:
  runtime prompt tokens: 425
  cached prompt tokens: 424
  cache hit ratio: 0.9976
  latency: 17565.59 ms
  quality pass: true
```

Derived result:

```text
runtime prompt token reduction: 88.27%
latency reduction: 44.43%
```

Offline routing estimate:

```text
raw token estimate: 3406
forget token estimate: 306
estimated token reduction: 91.02%
stable prefix estimate: 189
```

## Quality Oracle

The answer had to:

```text
mention backend/src/auth/cookie.ts
mention SameSite=None
mention Secure
mention SameSite=Lax for local/development
avoid focusing on QR/billing/frontend/analytics noise
```

Observed:

```text
industry_raw_cached:
  failed because the answer did not reach SameSite=Lax within the 96-token output budget.

forget_routed_cached:
  passed all checks.
```

This is not a final proof that Forget always improves quality. It is a useful signal: under the same model and output budget, a smaller active context helped the model reach the expected fix sooner.

## Why This Is More Useful Than The Earlier Benchmark

The earlier Forget-vs-industry benchmark mostly showed active prompt reduction. This one adds a task oracle.

The useful claim becomes:

> Forget can reduce active prompt size and preserve or improve task success under a fixed output budget in noisy coding sessions.

That is much closer to a practical agent system than only measuring latency.

## Current Weaknesses

The benchmark is still synthetic.

The quality oracle is string-based and simple.

The Forget decider is still rule-based:

```text
segment_scorer_v0
```

The result may depend on output-token budget. With more generated tokens, the raw baseline might eventually mention the missing fix. That does not invalidate the result, but it narrows the claim:

```text
Forget improved answer efficiency under constrained output budget.
```

## Next Steps

1. Run multiple seeds / trace variants.

   Vary:

   ```text
   number of noise files
   topic order
   relevant file position
   output token budget
   ```

2. Add a real repository trace.

   Use an actual small codebase where the agent reads wrong files before solving a targeted bug.

3. Add retrieval.

   The next benchmark should prove:

   ```text
   forgotten context can be recovered when the user switches back to that topic
   ```

4. Add stronger quality scoring.

   Keep oracle checks, then optionally add an LLM judge as a secondary signal.

5. Compare more baselines.

   Add:

   ```text
   rolling_window
   compact_summary
   retrieval_top_k
   forget_with_retrieval
   ```

## Practical Direction

The system becomes useful if it acts as an agent memory controller:

```text
keep all history externally
select active context per task
hard-forget explicitly excluded topics
pack stable context for cache reuse
retrieve omitted context on topic return
measure quality and token efficiency continuously
```

This benchmark suggests that direction is worth pursuing.
