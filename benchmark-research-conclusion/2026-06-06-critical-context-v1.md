# Critical Context V1 Matrix

Date: 2026-06-06

## Goal

Improve the previous negative result without hiding it.

The previous corrected matrix showed that naive Forget was token-efficient but
lost quality on the coding task. This run adds a new strategy:

```text
forget_critical_context_v1
```

It keeps the same external event log and still forgets unrelated QR, frontend,
billing, and analytics files, but it protects a small critical-context lane:

```text
failed tests
expected behavior
root-cause evidence
likely patch target
recent task instruction
```

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

Matrix:

```text
noise_files: 4, 8, 12
output_budgets: 48, 96, 160
warmup: 1
runs per case: 1
cases: 9
```

Result file:

```text
benchmark-results/2026-06-06-qwen3-4b-critical-context-matrix.json
```

## Compared Modes

```text
industry_raw_cached:
  full rolling history with runtime prompt cache

forget_routed_cached_v0:
  naive segment scorer and prompt packer

forget_critical_context_v1:
  V0 plus protected critical-context lane

oracle_relevant_only:
  non-deployable upper-bound baseline with only relevant auth context
```

## Aggregate Result

```text
cases: 9

industry quality pass rate avg: 0.1111

forget_routed_cached_v0:
  quality pass rate avg: 0.1111
  quality delta avg: 0.0000
  runtime prompt token reduction avg: 81.57%
  latency reduction avg: 23.75%

forget_critical_context_v1:
  quality pass rate avg: 0.6667
  quality delta avg: +0.5556
  runtime prompt token reduction avg: 80.02%
  latency reduction avg: 44.59%
  tokens per success avg: 436.0

oracle_relevant_only:
  quality pass rate avg: 0.6667
  runtime prompt token reduction avg: 84.56%
  latency reduction avg: 39.72%
  tokens per success avg: 337.0
```

## Interpretation

This improves the negative result.

The important signal is not just that V1 includes the right facts. V0, V1,
industry, and oracle all exposed the same four oracle facts in this synthetic
trace:

```text
patch file
SameSite=None without Secure
local development expects SameSite=Lax
production expects SameSite=None with Secure
```

V0 still often failed because the model used the local `SameSite=None` root
cause but proposed the wrong behavior change: adding `Secure` locally instead
of switching local development to `SameSite=Lax`.

V1 improves quality by making the failed test and expected behavior a protected,
first-class section before the implementation file and normal dynamic context.
So the result supports a narrower claim:

> For small local models, context ordering and critical-constraint preservation
> matter, not only token reduction.

## Single Diagnostic Case

For `noise_files=12` and `output_budget=160`:

```text
industry_raw_cached:
  runtime prompt tokens: 3600
  latency: 240200.29 ms
  quality pass: true

forget_routed_cached_v0:
  runtime prompt tokens: 402
  latency: 40146.79 ms
  quality pass: false

forget_critical_context_v1:
  runtime prompt tokens: 436
  latency: 32802.02 ms
  quality pass: true

oracle_relevant_only:
  runtime prompt tokens: 337
  latency: 36310.66 ms
  quality pass: true
```

V1 used slightly more prompt than V0 but recovered quality and was faster in
this case.

## Caveats

This is still a synthetic benchmark.

The oracle facts are present in the context because the agent read the failing
test and log. That is realistic for a coding-agent trace, but it means this run
does not prove discovery of hidden facts. It tests whether the memory controller
keeps and prioritizes task-critical facts after noisy exploration.

The quality oracle is still string-based. It should eventually be replaced or
augmented with executable patch tests.

The matrix uses one run per case. More runs and trace variants are needed before
making a strong claim.

## Next Steps

1. Add adversarial synthetic traces where noise is not named `noise_file` and
   does not explicitly say it is unrelated.

2. Add executable patch evaluation for the coding task.

3. Add a `rolling_window` baseline and a `retrieval_top_k` baseline.

4. Run multiple seeds per matrix cell.

5. Test on a real small repository trace where the agent reads wrong files
   before solving an actual bug.
