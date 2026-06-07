# Corrected Matrix Result: No Prompt Leakage

Date: 2026-06-06

## Why This Run Matters

The previous long-coding-task benchmark was useful as a pilot, but the prompt leaked expected answer facts by explicitly telling the model to mention:

```text
backend/src/auth/cookie.ts
SameSite=None without Secure
SameSite=Lax for local development
```

That made the quality result too easy. The prompt has now been corrected: expected facts live only in the hidden oracle, not in the model query.

## Corrected Query

```text
/no_think
Solve only the authentication cookie bug. Forget QR, frontend, billing, and analytics noise. Answer with the patch target, root cause, and behavior change.
```

## Matrix Setup

Model and runtime:

```text
Qwen3-4B-Q4_K_M.gguf
llama.cpp
context size: 8192
prompt cache: enabled
CPU-focused local run
```

Matrix:

```text
noise_files: 4, 8, 12
output_budgets: 48, 96
warmup: 1
runs per case: 1
cases: 6
```

Result file:

```text
benchmark-results/2026-06-06-qwen3-4b-long-coding-matrix.json
```

## Aggregate Result

```text
cases: 6
estimated token reduction avg: 85.47%
runtime prompt token reduction avg: 81.57%
latency reduction avg: 22.99%

Forget quality pass rate avg: 0.00
Industry quality pass rate avg: 0.1667
quality delta avg: -0.1667
```

## Interpretation

The corrected benchmark is a negative quality result for the current Forget implementation.

It shows:

```text
Forget is strongly more token-efficient.
Forget is generally faster.
Forget does not currently preserve task quality on this coding task.
```

This is more credible than the earlier pilot because it avoids prompt leakage and uses a small matrix instead of one favorable case.

## Diagnostic Case

A follow-up diagnostic run with 12 noise files and 160 output tokens showed:

```text
industry_raw_cached:
  runtime prompt tokens: 3600
  latency: 56386.89 ms
  quality pass: true

forget_routed_cached:
  runtime prompt tokens: 402
  latency: 34182.74 ms
  quality pass: false
```

Forget answer failure:

```text
It identified the file and root cause, but proposed adding Secure with SameSite=None instead of switching local development to SameSite=Lax.
```

So the failure is not that Forget loses all useful context. It preserves the broad bug location and cause, but it does not reliably preserve or emphasize the exact expected local/prod behavior.

## Research Conclusion

The useful research problem is now sharper:

> Active-context forgetting can dramatically reduce prompt tokens, but naive hard forgetting and simple prompt packing are not enough. The system must preserve task-critical constraints, especially test expectations and required behavior.

The current system is not yet a useful coding-agent memory manager. It is a benchmark harness that revealed the next technical requirement.

## Next Technical Step

Improve the active-context packer so task-critical constraints are explicit and prioritized.

Concrete next experiment:

```text
critical_context lane:
  failed tests
  expected behavior
  root cause evidence
  patch target

supporting_context lane:
  implementation file
  logs
  durable decisions

forgotten_context:
  excluded topic files
```

Then compare:

```text
industry_raw_cached
forget_routed_cached_v0
forget_with_critical_context_v1
oracle_relevant_only
```

Success condition:

```text
Forget v1 must keep >70% runtime prompt token reduction while matching or beating industry quality pass rate.
```

## Credibility Note

This negative result should stay in the repo. It prevents the project from looking like a cherry-picked demo and gives a clear research direction.
