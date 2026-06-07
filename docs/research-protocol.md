# Research Protocol

Date: 2026-06-06

## Research Question

Can active-context forgetting improve token efficiency in noisy coding-agent sessions without reducing task quality, compared with an industry-style cached rolling-history prompt?

## Primary Hypothesis

`forget_critical_context_v1` will use fewer runtime prompt tokens than `industry_raw_cached` while maintaining equal or better quality pass rate.

## Primary Metrics

- `quality_pass_rate`
- `runtime_prompt_token_reduction_ratio`
- `tokens_per_success`
- `oracle_fact_exposure_count`

## Secondary Metrics

- latency reduction ratio;
- prompt cache hit ratio;
- output tokens per second;
- estimated token reduction ratio.

## Compared Modes

Current:

- `industry_raw_cached`: full active history plus runtime prompt cache.
- `forget_routed_cached_v0`: same event log preserved externally, irrelevant segments omitted from active prompt, runtime prompt cache still enabled.
- `forget_critical_context_v1`: V0 plus a protected lane for failed tests, expected behavior, root-cause evidence, likely patch targets, and recent task instruction.
- `oracle_relevant_only`: non-deployable upper-bound baseline containing only relevant task context.

Planned:

- `rolling_window`
- `recent_k_events`
- `retrieval_top_k`
- `forget_with_retrieval`

## Fixed Initial Matrix

```text
noise_files: 4, 8, 12
output_budgets: 48, 96, 160
model: Qwen3-4B-Q4_K_M.gguf
context: 8192
runtime: llama.cpp
temperature: deterministic server defaults unless explicitly recorded
```

## Quality Oracle

Expected answer facts are hidden from the prompt and used only in evaluation.

The answer must:

- identify the auth cookie code path;
- mention that `SameSite=None` without `Secure` is rejected on local HTTP;
- describe local-development behavior using `SameSite=Lax`;
- avoid focusing on QR, frontend, billing, or analytics noise.

## Exclusion Rules

Do not discard a run just because it makes Forget look worse.

Discard only if:

- the model server crashes;
- the request exceeds context;
- the output is missing due to runtime error;
- the result JSON is incomplete.

Every discarded run must be logged with the error.

## Threats To Validity

- synthetic traces may overfit the heuristic scorer;
- Qwen3-4B is a small local model, not a frontier model;
- output budget strongly affects quality pass rate;
- prompt-cache warmup may compress latency differences;
- the current Forget decider is rule-based;
- `critical_context_v1` can become benchmark-specific if it only mirrors the oracle checks;
- oracle facts may be present in realistic failing-test context, so exposure must be reported;
- retrieval of forgotten context is not implemented yet;
- quality checks are string/oracle based, not executable patch tests.

## Reporting Rule

Conclusions should prioritize aggregate matrix results over single favorable examples.
