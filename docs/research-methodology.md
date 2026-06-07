# Research Methodology

This project should be treated as student research, not a product benchmark.

The current credible claim is narrow:

> In noisy coding traces, active-context forgetting can reduce runtime prompt tokens while preserving task-answer quality under a fixed output budget.

## Credibility Rules

1. Do not report a single cherry-picked run as the conclusion.

   Run a matrix over:

   ```text
   noise file count
   output token budget
   warm vs cold cache
   ```

2. Report both estimated and runtime tokens.

   Estimated tokens are useful for offline development. Runtime tokens from the model server are stronger evidence.

3. Keep quality checks deterministic first.

   Use oracle checks for required facts before adding LLM judges.

4. Separate latency from context efficiency.

   Warm prompt cache can make latency differences smaller. Active prompt size is still a useful metric because context budget and attention pollution remain.

5. Preserve limitations.

   Every report should state:

   ```text
   synthetic or real trace
   model and hardware
   cache mode
   output budget
   scorer version
   known failure modes
   ```

## Benchmark Matrix

Run offline:

```bash
ACL_ECHO=1 python scripts/run_long_coding_matrix.py
```

Run against local Qwen3-4B:

```bash
ACL_NOISE_FILES=4,8,12 \
ACL_OUTPUT_BUDGETS=48,96 \
ACL_WARMUP=1 \
ACL_RUNS=1 \
ACL_TIMEOUT_SECONDS=900 \
python scripts/run_long_coding_matrix.py
```

The script writes JSON results into `benchmark-results/`.

Before drawing conclusions, use [research-protocol.md](research-protocol.md) as the fixed protocol. In particular, expected answer facts should be kept in the oracle, not leaked into the model prompt.

## Minimum Evidence For A Useful Claim

A result is interesting only if:

```text
Forget quality pass rate >= industry raw cached pass rate
runtime prompt token reduction > 50%
the omitted context remains recoverable externally
```

The retrieval condition is not implemented yet, so current claims must not imply complete long-term memory management.
