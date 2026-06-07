# Cache Runtime Requirements

Date: 2026-06-06

## Why This Matters

The first benchmark only proves active-context routing:

```text
full event log
-> select useful context
-> build a smaller prompt
```

That is not enough for the original research idea.

The harder requirement is:

```text
Can the system stop using noisy context in the active prompt while keeping its
cache state reusable if the task returns to that context?
```

## Important Distinction

There are three different cache claims.

### 1. Active Prompt Forgetting

The system removes irrelevant segments from the current prompt.

Status:

```text
implemented
```

This saves prompt tokens, but does not prove KV-cache preservation.

### 2. Whole-Prompt Cache Residency

The runtime keeps multiple prompt shapes cached.

Example sequence:

```text
full_history_prompt
forgotten_active_prompt
full_history_prompt again
forgotten_active_prompt again
```

If the second `full_history_prompt` has a high `cached_tokens` ratio, then the
runtime kept the full prompt reusable after the forgotten prompt ran.

Status:

```text
probe implemented in scripts/run_cache_residency_probe.py
```

This is useful, but it still does not prove arbitrary KV recomposition.

### 3. Shared-Prefix KV Branching

The system evaluates a shared prefix once, copies that KV state into semantic
branches, then continues each branch independently.

Example:

```text
A
copy A KV -> A + auth
copy A KV -> A + QR
generate from A + auth
```

Status:

```text
implemented for llama.cpp native runner
```

The current native result validates:

```text
branch logits match scratch logits exactly
greedy generation from a copied branch matches scratch generation exactly
```

This is the practical near-term path for semantic cache branching.

### 4. Modular KV Recomposition

The system stores KV blocks for segments and later builds a new active context
by combining only selected blocks.

This is the original hard idea.

Status:

```text
not implemented
```

For a normal causal transformer, arbitrary middle-token removal is not free.
KV for later tokens depends on earlier tokens through attention and position.
If a prompt was:

```text
A + noise + B
```

the cached KV for `B` is not generally valid for:

```text
A + B
```

because `B` was computed while attending to `noise`, and its token positions
also changed.

So true modular reuse needs a runtime design that constrains or records how
segments are computed.

## What We Can Test Now

Use:

```bash
python scripts/run_cache_residency_probe.py
python scripts/run_fixture_repo_branch_benchmark.py
python scripts/run_fixture_repo_native_generation.py
```

The script runs:

```text
full_history_first
forgotten_active_first
full_history_return
forgotten_active_return
```

and reports:

```text
runtime_prompt_tokens
cached_prompt_tokens
runtime_prompt_cache_hit_ratio
full_history_likely_resident_after_forget
forgotten_prompt_likely_resident_on_reuse
```

## What Would Make The Big Idea Real

A real cache-aware runtime would need one of these approaches:

```text
prefix-tree cache:
  cache reusable shared prefixes and branch between prompt shapes

session snapshot cache:
  keep full-history sessions cached while running shortened sessions separately

chunked prefill with constrained attention:
  precompute chunks with explicit attention rules so they can be recombined

retrieval plus recomputation:
  keep forgotten context externally and recompute only when it becomes relevant
```

The first two are practical near-term experiments.

The third is the research-heavy path.

The fourth is easiest to ship, but least revolutionary.
