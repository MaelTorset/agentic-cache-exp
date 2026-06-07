# Fixture Repo Quality Matrix

Date: 2026-06-07

## Goal

Measure the part that the previous KV benchmark did not measure:

```text
Does routed context improve bug-resolution quality, token use, cache hits, or latency?
```

This benchmark uses the fake TypeScript repo fixture and compares:

```text
full_noise:
  complete session history with auth, QR, billing, and analytics files

routed_prompt:
  shared session context plus only the auth branch
```

The native KV branch probe is attached separately because llama-server measures
prompt-generation behavior, while the native runner measures exact KV sequence
branching.

## Setup

```text
model: Qwen3-4B-Q4_K_M.gguf
server: llama.cpp llama-server
base URL: http://127.0.0.1:8082
context: 8192
threads: 10
runs: 5
warmup: 1
temperature: 0.35
```

Output files:

```text
benchmark-results/2026-06-07-fixture-repo-quality-matrix.json
benchmark-results/2026-06-07-fixture-repo-quality-matrix-bullets.json
benchmark-results/fixture-repo-quality-kv-branch.json
```

## Quality Scoring

The scorer awards up to 4 points:

```text
1 point: mentions backend/src/auth/cookie.ts or equivalent
1 point: identifies SameSite=None without Secure/local HTTP issue
1 point: proposes SameSite=Lax for development/test
1 point: preserves SameSite=None + Secure in production
```

It subtracts 1 point if the answer blames unrelated QR, billing, or analytics
code as the cause.

## First Run: Free-Form Answer

```text
max output tokens: 128
runs per mode: 5
```

Results:

```text
full_noise:
  runtime prompt tokens avg: 857
  cached prompt tokens avg: 856
  cache hit ratio avg: 0.9988
  latency avg: 34.41 s
  score avg: 2.8 / 4
  scores: 3, 3, 3, 2, 3

routed_prompt:
  runtime prompt tokens avg: 322
  cached prompt tokens avg: 321
  cache hit ratio avg: 0.9969
  latency avg: 35.55 s
  score avg: 2.8 / 4
  scores: 3, 3, 3, 2, 3
```

Interpretation:

```text
routed_prompt used 62.4% fewer runtime prompt tokens
quality did not improve
latency did not improve because generation dominated and both prompts were cached
```

The free-form prompt was a weak quality test because Qwen3-4B often spent the
128-token budget explaining the local/dev failure and got truncated before
mentioning production behavior.

## Second Run: Forced Four-Bullet Answer

The prompt was tightened to request exactly four short bullets:

```text
1 file to patch
2 root cause
3 development/test behavior
4 production behavior
```

```text
max output tokens: 96
runs per mode: 5
```

Results:

```text
full_noise:
  runtime prompt tokens avg: 865
  cached prompt tokens avg: 864
  cache hit ratio avg: 0.9988
  latency avg: 15.04 s
  score avg: 2.8 / 4
  scores: 3, 3, 2, 3, 3

routed_prompt:
  runtime prompt tokens avg: 330
  cached prompt tokens avg: 329
  cache hit ratio avg: 0.9970
  latency avg: 12.62 s
  score avg: 2.6 / 4
  scores: 3, 3, 2, 2, 3
```

Interpretation:

```text
routed_prompt used 61.8% fewer runtime prompt tokens
routed_prompt was about 16.1% faster in this shorter-output run
quality still did not improve
neither mode reached 4/4 because the model still failed to include production behavior
```

This is an important negative result: on this fixture and this small model,
semantic routing reduces prompt size but does not automatically improve bug-fix
quality.

## Runtime Cache Result

llama-server prompt cache worked extremely well on repeated identical prompts:

```text
full_noise cache hit ratio: about 0.999
routed_prompt cache hit ratio: about 0.997
```

This means the "industry prompt cache" baseline is strong once the same prompt
shape repeats. In that regime, latency is dominated by generation tokens, not
prompt prefill.

The routed prompt still matters for:

```text
lower active context size
lower cache memory footprint
less noise in the prompt
more room before context limits
```

but this run does not prove better answer quality.

## Native KV Branch Result

The native KV branch probe still gives the strongest cache-specific result:

```text
auth_branch_vs_scratch:
  mean_abs_diff: 0.00000000
  cosine_similarity: 1.00000000
  top_k_overlap: 5/5

qr_branch_vs_scratch:
  mean_abs_diff: 0.00000000
  cosine_similarity: 1.00000000
  top_k_overlap: 5/5
```

Measured KV branch data:

```text
root tokens: 68
branch count: 4
estimated root prefill tokens avoided: 272
root eval latency: 2.57 s in the separate KV artifact
total KV copy latency: 0.174 ms in the separate KV artifact
```

Conclusion:

```text
KV branch preservation works exactly.
KV copy is effectively free compared with evaluating the root.
This benchmark still does not generate final answers from the native KV branch.
```

## Current Honest Claim

The strongest accurate claim is now:

> Semantic routing can reduce active prompt size by about 62% on this fixture,
> and shared-prefix KV branches preserve the model state exactly. On repeated
> prompts, llama.cpp's normal prompt cache already gives near-perfect cache
> hits, so the current benchmark does not show a quality win and only shows a
> latency win when output length is constrained.

## Next Steps

1. Add native generation from a branched KV sequence, so `kv_branch` is a true
   generation mode, not only a logits/prefill probe.
2. Make the quality task harder and less answer-leaky:

```text
larger repo fixture
more plausible wrong files
less explicit test-output wording
patch-style expected answer
```

3. Measure cold-cache, warm-cache, and branch-cache separately.
4. Track cache memory footprint, not just cached token count.
5. Repeat with a stronger local model if available, because Qwen3-4B did not
   reliably follow the forced answer format.

