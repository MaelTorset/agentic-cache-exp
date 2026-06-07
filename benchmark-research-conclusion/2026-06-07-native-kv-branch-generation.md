# Native KV Branch Generation

Date: 2026-06-07

## Goal

Move beyond logits comparison and generate text directly from a native KV
branch.

The tested plan is:

```text
eval root on seq 0
copy root KV to seq 1
eval auth branch on seq 1
eval task on seq 1
generate greedy text from seq 1

eval root + auth branch + task from scratch on seq 2
compare pre-generation logits
generate greedy text from seq 2
```

This tests whether a semantic KV branch can be used as a real generation state,
not only as a prefill/logits probe.

## Command

```bash
python scripts/run_fixture_repo_native_generation.py
```

Result file:

```text
benchmark-results/2026-06-07-fixture-repo-native-generation.json
```

## Setup

```text
model: /data/llama/models/Qwen3-4B-Q4_K_M.gguf
runner: build/native-probes/semantic-kv-runner
context: 2048
sequences: 4
threads: 10
generation: greedy
max output tokens: 96
```

## Result

The native branch and scratch path matched before generation:

```text
pre_generation_branch_vs_scratch:
  top_token_match: true
  mean_abs_diff: 0.00000000
  rms_diff: 0.00000000
  max_abs_diff: 0.00000000
  cosine_similarity: 1.00000000
  top_k_overlap: 5/5
```

The generated text also matched exactly:

```text
generation_text_match: true
```

Both modes produced the same 96-token greedy output:

```text
1. Patch `backend/src/auth/cookie.ts` to set `secure` to `false` when `env`
is "development" or "test".
2. Root cause: The `secure` flag was incorrectly set to `true` in
development/test environments, leading to cookie rejection.
3. Development/test behavior: Cookies are sent without `Secure` flag, which is
expected for local HTTP traffic.
4. Production behavior: Cookies are sent with `Secure` flag,
```

This answer is only partially correct for the fixture bug. The fixture bug is
primarily about `SameSite=None` without `Secure` on local HTTP, not only the
`Secure` flag. This confirms again that the cache mechanism can be correct while
the small local model's bug-resolution quality remains limited.

## Token and Latency Data

Prompt segments:

```text
root: 68 tokens
branch_auth: 171 tokens
task: 55 tokens
```

Branch path:

```text
root eval once: 3354.02 ms
copy root -> auth branch: 0.053956 ms
branch_auth eval: 7057.18 ms
task eval: 2586.19 ms
generation: 33212.00 ms / 96 tokens
```

Scratch path:

```text
root eval: 4494.54 ms
branch_auth eval: 7381.82 ms
task eval: 2498.32 ms
generation: 15157.11 ms / 96 tokens
```

The generation latency is noisy in this single native run, so it should not be
used yet as a speed claim. The robust result is exactness:

```text
same pre-generation logits
same greedy generated text
```

The prefill-side cache result is still meaningful:

```text
root KV copy latency: about 0.054 ms
root eval latency: about 3.35-4.49 s in this run
```

## Conclusion

This is the strongest runtime result so far.

The project now validates:

```text
Python semantic routing
-> JSON native KV plan
-> llama.cpp sequence KV copy
-> native generation from the copied branch
-> exact match against scratch generation under greedy decoding
```

The claim can be upgraded from:

```text
shared-prefix KV branching preserves logits
```

to:

```text
shared-prefix KV branching preserves greedy generation
```

The next research step is not to prove exactness again. The next step is to
measure usefulness:

```text
cold-cache vs warm-cache vs branch-cache
larger repo fixture
more realistic task oracle
multiple branch switches before generation
memory footprint as branch count grows
```

