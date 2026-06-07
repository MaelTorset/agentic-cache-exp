# Fixture Repo Branch Benchmark

Date: 2026-06-07

## Goal

Move from a purely synthetic trace to a small fake repository that behaves like
a real coding session.

The fixture repo contains real files for:

```text
auth cookie generation
auth cookie tests
QR scanner UI
QR route fallback
billing invoice helpers
analytics events
```

The benchmark simulates an agent reading those files in one session, then
switching focus to the auth cookie bug while keeping QR available as a separate
branch.

## Command

```bash
python scripts/run_fixture_repo_branch_benchmark.py
```

Result file:

```text
benchmark-results/2026-06-07-fixture-repo-branch-benchmark.json
```

Trace file:

```text
benchmark-results/fixture-repo-session.jsonl
```

## Result

```text
event count: 12
branches: auth, qr, billing, analytics
measured branches: auth, qr
root tokens: 68
estimated root prefill tokens avoided: 272
```

The avoided-token estimate is:

```text
68 root tokens * 4 branch copies = 272 root tokens not re-prefilled
```

## Correctness

The native runner compared cached shared-prefix branches against full scratch
evaluations of the same logical prompt.

```text
auth_branch_vs_scratch:
  top_token_match: true
  mean_abs_diff: 0.00000000
  rms_diff: 0.00000000
  max_abs_diff: 0.00000000
  cosine_similarity: 1.00000000
  top_k_overlap: 5/5

qr_branch_vs_scratch:
  top_token_match: true
  mean_abs_diff: 0.00000000
  rms_diff: 0.00000000
  max_abs_diff: 0.00000000
  cosine_similarity: 1.00000000
  top_k_overlap: 5/5
```

## Conclusion

This is the first repo-shaped test for the project.

It is still controlled and fake, but it is closer to the real target workflow:

```text
agent reads actual files
semantic router labels them by topic
Python emits a JSON KV plan
native llama.cpp runner branches the shared prefix
branch logits match scratch exactly
```

The useful next experiment is to replay a real anonymized agent trace from this
repository or another small open-source repo, then compare:

```text
full reread baseline
prompt-only routed baseline
semantic KV branch cache
```

