# Python Semantic Branch Benchmark

Date: 2026-06-07

## Goal

Connect the Python semantic router to the native JSON-plan KV runner.

The experiment is:

```text
agent event trace
-> Python semantic segmentation
-> shared root context
-> auth / QR / billing branches
-> JSON KV plan
-> llama.cpp sequence KV copy/eval/compare
```

This tests whether semantic branches chosen in Python can be executed as exact
shared-prefix KV branches in the native runtime.

## Hardware and Model

This is student research on local hardware, not a datacenter GPU benchmark.

The project intentionally uses a small local GGUF model because it is practical
for repeated experiments on this machine:

```text
model: /data/llama/models/Qwen3-4B-Q4_K_M.gguf
threads: 10
context: 2048
sequences: 8
batch: 1024
observed local speed target: about 20 tok/s class for Qwen3-4B in this setup
```

The point of this benchmark is cache architecture correctness, not frontier
model answer quality.

## Command

```bash
python scripts/run_semantic_branch_benchmark.py
```

Result file:

```text
benchmark-results/2026-06-07-semantic-branch-benchmark.json
```

## Plan Summary

The Python planner emitted:

```text
segments: 4
ops: 13
branches: auth, qr, billing
measured branches: auth, qr
root tokens: 67
estimated root prefill tokens avoided: 201
```

The avoided-token estimate is:

```text
67 root tokens * 3 branch copies = 201 root tokens not re-prefilled
```

## Native KV Result

The native runner copied the shared root KV into separate branch sequences,
extended each branch, then compared selected branches against scratch
evaluations of:

```text
root + same branch text
```

Results:

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

KV root copy latency was effectively negligible in this small run:

```text
root -> auth: 0.049631 ms
root -> qr: 0.031680 ms
root -> billing: 0.028854 ms
```

## Conclusion

This is a positive result for the next layer of the project.

The repo now proves more than prompt rewriting:

```text
Python can choose semantic branches.
Python can emit a native KV execution plan.
llama.cpp can execute that plan with sequence-level KV copies.
Shared-prefix branches are exact against scratch evaluation.
```

The current strongest claim is:

> Semantic shared-prefix KV branching is exact and can be driven by a Python
> semantic router.

This is not yet arbitrary semantic cache recomposition. Removing a middle block
like:

```text
A + noise + B -> A + B
```

still produced divergent logits in the earlier native probe. The useful path is
therefore a semantic prefix tree / branch cache, not unsafe middle-token KV
surgery.

## Next Steps

1. Replace the synthetic trace with a small real repo trace where an agent reads
   wrong files, switches topic, then returns to an old topic.
2. Benchmark branch switching order:

```text
A
A + QR
A + auth
switch QR -> auth -> QR
```

3. Track memory limits as branch count grows.
4. Add a quality task on top of the KV benchmark, so the result measures both
   token efficiency and task success.
5. Write the first public research note around the narrow validated claim:

```text
Semantic KV Branch Caching for Long-Running Coding Agents
```
