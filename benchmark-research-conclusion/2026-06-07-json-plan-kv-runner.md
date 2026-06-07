# JSON-Plan Native KV Runner

Date: 2026-06-07

## Goal

Replace the hardcoded native probe with a reusable JSON-plan runner.

The runner accepts:

```text
segments + ops
```

Supported ops:

```text
eval
copy
remove
shift
keep
compare
```

This is the first bridge from a Python semantic router toward a native
llama.cpp KV-cache execution plan.

## Setup

```text
runtime: native llama.cpp C API
model: Qwen3-4B-Q4_K_M.gguf
threads: 10
context: 2048
batch: 1024
```

Build:

```bash
cmake -S native -B build/native-probes
cmake --build build/native-probes -j 2
```

## Prefix Branch Plan

Command:

```bash
./build/native-probes/semantic-kv-runner \
  -m /data/llama/models/Qwen3-4B-Q4_K_M.gguf \
  --plan examples/native/prefix_branch_plan.json \
  --threads 10 \
  --ctx 2048 \
  --batch 1024
```

Result:

```text
auth_branch_vs_scratch:
  top_token_match: true
  mean_abs_diff: 0.00000000
  rms_diff: 0.00000000
  max_abs_diff: 0.00000000
  cosine_similarity: 1.00000000
  top_k_overlap: 5/5
```

KV copy latency:

```text
copy root -> qr branch: 0.040536 ms
copy root -> auth branch: 0.022017 ms
```

Conclusion:

> The JSON runner reproduces the exact shared-prefix branching result.

## Middle Removal Plan

Command:

```bash
./build/native-probes/semantic-kv-runner \
  -m /data/llama/models/Qwen3-4B-Q4_K_M.gguf \
  --plan examples/native/middle_removal_plan.json \
  --threads 10 \
  --ctx 2048 \
  --batch 1024
```

Result:

```text
mutated_vs_scratch:
  top_token_match: true
  mean_abs_diff: 0.30245631
  rms_diff: 0.37428763
  max_abs_diff: 2.26411819
  cosine_similarity: 0.98860057
  top_k_overlap: 3/5
```

KV mutation latency:

```text
copy full -> mutated: 0.690159 ms
remove noise: 0.020348 ms
shift task positions: 0.051090 ms
remove final task token: 0.007184 ms
```

Conclusion:

> Middle removal remains mechanically possible but not distribution-preserving.

The top token matched in this case, but the logits diverged. This should remain
classified as an unsafe experimental mutation.

## Research Conclusion

The runner makes the next research step concrete:

```text
Python semantic router
-> JSON KV plan
-> native llama.cpp sequence operations
-> measured branch correctness and latency
```

The strongest validated path remains:

```text
semantic prefix tree / branch cache
```

not arbitrary middle-token deletion.
