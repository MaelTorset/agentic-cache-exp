# Native KV Probe Result

Date: 2026-06-07

## Goal

Move beyond prompt rewriting and test direct llama.cpp KV-cache operations.

This probe uses:

```text
llama_get_memory
llama_memory_seq_cp
llama_memory_seq_rm
llama_memory_seq_add
```

The goal is to evaluate two possible semantic-cache directions:

```text
valid shared-prefix branching
middle-segment deletion and position shifting
```

## Setup

Hardware/runtime:

```text
CPU: Intel Xeon E-2176M, 6 cores / 12 threads
RAM: 62 GiB
runtime: native llama.cpp C API
model: Qwen3-4B-Q4_K_M.gguf
threads: 10
context: 2048
batch: 1024
```

Probe segments:

```text
A prefix tokens: 14
noise tokens: 14
task tokens: 22
```

## Experiment 1: Valid Prefix Branching

Operation:

```text
compute A once
copy A KV to branch_noise
copy A KV to branch_task
continue branch_task with task
compare branch_task with scratch A+task
```

Result:

```text
branch_task_top: " yes"
scratch_task_top: " yes"
matches_scratch: true

mean_abs_diff: 0.00000000
rms_diff: 0.00000000
max_abs_diff: 0.00000000
cosine_similarity: 1.00000000
top_k_overlap: 5/5
```

Conclusion:

> Shared-prefix semantic branching is exactly valid in this probe.

This is the first strong runtime-level result. It means a semantic cache layer
can safely keep a stable prefix and branch into topic-specific continuations
using llama.cpp sequence KV copies.

## Experiment 2: Middle Removal

Operation:

```text
compute A + noise + task
copy sequence
remove noise KV range
shift later positions
recompute final task token
compare mutated A+task with scratch A+task
```

Result:

```text
seq_rm_returned: true
full_top: " "
mutated_top: " yes"
scratch_task_top: " yes"
mutated_matches_scratch: true

mutated_vs_scratch:
  mean_abs_diff: 0.45024302
  rms_diff: 0.53333262
  max_abs_diff: 2.48098564
  cosine_similarity: 0.97683771
  top_k_overlap: 4/5

full_vs_scratch:
  mean_abs_diff: 0.36498043
  rms_diff: 0.45721018
  max_abs_diff: 2.78014803
  cosine_similarity: 0.98162745
  top_k_overlap: 4/5
```

Conclusion:

> Middle removal is operationally possible, but not distribution-preserving.

The top token happened to match scratch `A+task`, but the logits are not equal.
So this cannot yet be treated as correct KV recomposition. It is an experimental
mutation, not a safe cache strategy.

## Research Conclusion

The promising direction is not arbitrary `A + noise + B -> A + B` reuse.

The promising direction is:

```text
semantic prefix tree:
  stable shared prefix
  topic branches copied from that prefix
  active task runs on the relevant branch
  irrelevant branches remain cached separately
```

This matches the original idea better than prompt routing alone while staying
within operations llama.cpp can do exactly.

## Next Step

Build a JSON-plan native runner:

```json
{
  "segments": [
    {"id": "root", "text": "..."},
    {"id": "qr", "text": "..."},
    {"id": "auth", "text": "..."}
  ],
  "ops": [
    {"op": "eval", "seq": 0, "segment": "root"},
    {"op": "copy", "from": 0, "to": 1},
    {"op": "copy", "from": 0, "to": 2},
    {"op": "eval", "seq": 1, "segment": "qr"},
    {"op": "eval", "seq": 2, "segment": "auth"}
  ]
}
```

Then the Python semantic router can generate cache plans instead of only
generating prompts.
