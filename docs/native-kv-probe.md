# Native KV Probe

Date: 2026-06-06

## Why This Exists

The HTTP server probes can show prompt-cache residency, but they cannot
modulate the KV cache directly.

The native probe moves the project into the real research layer:

```text
llama.cpp C API
-> llama_get_memory(ctx)
-> llama_memory_seq_cp
-> llama_memory_seq_rm
-> llama_memory_seq_add
-> llama_memory_seq_keep
```

This is where a semantic cache layer can start experimenting with actual
sequence-level KV operations.

## Build

The probe expects a local llama.cpp checkout and build. Defaults match this
machine:

```text
LLAMA_CPP_ROOT=/data/llama/llama.cpp
LLAMA_CPP_BUILD=/data/llama/llama.cpp/build
```

Build:

```bash
cmake -S native -B build/native-probes
cmake --build build/native-probes -j 2
```

## Run

```bash
./build/native-probes/semantic-kv-probe \
  -m /data/llama/models/Qwen3-4B-Q4_K_M.gguf \
  --threads 10 \
  --ctx 2048 \
  --batch 1024
```

## Experiments In The Probe

### Valid Prefix Branching

This tests the practical semantic-cache path:

```text
A
copy A KV to branch_noise
copy A KV to branch_task
continue branch_noise with noise
continue branch_task with task
compare branch_task with scratch A+task
```

If branch task output matches scratch task output, then shared-prefix branching
is valid for this runtime.

This is the near-term path for a semantic cache layer:

```text
stable reusable prefix
-> many semantic branches
-> keep hot branches by topic
```

Current Qwen3-4B result:

```text
branch_vs_scratch_logits:
  mean_abs_diff: 0.00000000
  max_abs_diff: 0.00000000
  cosine_similarity: 1.00000000
  top_k_overlap: 5/5
```

So prefix branching is exact in the current probe.

### Middle Removal Experiment

This intentionally tests the hard/unsafe idea:

```text
compute A + noise + task
copy sequence
remove noise KV range
shift later positions
compare mutated A+task with scratch A+task
```

This operation is experimentally possible with llama.cpp sequence APIs, but it
is not expected to be generally correct for a causal transformer.

The correctness test is whether the mutated branch matches scratch `A+task`.

Current Qwen3-4B result:

```text
mutated_matches_scratch top token: true
mutated_vs_scratch_logits:
  mean_abs_diff: 0.45024302
  max_abs_diff: 2.48098564
  cosine_similarity: 0.97683771
  top_k_overlap: 4/5
```

So middle deletion is operationally possible, but not distribution-preserving.
It should be treated as an unsafe experimental mutation, not a valid cache
strategy yet.

## Research Boundary

What this probe enables:

```text
direct sequence-level KV experiments
valid shared-prefix branching
cache branch retention
middle-range deletion experiments
position-shift experiments
```

What it does not magically solve:

```text
semantic middle-token removal with guaranteed correctness
arbitrary recomposition of cached blocks
provider-level KV surgery for closed APIs
```

## JSON Plan Runner

The fixed probe is kept for reproducible historical benchmarking. The native
runner accepts a JSON plan:

```json
{
  "config": {"top_k": 5, "suppress_logs": true},
  "segments": [
    {"id": "A", "text": "..."},
    {"id": "noise", "text": "..."},
    {"id": "task", "text": "..."}
  ],
  "ops": [
    {"op": "eval", "seq": 0, "segment": "A"},
    {"op": "copy", "from": 0, "to": 1},
    {"op": "eval", "seq": 1, "segment": "task"}
  ]
}
```

Run:

```bash
./build/native-probes/semantic-kv-runner \
  -m /data/llama/models/Qwen3-4B-Q4_K_M.gguf \
  --plan examples/native/prefix_branch_plan.json \
  --threads 10 \
  --ctx 2048 \
  --batch 1024
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

Example plans:

```text
examples/native/prefix_branch_plan.json
examples/native/middle_removal_plan.json
```

This lets the Python semantic router produce cache-operation plans instead of
only producing prompts.
