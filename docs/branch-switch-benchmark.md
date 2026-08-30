# Branch-Switch Benchmark

Measures the cost of switching between agent subtasks under three cache
strategies, using the native `semantic-kv-runner` and the fixture repo trace.

## Question

Does explicit multi-branch KV residency beat a single-slot prefix cache and
full re-prefill when an agent alternates between subtasks?

This is the core viability question for the project: exactness of shared-prefix
branching is already validated; this benchmark measures whether it is *useful*.

## Conditions

All three conditions run inside one native plan (same process, same model
weights in memory), tagged per op via `metadata.op_tags`:

- `branch_cache`: root evaluated once on seq 0, copied to one sequence per
  branch (`llama_memory_seq_cp`), branch context evaluated once per branch.
  A switch evaluates only the new turn tokens on the resident branch sequence.
- `prefix_slot`: emulates a one-slot prefix cache (single llama.cpp server
  slot). Root stays resident; switching branches removes the other branch's
  suffix (`llama_memory_seq_rm` per segment) and re-evaluates branch context
  plus all prior turns of the target branch.
- `scratch`: every switch re-evaluates root + branch + prior turns on a fresh
  sequence. No reuse.

## Workload

- Shared root = fixture-repo root context, padded with a stable-convention
  paragraph to a configurable size (`ACL_ROOT_PAD_WORDS`, default 1024 words)
  to model a realistic agent system prompt + project conventions.
- Two branches: `auth` and `qr` (fixture-repo semantic branches).
- Switch schedule alternates: `auth, qr, auth, qr, ...` (`ACL_SWITCHES`).
- Each switch appends a small turn segment (~30 tokens).

## Metrics

- Per-condition setup cost (ms) and per-switch cost (ms), summed from native
  op latencies; repeated `ACL_REPEATS` times (fresh process each repeat),
  reported as medians.
- Exactness bonus: the final same-branch states under `branch_cache` and
  `scratch` contain identical tokens at identical positions, so the plan
  compares pre-generation logits and greedily generates from both.

## Runner fix required by this benchmark

`generate` previously sampled its first token from the context-global logits
of the *last decode*, which may belong to another sequence. The runner now
takes the first greedy token from the sequence's captured logit snapshot
(requires `logits: true` on the final eval of that sequence). Without this fix
the second `generate` in a plan starts from another sequence's distribution.

## Usage

```bash
python scripts/run_branch_switch_benchmark.py
# knobs: ACL_MODEL_PATH ACL_REPEATS ACL_SWITCHES ACL_ROOT_PAD_WORDS ACL_CTX ACL_SEQS
```

Output: `benchmark-results/branch-switch-benchmark.json`.

## Interpretation notes

- `prefix_slot` is a *model* of single-slot prefix caching, not a measurement
  of llama-server itself; it retains root but pays branch + turn re-eval on
  every switch, which matches the best case of a longest-prefix-match slot
  cache with one slot.
- KV memory: with the default split KV cache, each sequence owns its cells, so
  `branch_cache` trades memory for switch latency. Sequence token residency is
  reported in the native `sequences` output.
