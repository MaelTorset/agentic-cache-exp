# Benchmark Research Conclusion: Forget vs Industry Cache

Date: 2026-06-06

## Research Question

Can an agentic "Forget" layer be useful even when the industry baseline already has prompt caching?

The comparison we ran:

- `industry_raw_cached`: full rolling history remains in the active prompt, with llama.cpp prompt cache enabled.
- `forget_routed_cached`: the same full event log is preserved externally, but irrelevant segments are omitted from the active prompt, with llama.cpp prompt cache still enabled.

This is the right first comparison because it does not compare against a weak no-cache baseline. It compares active-context forgetting against a realistic prompt-cache baseline.

## Experimental Setup

Model:

```text
Qwen3-4B-Q4_K_M.gguf
```

Runtime:

```text
llama.cpp llama-server
context size: 8192
threads: 10
prompt cache: enabled
host: 127.0.0.1
port: 8081
```

Hardware:

```text
CPU: Intel Xeon E-2176M, 6 cores / 12 threads
RAM: 62 GiB
GPU: NVIDIA Quadro P2000 Mobile
serving mode used here: llama.cpp local GGUF, CPU-focused run
```

This is student research on local hardware, not a datacenter GPU benchmark. The project intentionally uses small GGUF models such as Qwen3-4B because they are practical on this machine and allow repeated experiments without cloud cost. The purpose is to test the memory/cache architecture, not to claim that a 4B model represents frontier-model quality.

Benchmark script:

```text
scripts/run_forget_vs_industry.py
```

Synthetic trace:

```text
noise blocks: 4
active task: authentication cookie bug
irrelevant context: QR scanner and frontend noise files
output tokens: 8
warmup: 1
runs: 1
```

## Measured Data

Warm-cache Qwen3-4B result:

```text
industry_raw_cached:
  runtime prompt tokens: 1502
  cached prompt tokens: 1501
  cache hit ratio: 0.9993
  latency: 2033.68 ms

forget_routed_cached:
  runtime prompt tokens: 627
  cached prompt tokens: 626
  cache hit ratio: 0.9984
  latency: 1851.40 ms
```

Derived result:

```text
runtime prompt token reduction: 58.26%
latency reduction: 8.96%
```

Offline routing estimate from the same benchmark:

```text
raw token estimate: 1462
forget token estimate: 558
estimated token reduction: 61.83%
stable prefix estimate: 450
```

## Conclusion

The Forget approach is not mainly a replacement for prompt caching. It is a layer above prompt caching.

Prompt caching answers this question:

```text
If I send the same prompt again, can the runtime avoid recomputing most of it?
```

Forget answers a different question:

```text
Should this context still be in the active prompt at all?
```

The benchmark shows that even when the industry baseline is almost fully cached, it still carries a much larger active prompt. In this run, the raw cached baseline carried 1502 runtime prompt tokens, while Forget carried 627. Both had near-perfect cache hit ratios, so the comparison is fair for warm cache.

The latency gain was modest because warm-cache prompt evaluation is already cheap. The stronger result is context-budget reduction: Forget reduced active prompt size by about 58% while preserving the external event log.

This matters because useful agent sessions are not one repeated prompt. They involve:

- topic switching;
- exploratory reads of useless files;
- tool results that become stale;
- long debugging histories;
- cache pressure;
- branch changes;
- repeated retrieval of prior context;
- active tasks that only need a small slice of the past.

In those cases, carrying every cached segment still wastes context budget and can pollute the model's attention. A useful Forget system should keep all past data externally but only expose the currently relevant working set.

## Current Limitation

The V0 Forget decider is heuristic:

```text
forget_decider: segment_scorer_v0
```

It forgot several irrelevant sources, but it still kept one frontend noise file in the tested trace. That means the current system is useful as a prototype, not yet as a reliable agent memory manager.

Also, this benchmark is still synthetic. It proves the measurement harness and the direction of the effect, but it does not yet prove production usefulness.

## What Would Make This Actually Useful?

A useful system should not be just "shorter prompts." It needs to become an active memory controller for agents.

Required capabilities:

1. Complete external event log

   The system must keep everything outside the active prompt:

   ```text
   files read, commands run, errors, decisions, diffs, user goals, tool results
   ```

2. Active context selection

   At every turn, the system decides what belongs in the active prompt:

   ```text
   durable decisions
   current task files
   recent errors
   relevant prior observations
   ```

3. Explicit forgetting

   Forgetting must mean:

   ```text
   remove from active prompt, but keep retrievable externally
   ```

   It must not mean deleting history.

4. Cache-aware packing

   Selected context should be ordered to help runtime/provider caching:

   ```text
   stable prefix first
   volatile suffix last
   repeated durable context unchanged across turns
   ```

5. Recovery path

   If the model needs forgotten context, it must be able to ask for retrieval:

   ```text
   "retrieve QR scanner context"
   "reload auth/session.ts"
   "show omitted segment sources"
   ```

6. Quality checks

   The benchmark must track whether forgetting causes wrong answers or missing-context failures.

## Next Steps

### Step 1: Improve the Forget Decider

Replace the current heuristic-only scorer with a stricter policy:

```text
hard negative intent:
  "forget QR", "omit frontend", "ignore billing"

task labels:
  auth, qr, frontend, backend, billing, cache

source-level penalties:
  if source topic is explicitly excluded, omit unless it contains a durable decision
```

Expected outcome:

```text
all irrelevant synthetic noise files are omitted
auth files, auth errors, and durable decisions are retained
```

### Step 2: Add Multi-Turn Topic Switching

Create a benchmark trace:

```text
turns 1-5: QR work
turns 6-10: auth work
turns 11-15: frontend work
turn 16: return to auth
turn 17: ask for QR again
```

Compare:

```text
industry_raw_cached
rolling_window
forget_routed_cached
forget_with_retrieval
```

This is the first benchmark that can show whether Forget is useful in a realistic agent workflow.

### Step 3: Add Quality Assertions

The benchmark should not only measure tokens and latency. It should verify task success.

Examples:

```text
auth task answer must mention SameSite=None without Secure on localhost
QR task answer must not hallucinate auth cookie changes
forgotten context retrieval must recover the correct omitted source
```

### Step 4: Add Cold vs Warm Cache Modes

Separate:

```text
cold cache latency
warm cache latency
cache-hit token count
active prompt token count
```

The key claim should be:

```text
Forget reduces active context size even when prompt cache is already warm.
```

Latency is secondary unless cache pressure or cold starts are involved.

### Step 5: Make the Forget Decision Inspectable

Every omitted segment should have a reason:

```json
{
  "source": "frontend/src/features/qr/noise_0.tsx",
  "decision": "forgotten",
  "reason": "query explicitly excludes QR scanner context",
  "recoverable": true
}
```

This is necessary for debugging and for trust.

## Research Position

The strongest current framing is:

> Forget is a cache-aware active-context controller for agents. It does not compete with prompt caching; it makes prompt caching and long-running agent sessions more useful by deciding what should remain in the active prompt.

The current evidence supports this narrower claim:

> In a synthetic noisy trace, Forget reduced active prompt tokens by about 58% compared with a fully cached raw-history baseline, while preserving near-perfect runtime cache hit ratio.

The current evidence does not yet prove:

```text
production reliability
better answer quality
lower cost on hosted provider APIs
superiority over compact summaries or retrieval memory
```

Those need the next benchmark matrix.
