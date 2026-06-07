# Roadmap

## Milestone 1: Offline Context Router

- Maintain a complete JSONL event log.
- Convert events into reusable segments.
- Score segments by relevance, importance, recency, volatility, and token cost.
- Pack stable context before dynamic context.
- Report token savings and omitted segments.

Status: initial scaffold complete.

## Milestone 2: Local Runtime Adapter

- Add a benchmark mode that calls a local OpenAI-compatible server.
- Support vLLM and SGLang without changing scorer or packer code.
- Capture latency and runtime usage metadata when available.
- Keep offline tests independent from local model availability.

Status: initial OpenAI-compatible harness complete.

## Milestone 3: Prefix Cache Experiments

- Run the same task with raw history, routed context, and routed stable-prefix prompts.
- Measure time to first token across repeated turns.
- Track stable-prefix token estimates and runtime cache-hit signals.
- Identify when segment omission causes quality loss or reretrieval.

Status: local prompt-cache residency probe complete; fixture quality matrix
complete.

## Milestone 4: Native Shared-Prefix KV Branching

- Build a native llama.cpp runner that accepts JSON plans.
- Support sequence operations: `eval`, `copy`, `remove`, `shift`, `keep`,
  `compare`, and `generate`.
- Drive the native runner from a Python semantic router.
- Compare branch logits against scratch evaluation.
- Generate directly from a copied KV branch.

Status: complete for greedy generation on shared-prefix branches.

Validated claim:

```text
shared-prefix KV branching preserves greedy generation
```

Non-goal:

```text
arbitrary middle-token KV recomposition
```

## Milestone 5: LMCache / KV Reuse

- Add optional LMCache integration notes and benchmark recipes.
- Compare prefix-only reuse against cache-aware segment packing.
- Explore prewarming stable files or durable decisions.

Status: not started.

## Milestone 6: Modular Cache Research

- Investigate position-safe reusable prompt modules.
- Compare exact-prefix, radix, and non-prefix reuse strategies.
- Publish reproducible traces and benchmark reports.

Near-term next steps:

- replay a larger fake repo or anonymized real agent trace;
- benchmark `QR -> auth -> QR -> auth` switching before generation;
- report memory footprint as branch count grows;
- compare cold-cache, warm prompt-cache, and native branch-cache separately;
- improve patch-quality scoring beyond keyword checks.
