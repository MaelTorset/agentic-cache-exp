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

## Milestone 4: LMCache / KV Reuse

- Add optional LMCache integration notes and benchmark recipes.
- Compare prefix-only reuse against cache-aware segment packing.
- Explore prewarming stable files or durable decisions.

## Milestone 5: Modular Cache Research

- Investigate position-safe reusable prompt modules.
- Compare exact-prefix, radix, and non-prefix reuse strategies.
- Publish reproducible traces and benchmark reports.
