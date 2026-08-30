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

## Milestone 6: Branch-Switch Usefulness

- Benchmark `auth -> qr -> auth -> qr` switching under three conditions:
  resident branch cache, single-slot prefix cache, full re-prefill.
- Report setup vs per-switch cost with repeated runs and medians.
- Keep an exactness check (compare + generate) inside the same plan.

Status: complete, and the result does not hold against the state of practice.
Qwen3-4B, 5 repeats
(`benchmark-research-conclusion/2026-07-23-branch-switch-benchmark.md`):
1.11 s/switch resident branches vs 7.0 s single-slot prefix cache (6.3x) vs
40.6 s full re-prefill (36.6x), exact greedy match preserved in 5/5 runs.
Includes a native runner fix so `generate` takes its first greedy token from
the sequence's own captured logits.

Retracted 2026-07-25
(`benchmark-research-conclusion/2026-07-25-llama-server-multislot-baseline.md`):
the `prefix_slot` baseline models a *single* server slot. Replaying the same
schedule through `llama-server -np 2` costs 1.20 s/switch, matching the custom
runner within noise, so the 6.3x speedup measured a configuration choice rather
than semantic branching. `llama-server` also already exposes `--kv-unified` for
cross-sequence KV sharing, so the one architectural advantage branching might
have held is stock functionality. This milestone contributes nothing that
llama.cpp does not already do.

## Milestone 7: Recomposition Repair (CacheBlend-style)

Informed by `docs/research-watch-2026-07.md`: naive A+B recomposition stays a
non-goal, but selective boundary recompute (CacheBlend), block-level reuse
(MEPIC), and RoPE re-alignment (KV Packet) make repaired recomposition a
measurable axis.

- Add a native op to re-evaluate a chosen fraction of boundary tokens after a
  non-prefix KV splice.
- Plot recompute ratio vs greedy exact-match rate on the fixture trace.
- Test shift-only RoPE re-alignment (`llama_memory_seq_add`) as the 0%-repair
  point of the same curve.

Status: 0%-repair probe complete
(`benchmark-research-conclusion/2026-07-23-kv-splice-shift-only-probe.md`):
shift-only splice keeps top-1 and 18/20 top-k but greedy diverges within ~21
characters — not exact, not catastrophic. Selective-recompute op not started.

Closed 2026-07-25 by a pre-registered kill criterion
(`benchmark-research-conclusion/2026-07-25-attention-does-not-predict-splice-damage.md`).
The open question was whether a cheap signal could tell a harness, before
deleting a span, whether deleting it is safe. Across 36 factorial cases no
attention-derived predictor reached |rho| = 0.5 against measured damage, and
none beat the trivial `distance_to_end` baseline (0.46). Conditioning shows why:
attention mass re-encodes position, and its correlation flips sign once position
is held constant. Splice damage is governed by how much context was computed
after the span, not by how much the model attended to it.

Recompute-based repair remains untested and would make splicing cheaper to
*repair*, not more *predictable* — it does not reopen this question.

## Milestone 8: Modular Cache Research

- Investigate position-safe reusable prompt modules.
- Compare exact-prefix, radix, and non-prefix reuse strategies.
- Publish reproducible traces and benchmark reports.

Near-term next steps:

- replay a larger fake repo or anonymized real agent trace;
- benchmark `QR -> auth -> QR -> auth` switching before generation;
- report memory footprint as branch count grows;
- compare cold-cache, warm prompt-cache, and native branch-cache separately;
- improve patch-quality scoring beyond keyword checks.
