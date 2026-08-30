# Research Watch — July 2026 (KV reuse, agentic context)

Compiled from a web survey on 2026-07-23. Mechanisms were cross-checked across
sources, but exact figures (17x, 2.1x, 15%) should be re-verified against the
papers before citing.

## Key takeaway for this project

The current non-goal ("arbitrary mid-token KV recomposition is unsafe") holds
only for *naive concatenation*. Three families of 2025-2026 work make
recomposition approximately or fully recoverable, which turns the non-goal
into a measurable research axis:

- **CacheBlend** (arxiv.org/html/2605.24022): fuse precomputed KV chunks and
  selectively recompute ~15% of boundary/high-divergence tokens (identified at
  layer 1, reused across layers). Training-free — directly implementable in
  the native runner.
- **EPIC / MEPIC / HYPIC** (arxiv.org/abs/2410.15332, /abs/2512.16822,
  /pdf/2607.01299): position-independent chunking (KVSplit) plus sparse
  recompute (AttnLink); MEPIC moves recompute to block granularity, close to
  the llama.cpp memory model.
- **KV Packet** (arxiv.org/pdf/2604.13226): recompute-free reuse via RoPE
  re-adaptation of cached K at arbitrary positions — llama.cpp already has KV
  shifting (`llama_memory_seq_add`), so the primitive exists here.
- **C²KV** (arxiv.org/html/2607.17715): trained composable memory tokens;
  needs fine-tuning, out of scope, but a useful "training-based ceiling"
  baseline. Notes CacheBlend instability across models/datasets.

## Ecosystem notes

- **LMCache** (github.com/LMCache/LMCache): pluggable KV engine for
  vLLM/SGLang; CPU/NVMe/Redis/S3 backends, P2P sharing in prod since 2026-01,
  integrates CacheBlend. GPU-serving comparison baseline for Milestone on
  LMCache.
- **KV compression in production**: FP8/quantized KV, outlier-aware per-channel
  quantization in SGLang/vLLM; llama.cpp already exposes
  `--cache-type-k/v q8_0` — footprint benchmarks should add an fp16-vs-q8_0
  axis at constant greedy quality.
- **llama.cpp**: 2026 rewrite reorganized KV contiguous per attention head;
  paged KV + scheduler design in progress
  (github.com/ggml-org/llama.cpp/discussions/21961) — prerequisite for
  MEPIC-style block reuse. Server slot save/restore
  (`/slots/{id}/save|restore`) persists KV to disk; hard-fails on
  flash-attention mismatch.
- **Agentic context management**: Leyline (arxiv.org/pdf/2606.01065) —
  per-segment keep/drop/reuse KV directives, isomorphic to this project's
  JSON plans; TokenPilot (arxiv.org/pdf/2606.17016) measures cache-hit vs
  task-success on SWE-bench/TerminalBench; TokenCake, SideQuest.

## Ranked next experiments (novel + measurable on consumer hardware)

1. **Recompute-ratio vs greedy-exact-match curve (native CacheBlend-style).**
   Implement selective boundary-token recompute in the runner; plot % tokens
   recomputed vs exact-match rate for A+B recomposition. Turns the non-goal
   into a quantified result. Highest ROI.
2. **Branch-switch latency: in-memory `llama_memory_seq_cp` vs disk slot
   save/restore.** Extends the existing branch-switch benchmark with the
   disk-persistence condition (multi-GB state files, flash-attention
   constraint).
3. **Memory footprint vs branch count, fp16 vs quantized KV** using
   `--cache-type-k/v`.
4. **RoPE positional re-alignment of cached K (KV-Packet-lite)** via
   `llama_memory_seq_add`: does shift-only non-prefix reuse work, and where
   does quality break vs recompute-based repair? Informative either way.
5. **Position the JSON plans as Leyline-style directives** and measure
   cache-hit vs task-success on a standard agent benchmark subset.
