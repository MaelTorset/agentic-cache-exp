# Agentic Cache Lab

Agentic Cache Lab is an experimental Python project for cache-aware context routing in long-running AI agents.

The core idea:

> Agents should not carry their entire raw conversation forever. They should keep a complete external event log, route only the currently useful context into the active prompt, and structure stable context so local inference engines can reuse prefix/KV cache more effectively.

This first version does **not** require vLLM, SGLang, LMCache, or any paid API. It is a small offline benchmark harness that proves the context-routing layer before deeper KV-cache integration.

## What It Does

- Stores agent events such as user messages, file reads, commands, errors, and decisions.
- Converts events into labeled context segments.
- Scores segments by relevance, importance, recency, volatility, and token cost.
- Packs prompts into a stable-prefix / dynamic-suffix layout.
- Compares raw-history prompts against routed prompts.
- Includes an OpenAI-compatible local client stub for future vLLM/SGLang integration.

## Quick Start

```bash
git clone <your-repo-url>
cd agentic-cache-lab

python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

acl benchmark --trace examples/repo_debug_session.jsonl
```

Without installing the package:

```bash
PYTHONPATH=src python -m agentic_cache_lab.cli benchmark --trace examples/repo_debug_session.jsonl --json
```

Run tests:

```bash
python -m unittest discover -s tests
```

Run the smoke benchmark script:

```bash
python scripts/run_benchmark.py
```

## Example Output

The included fixture simulates a coding session where QR-code work and authentication work were mixed together. The router keeps durable authentication context and omits unrelated QR files from the active prompt.

```json
{
  "raw_tokens_estimate": 276,
  "routed_tokens_estimate": 238,
  "saved_tokens_estimate": 38,
  "saved_ratio": 0.1377
}
```

Token counts are deterministic estimates for offline experiments. They are not model-tokenizer exact.

## Project Layout

```text
src/agentic_cache_lab/
  cli.py             # CLI entrypoint
  event_log.py       # JSONL trace loading
  models.py          # shared dataclasses and token estimates
  segment_store.py   # segment creation, hashing, heuristic labels
  scorer.py          # relevance / reuse / volatility scoring
  packer.py          # cache-aware prompt packing
  llm_client.py      # offline echo client and local OpenAI-compatible client
benchmarks/
  coding_task_runner.py
examples/
  repo_debug_session.jsonl
docs/
  architecture.md
  roadmap.md
tests/
```

## Roadmap

The next technical milestone is to connect the prompt packer to a local vLLM or SGLang server with prefix caching enabled, then measure:

- time to first token;
- prompt/prefix token estimates;
- cache hit signals exposed by the serving runtime;
- quality regressions when context is omitted;
- reretrieval events when the packer drops something too aggressively.

See [docs/architecture.md](docs/architecture.md) and [docs/roadmap.md](docs/roadmap.md).

## Status

Research prototype. The APIs and scoring heuristics are intentionally small and expected to change.
