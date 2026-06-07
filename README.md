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
- Compares naive Forget, critical-context Forget, and an oracle relevant-only baseline.
- Runs raw vs routed prompts against a local OpenAI-compatible model server.
- Includes an offline echo mode for CI and harness plumbing.

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

Run the local-model harness against a llama.cpp/vLLM/SGLang OpenAI-compatible server:

```bash
acl model-harness \
  --trace examples/repo_debug_session.jsonl \
  --base-url http://127.0.0.1:8080 \
  --model SmolLM2-135M-Instruct-Q8_0 \
  --runs 5 \
  --warmup 1 \
  --max-output-tokens 32
```

For llama.cpp, start a compatible server separately:

```bash
llama-server -m /path/to/SmolLM2-135M-Instruct-Q8_0.gguf --port 8080
```

To test the harness without a model server:

```bash
acl model-harness --trace examples/repo_debug_session.jsonl --echo
```

For a heavier synthetic trace suited to an 8k-context local model:

```bash
python scripts/run_qwen3_4b_harness.py
```

You can tune it with environment variables:

```bash
ACL_RUNS=2 ACL_NOISE_BLOCKS=12 ACL_MAX_PROMPT_TOKENS=2048 python scripts/run_qwen3_4b_harness.py
```

To compare the industry-style raw cached prompt against the Forget routing baseline:

```bash
python scripts/run_forget_vs_industry.py
```

Offline smoke:

```bash
ACL_ECHO=1 python scripts/run_forget_vs_industry.py
```

To run the long coding-task benchmark with quality checks:

```bash
python scripts/run_long_coding_task_benchmark.py
```

Offline smoke:

```bash
ACL_ECHO=1 python scripts/run_long_coding_task_benchmark.py
```

For a more credible multi-case matrix:

```bash
ACL_ECHO=1 python scripts/run_long_coding_matrix.py
```

To test whether a local runtime keeps prompt shapes cached while alternating
between full-history and forgotten prompts:

```bash
python scripts/run_cache_residency_probe.py
```

The current research benchmark compares:

```text
industry_raw_cached
forget_routed_cached_v0
forget_critical_context_v1
oracle_relevant_only
```

See `benchmark-research-conclusion/` for dated results and caveats. The best
current signal is that `critical_context_v1` improves quality over naive Forget
while keeping large prompt-token reductions on the synthetic coding task.

See `docs/cache-runtime-requirements.md` for the current boundary between
active prompt forgetting, whole-prompt cache residency, and true modular KV-cache
recomposition.

For direct llama.cpp KV-cache experiments, build the native probe:

```bash
cmake -S native -B build/native-probes
cmake --build build/native-probes -j 2
```

Then run:

```bash
./build/native-probes/semantic-kv-probe \
  -m /data/llama/models/Qwen3-4B-Q4_K_M.gguf \
  --threads 10 \
  --ctx 2048
```

See `docs/native-kv-probe.md`.

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
  synthetic.py       # deterministic long-context fixture generation
benchmarks/
  coding_task_runner.py
examples/
  repo_debug_session.jsonl
docs/
  architecture.md
  local-model-harness.md
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
