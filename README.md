# Agentic Cache Lab

Agentic Cache Lab is an experimental Python + native llama.cpp project for
cache-aware context routing in long-running AI agents.

The core idea:

> Agents should not carry their entire raw conversation forever. They should keep a complete external event log, route only the currently useful context into the active prompt, and structure stable context so local inference engines can reuse prefix/KV cache more effectively.

This project started as an offline prompt-routing harness. It now also includes
a native llama.cpp JSON-plan runner that can copy sequence KV state, create
semantic shared-prefix branches, compare them against scratch evaluation, and
generate text directly from a copied KV branch.

It does **not** claim arbitrary KV recomposition. In particular, the hard form:

```text
A + noise + B -> reuse cached B directly as A + B
```

is still unsafe for a normal causal transformer unless `B` was computed under
compatible attention and positions. The validated path is semantic
shared-prefix branching.

## Current Research Claim

The strongest result so far:

> Semantic shared-prefix KV branching preserves greedy generation exactly in
> the local llama.cpp runtime.

Validated on the fixture repo with Qwen3-4B-Q4_K_M:

```text
root -> copy root KV -> auth branch -> generate
matches
root + auth branch from scratch -> generate
```

Measured result:

```text
pre-generation logits:
  mean_abs_diff: 0
  cosine_similarity: 1
  top_k_overlap: 5/5

generation_text_match: true
```

On the usefulness side the honest answer is negative, and two measurements say
so.

**Branch switching is not faster than stock llama.cpp.** The branch-switch
benchmark (2026-07-23) measured 1.11 s/switch for resident KV branches against
7.00 s for a *single* prefix-cache slot. But a real deployment is not limited to
one slot: replaying the identical schedule through `llama-server -np 2` costs
1.20 s/switch (2026-07-25, 5-run medians), matching the custom runner to within
run-to-run noise. The single-slot figure measured a configuration choice, not a
property of semantic branching. See
`benchmark-research-conclusion/2026-07-25-llama-server-multislot-baseline.md`.

**Selective forgetting is not steerable.** Removing a span from mid-context and
RoPE-shifting the survivors is cheap but inexact, and the damage varies 27x
depending on what is removed. Across 36 factorial cases, no attention-derived
signal predicts that damage (best |rho| = 0.37) and none beats the trivial
baseline of "how far the span sits from the end of the context" (rho = 0.46).
Attention mass turns out to re-encode position rather than dependence. Only
19.4% of splices reproduced greedy output exactly. See
`benchmark-research-conclusion/2026-07-25-attention-does-not-predict-splice-damage.md`.

The quality result is more conservative: routing reduced active prompt tokens
by about 62% on the fixture benchmark, but Qwen3-4B did not solve the bug better
than the full noisy prompt. Cache correctness is currently stronger than
agentic-task quality.

## What It Does

- Stores agent events such as user messages, file reads, commands, errors, and decisions.
- Converts events into labeled context segments.
- Scores segments by relevance, importance, recency, volatility, and token cost.
- Packs prompts into a stable-prefix / dynamic-suffix layout.
- Compares raw-history prompts against routed prompts.
- Compares naive Forget, critical-context Forget, and an oracle relevant-only baseline.
- Runs raw vs routed prompts against a local OpenAI-compatible model server.
- Includes an offline echo mode for CI and harness plumbing.
- Builds semantic branch plans from Python and executes them in native llama.cpp.
- Copies shared-prefix KV state into semantic branches with `llama_memory_seq_cp`.
- Compares branch logits against scratch evaluation.
- Generates text directly from copied KV branches with greedy decoding.
- Benchmarks branch switching (`auth -> qr -> auth -> qr`) under resident
  branch cache, single-slot prefix cache, and full re-prefill conditions
  (`scripts/run_branch_switch_benchmark.py`, `docs/branch-switch-benchmark.md`).

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

Or execute a JSON KV plan:

```bash
./build/native-probes/semantic-kv-runner \
  -m /data/llama/models/Qwen3-4B-Q4_K_M.gguf \
  --plan examples/native/prefix_branch_plan.json \
  --threads 10 \
  --ctx 2048 \
  --seqs 8
```

To generate a semantic branch plan in Python and run it through the native KV
runner:

```bash
python scripts/build_semantic_kv_plan.py
python scripts/run_semantic_branch_benchmark.py
```

To test the same path on a small fake TypeScript repo fixture:

```bash
python scripts/run_fixture_repo_branch_benchmark.py
```

To measure bug-fix answer quality over several model runs and attach the native
KV branch metrics:

```bash
ACL_BASE_URL=http://127.0.0.1:8082 ACL_RUNS=5 python scripts/run_fixture_repo_quality_matrix.py
```

To generate directly from a native KV branch and compare against scratch:

```bash
python scripts/run_fixture_repo_native_generation.py
```

See `docs/native-kv-probe.md`.

## What The Latest Benchmarks Show

See `benchmark-research-conclusion/` for dated reports. The most important
current files are:

```text
2026-06-07-native-kv-branch-generation.md
2026-06-07-fixture-repo-quality-matrix.md
2026-06-07-fixture-repo-branch-benchmark.md
```

Current summary:

```text
Native KV exactness:
  shared-prefix branch logits match scratch exactly
  greedy generation from branch matches scratch exactly

Token efficiency:
  routed prompt uses about 62% fewer runtime prompt tokens on the fixture task

Quality:
  no quality win yet on Qwen3-4B for the fake auth-cookie bug

Latency:
  KV copy is effectively free compared with root prefill
  end-to-end answer latency is often dominated by generation and prompt-cache state
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
  bug_resolution_benchmark.py  # fixture bug-quality scoring and native generation plans
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
  fixtures/shopbug-repo/
docs/
  architecture.md
  local-model-harness.md
  native-kv-probe.md
  roadmap.md
native/
  semantic_kv_probe.cpp
  semantic_kv_runner.cpp
tests/
```

## Roadmap

The next technical milestone is to move from controlled fixture proof to a more
credible agent trace:

- replay a real anonymized coding-agent session or a larger open-source fixture;
- compare cold-cache, warm prompt-cache, and native branch-cache modes;
- measure cache memory footprint as branch count grows;
- add branch switching before generation: `QR -> auth -> QR -> auth`;
- improve the task oracle so quality is judged on patch correctness, not only
  keyword presence;
- test a stronger local model when available.

See [docs/architecture.md](docs/architecture.md) and [docs/roadmap.md](docs/roadmap.md).

## Status

Research prototype. The APIs and scoring heuristics are intentionally small and expected to change.
