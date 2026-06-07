# Local Model Harness

The local model harness compares two prompt shapes against the same local model server:

- `raw`: the full trace history plus the current query;
- `routed`: the cache-aware packed prompt with stable context first and dynamic context last.

This is useful with small fast models because iteration is cheap. The first goal is not model quality. The first goal is to measure whether routing changes prompt size and latency in a repeatable way.

## Server

Any OpenAI-compatible `/v1/chat/completions` server should work.

Example with llama.cpp:

```bash
llama-server -m /path/to/SmolLM2-135M-Instruct-Q8_0.gguf --port 8080
```

## Run

```bash
acl model-harness \
  --trace examples/repo_debug_session.jsonl \
  --base-url http://127.0.0.1:8080 \
  --model SmolLM2-135M-Instruct-Q8_0 \
  --runs 5 \
  --warmup 1 \
  --max-output-tokens 32
```

The output includes:

- prompt token estimates;
- stable-prefix token estimates;
- latency per run;
- estimated output tokens per second;
- runtime `usage` metadata when the server returns it;
- short output previews for sanity checks.

## Offline Smoke

```bash
acl model-harness --trace examples/repo_debug_session.jsonl --echo
```

The echo mode verifies the harness plumbing without requiring a running model server.

## Qwen3 4B Long-Context Smoke

For an 8k-context Qwen3-4B local server, use:

```bash
python scripts/run_qwen3_4b_harness.py
```

Optional knobs:

```bash
ACL_BASE_URL=http://127.0.0.1:8080 \
ACL_MODEL=Qwen3-4B \
ACL_RUNS=2 \
ACL_WARMUP=1 \
ACL_NOISE_BLOCKS=12 \
ACL_MAX_PROMPT_TOKENS=2048 \
ACL_MAX_OUTPUT_TOKENS=24 \
ACL_TIMEOUT_SECONDS=300 \
python scripts/run_qwen3_4b_harness.py
```

This generates a temporary noisy trace and compares the raw prompt with the routed prompt.

## Industry Cache vs Forget

The explicit comparison script uses two modes:

- `industry_raw_cached`: full rolling history plus runtime prompt cache;
- `forget_routed_cached`: same event log preserved externally, but irrelevant segments omitted from the active prompt.

```bash
python scripts/run_forget_vs_industry.py
```

Offline smoke:

```bash
ACL_ECHO=1 python scripts/run_forget_vs_industry.py
```

## Long Coding Task Benchmark

This benchmark makes both modes ingest irrelevant QR, frontend, billing, and analytics files before solving an auth cookie bug. It reports token efficiency and simple quality checks.

```bash
python scripts/run_long_coding_task_benchmark.py
```

Offline smoke:

```bash
ACL_ECHO=1 python scripts/run_long_coding_task_benchmark.py
```

Run a matrix instead of a single case:

```bash
ACL_NOISE_FILES=4,8,12 \
ACL_OUTPUT_BUDGETS=48,96 \
ACL_WARMUP=1 \
ACL_RUNS=1 \
ACL_TIMEOUT_SECONDS=900 \
python scripts/run_long_coding_matrix.py
```
