from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.coding_benchmark import run_long_coding_task_benchmark


def main() -> None:
    base_url = os.environ.get("ACL_BASE_URL", "http://127.0.0.1:8081")
    model = os.environ.get("ACL_MODEL", "Qwen3-4B")
    noise_files = int(os.environ.get("ACL_NOISE_FILES", "12"))
    max_prompt_tokens = int(os.environ.get("ACL_MAX_PROMPT_TOKENS", "2048"))
    runs = int(os.environ.get("ACL_RUNS", "1"))
    warmup = int(os.environ.get("ACL_WARMUP", "1"))
    max_output_tokens = int(os.environ.get("ACL_MAX_OUTPUT_TOKENS", "96"))
    timeout_seconds = float(os.environ.get("ACL_TIMEOUT_SECONDS", "300"))
    echo = os.environ.get("ACL_ECHO", "0") == "1"

    with tempfile.TemporaryDirectory() as directory:
        trace_path = Path(directory) / "long_coding_task.jsonl"
        result = run_long_coding_task_benchmark(
            trace_path=trace_path,
            base_url=base_url,
            model=model,
            api_key="local",
            noise_files=noise_files,
            max_prompt_tokens=max_prompt_tokens,
            runs=runs,
            warmup=warmup,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            echo=echo,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
