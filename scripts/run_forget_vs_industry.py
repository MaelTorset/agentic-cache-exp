from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.event_log import dump_jsonl
from agentic_cache_lab.forget_benchmark import run_forget_vs_industry
from agentic_cache_lab.synthetic import build_long_context_events


def main() -> None:
    base_url = os.environ.get("ACL_BASE_URL", "http://127.0.0.1:8081")
    model = os.environ.get("ACL_MODEL", "Qwen3-4B")
    runs = int(os.environ.get("ACL_RUNS", "1"))
    warmup = int(os.environ.get("ACL_WARMUP", "1"))
    noise_blocks = int(os.environ.get("ACL_NOISE_BLOCKS", "4"))
    max_prompt_tokens = int(os.environ.get("ACL_MAX_PROMPT_TOKENS", "1024"))
    max_output_tokens = int(os.environ.get("ACL_MAX_OUTPUT_TOKENS", "8"))
    timeout_seconds = float(os.environ.get("ACL_TIMEOUT_SECONDS", "300"))
    echo = os.environ.get("ACL_ECHO", "0") == "1"

    with tempfile.TemporaryDirectory() as directory:
        trace_path = Path(directory) / "forget_vs_industry.jsonl"
        dump_jsonl(build_long_context_events(noise_blocks=noise_blocks), trace_path)
        result = run_forget_vs_industry(
            trace_path=trace_path,
            query="Focus only on the authentication cookie bug. Forget QR scanner and frontend noise for this turn.",
            objective="Compare industry raw-history prompt caching with explicit active-context forgetting.",
            max_prompt_tokens=max_prompt_tokens,
            base_url=base_url,
            model=model,
            api_key="local",
            runs=runs,
            warmup=warmup,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            echo=echo,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
