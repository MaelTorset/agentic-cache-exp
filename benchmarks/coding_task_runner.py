from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.cli import run_benchmark


def main() -> None:
    trace = ROOT / "examples" / "repo_debug_session.jsonl"
    result = run_benchmark(
        trace,
        query="We are now working only on authentication cookies. Drop unrelated QR scanner details.",
        objective="Measure whether context routing removes unrelated work while preserving useful decisions.",
        max_prompt_tokens=320,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
