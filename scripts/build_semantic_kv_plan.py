from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.semantic_kv_planner import build_semantic_kv_plan
from agentic_cache_lab.synthetic import build_long_coding_task_events


def main() -> None:
    noise_files = int(os.environ.get("ACL_NOISE_FILES", "12"))
    branch_labels = parse_labels(os.environ.get("ACL_BRANCH_LABELS", "auth,qr,billing"))
    measured_labels = parse_labels(os.environ.get("ACL_MEASURED_LABELS", "auth,qr"))
    top_k = int(os.environ.get("ACL_TOP_K", "5"))
    output = Path(os.environ.get("ACL_OUTPUT", str(ROOT / "benchmark-results" / "semantic-branch-plan.json")))

    plan = build_semantic_kv_plan(
        build_long_coding_task_events(noise_files=noise_files),
        branch_labels=branch_labels,
        measured_labels=measured_labels,
        top_k=top_k,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "segments": len(plan["segments"]), "ops": len(plan["ops"])}, indent=2))


def parse_labels(raw: str) -> tuple[str, ...]:
    labels = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not labels:
        raise ValueError("expected at least one label")
    return labels


if __name__ == "__main__":
    main()
