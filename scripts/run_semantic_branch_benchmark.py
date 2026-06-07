from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.semantic_kv_planner import build_semantic_kv_plan
from agentic_cache_lab.synthetic import build_long_coding_task_events


def main() -> None:
    model = os.environ.get("ACL_MODEL_PATH", "/data/llama/models/Qwen3-4B-Q4_K_M.gguf")
    runner = Path(os.environ.get("ACL_NATIVE_RUNNER", str(ROOT / "build" / "native-probes" / "semantic-kv-runner")))
    noise_files = int(os.environ.get("ACL_NOISE_FILES", "12"))
    branch_labels = parse_labels(os.environ.get("ACL_BRANCH_LABELS", "auth,qr,billing"))
    measured_labels = parse_labels(os.environ.get("ACL_MEASURED_LABELS", "auth,qr"))
    threads = os.environ.get("ACL_THREADS", "10")
    ctx = os.environ.get("ACL_CTX", "2048")
    batch = os.environ.get("ACL_BATCH", "1024")
    seqs = os.environ.get("ACL_SEQS", "8")
    top_k = int(os.environ.get("ACL_TOP_K", "5"))
    output = os.environ.get("ACL_OUTPUT")

    plan = build_semantic_kv_plan(
        build_long_coding_task_events(noise_files=noise_files),
        branch_labels=branch_labels,
        measured_labels=measured_labels,
        top_k=top_k,
    )

    with tempfile.TemporaryDirectory() as directory:
        plan_path = Path(directory) / "semantic_branch_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = run_native_runner(runner, model, plan_path, threads=threads, ctx=ctx, batch=batch, seqs=seqs)

    result["planner_summary"] = summarize_plan(plan, result)
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, sort_keys=True))


def run_native_runner(
    runner: Path,
    model: str,
    plan_path: Path,
    threads: str,
    ctx: str,
    batch: str,
    seqs: str,
) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(runner),
            "-m",
            model,
            "--plan",
            str(plan_path),
            "--threads",
            threads,
            "--ctx",
            ctx,
            "--batch",
            batch,
            "--seqs",
            seqs,
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"semantic-kv-runner failed: {completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout)


def summarize_plan(plan: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    metadata = plan.get("metadata", {})
    branch_sequences = {str(value): key for key, value in dict(metadata.get("branch_sequences", {})).items()}
    scratch_sequences = {str(value): key for key, value in dict(metadata.get("scratch_sequences", {})).items()}
    sequences = dict(result.get("sequences", {}))
    segments = dict(result.get("segments", {}))
    root_tokens = int(dict(segments.get("root", {})).get("tokens", 0))
    avoided = root_tokens * len(branch_sequences)

    return {
        "branch_labels": list(dict(metadata.get("branch_sequences", {})).keys()),
        "measured_labels": list(dict(metadata.get("scratch_sequences", {})).keys()),
        "root_tokens": root_tokens,
        "estimated_prefill_tokens_avoided": avoided,
        "branch_sequence_count": len(branch_sequences),
        "scratch_sequence_count": len(scratch_sequences),
        "sequence_count": len(sequences),
    }


def parse_labels(raw: str) -> tuple[str, ...]:
    labels = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not labels:
        raise ValueError("expected at least one label")
    return labels


if __name__ == "__main__":
    main()
