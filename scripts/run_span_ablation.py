"""Measure what deleting a mid-context span actually costs, case by case.

Reads a JSON array of cases (see ``src/agentic_cache_lab/span_ablation.py`` for
the four-segment shape) and, for each one, runs the native runner twice:

1. an attention pass over the unspliced context, to read the mass the span
   received from every surviving query;
2. an ablation pass comparing a clean reference against the spliced state.

Both passes are separate runner invocations so each case gets a fresh KV cache
and cannot be perturbed by its neighbours.

Usage::

    ACL_CASES=path/to/cases.json python scripts/run_span_ablation.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.span_ablation import (
    build_ablation_plan,
    build_attention_plan,
    summarize_case,
)


def run_native_runner(
    *,
    runner: Path,
    model: str,
    plan: dict,
    threads: str,
    ctx: str,
    batch: str,
    seqs: str,
    dump_attention: bool,
) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        plan_path = Path(directory) / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = [
            str(runner),
            "-m", model,
            "--plan", str(plan_path),
            "--threads", threads,
            "--ctx", ctx,
            "--batch", batch,
            "--seqs", seqs,
        ]
        if dump_attention:
            command.append("--dump-attention")
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"native runner failed ({completed.returncode}): {completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


def main() -> None:
    model = os.environ.get("ACL_MODEL_PATH", "/data/llama/models/Qwen3-4B-Q4_K_M.gguf")
    runner = Path(os.environ.get("ACL_NATIVE_RUNNER", str(ROOT / "build" / "native-probes" / "semantic-kv-runner")))
    cases_path = Path(os.environ.get("ACL_CASES", str(ROOT / "benchmark-results" / "span-ablation-cases.json")))
    threads = os.environ.get("ACL_THREADS", "10")
    ctx = os.environ.get("ACL_CTX", "8192")
    batch = os.environ.get("ACL_BATCH", "2048")
    seqs = os.environ.get("ACL_SEQS", "4")
    generate_tokens = int(os.environ.get("ACL_GENERATE_TOKENS", "64"))
    output = Path(os.environ.get("ACL_OUTPUT", str(ROOT / "benchmark-results" / "span-ablation.json")))

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    records = []
    started_all = time.time()
    for index, case in enumerate(cases, start=1):
        started = time.time()
        attention_native = run_native_runner(
            runner=runner, model=model, plan=build_attention_plan(case),
            threads=threads, ctx=ctx, batch=batch, seqs=seqs, dump_attention=True,
        )
        ablation_native = run_native_runner(
            runner=runner, model=model,
            plan=build_ablation_plan(case, generate_tokens=generate_tokens),
            threads=threads, ctx=ctx, batch=batch, seqs=seqs, dump_attention=False,
        )
        record = summarize_case(case, attention_native, ablation_native)
        record["wall_seconds"] = round(time.time() - started, 2)
        records.append(record)
        print(
            f"[{index}/{len(cases)}] {record['case_id']}: "
            f"cos={record['cosine_similarity']:.5f} "
            f"top{record['top_k_overlap']} "
            f"div@{record['greedy_divergence_index']} "
            f"attn/row={record['attention_mass_per_row']:.5f} "
            f"({record['wall_seconds']}s)",
            flush=True,
        )

    payload = {
        "metadata": {
            "mode": "span_ablation",
            "model": model,
            "cases": len(records),
            "generate_tokens": generate_tokens,
            "wall_seconds": round(time.time() - started_all, 2),
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
