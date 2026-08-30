"""Non-prefix KV splice probe: shift-only RoPE re-alignment (0%-repair point).

Question: after evaluating ``A + noise + B``, can we delete the noise KV and
shift B left (``llama_memory_seq_add``) to impersonate a scratch ``A + B``
state? Expectation from theory: no — B's KV was computed attending to the
noise — but the *magnitude* of the damage (logit diff, top-k overlap, greedy
divergence point) is the measurable result, and the baseline for future
CacheBlend-style selective recompute (see docs/research-watch-2026-07.md).

Conditions compared at the pre-generation position:
- seq 1 ``spliced``:  eval A+noise+B, copy, remove noise KV, shift B left, eval task
- seq 2 ``scratch``:  eval A+B, eval task
- seq 3 ``noisy``:    eval A+noise+B, eval task (quality reference: what the
  spliced state "remembers")
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_cache_lab.fixture_repo import build_fixture_repo_events
from agentic_cache_lab.segment_store import SegmentStore
from agentic_cache_lab.semantic_kv_planner import build_branch_text, build_root_text

TASK_TEXT = (
    "\nTask:\nName the file to patch for the failing auth cookie test and state "
    "the root cause in one sentence.\nAnswer:\n"
)


def build_splice_plan(events, generate_tokens: int = 48) -> dict[str, object]:
    store = SegmentStore()
    segments = store.add_events(events)
    seg_a = build_root_text(segments)
    seg_b = build_branch_text("auth", segments, max_segments=4, max_words=80)
    noise = build_branch_text("qr", segments, max_segments=4, max_words=80)

    return {
        "config": {"top_k": 20, "suppress_logs": True},
        "metadata": {
            "mode": "kv_splice_shift_only_probe",
            "generate_tokens": generate_tokens,
        },
        "segments": [
            {"id": "seg_a", "text": seg_a, "add_bos": True},
            {"id": "noise", "text": noise},
            {"id": "seg_b", "text": seg_b},
            {"id": "task", "text": TASK_TEXT},
        ],
        "ops": [
            # full noisy context on seq 0
            {"op": "eval", "seq": 0, "segment": "seg_a", "pos": 0},
            {"op": "eval", "seq": 0, "segment": "noise", "start_after_segment": "seg_a"},
            {"op": "eval", "seq": 0, "segment": "seg_b", "start_after_segment": "noise"},
            # spliced condition: copy, drop noise KV, shift B left onto A
            {"op": "copy", "from": 0, "to": 1},
            {"op": "remove", "seq": 1, "segment": "noise"},
            {
                "op": "shift",
                "seq": 1,
                "start_after_segment": "noise",
                "p1": -1,
                "delta_segment": "noise",
                "direction": "negative",
            },
            {"op": "eval", "seq": 1, "segment": "task", "start_after_segment": "seg_b", "logits": True},
            # scratch A+B reference
            {"op": "eval", "seq": 2, "segment": "seg_a", "pos": 0},
            {"op": "eval", "seq": 2, "segment": "seg_b", "start_after_segment": "seg_a"},
            {"op": "eval", "seq": 2, "segment": "task", "start_after_segment": "seg_b", "logits": True},
            # noisy reference (what seq 1's B actually attended to)
            {"op": "copy", "from": 0, "to": 3},
            {"op": "eval", "seq": 3, "segment": "task", "start_after_segment": "seg_b", "logits": True},
            {"op": "compare", "left": 1, "right": 2, "label": "spliced_vs_scratch"},
            {"op": "compare", "left": 1, "right": 3, "label": "spliced_vs_noisy"},
            {"op": "compare", "left": 3, "right": 2, "label": "noisy_vs_scratch"},
            {"op": "generate", "seq": 1, "label": "gen_spliced", "max_tokens": generate_tokens},
            {"op": "generate", "seq": 2, "label": "gen_scratch", "max_tokens": generate_tokens},
            {"op": "generate", "seq": 3, "label": "gen_noisy", "max_tokens": generate_tokens},
        ],
    }


def greedy_divergence(a: str, b: str) -> int:
    """Index of first differing character (proxy for divergence point)."""
    for i, (ca, cb) in enumerate(zip(a, b, strict=False)):
        if ca != cb:
            return i
    return min(len(a), len(b)) if a != b else -1


def main() -> None:
    repo_root = Path(os.environ.get("ACL_FIXTURE_REPO", str(ROOT / "examples" / "fixtures" / "shopbug-repo")))
    model = os.environ.get("ACL_MODEL_PATH", "/data/llama/models/Qwen3-4B-Q4_K_M.gguf")
    runner = Path(os.environ.get("ACL_NATIVE_RUNNER", str(ROOT / "build" / "native-probes" / "semantic-kv-runner")))
    generate_tokens = int(os.environ.get("ACL_GENERATE_TOKENS", "48"))
    output = Path(os.environ.get("ACL_OUTPUT", str(ROOT / "benchmark-results" / "kv-splice-shift-only-probe.json")))

    events = build_fixture_repo_events(repo_root)
    plan = build_splice_plan(events, generate_tokens=generate_tokens)

    with tempfile.TemporaryDirectory() as directory:
        plan_path = Path(directory) / "kv_splice_plan.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        completed = subprocess.run(
            [
                str(runner),
                "-m",
                model,
                "--plan",
                str(plan_path),
                "--threads",
                os.environ.get("ACL_THREADS", "10"),
                "--ctx",
                os.environ.get("ACL_CTX", "4096"),
                "--batch",
                os.environ.get("ACL_BATCH", "1024"),
                "--seqs",
                os.environ.get("ACL_SEQS", "4"),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"semantic-kv-runner failed: {completed.stdout}\n{completed.stderr}")
    raw = json.loads(completed.stdout)

    generations = {label: dict(g) for label, g in dict(raw.get("generations", {})).items()}
    texts = {label: str(g.get("text", "")) for label, g in generations.items()}
    result = {
        "benchmark": "kv_splice_shift_only_probe",
        "repo_root": str(repo_root),
        "model": model,
        "plan_metadata": plan["metadata"],
        "segment_tokens": raw.get("segments", {}),
        "comparisons": raw.get("comparisons", {}),
        "generation_texts": texts,
        "greedy_match": {
            "spliced_vs_scratch": texts.get("gen_spliced", "") == texts.get("gen_scratch", "x"),
            "spliced_vs_noisy": texts.get("gen_spliced", "") == texts.get("gen_noisy", "x"),
            "divergence_char_spliced_vs_scratch": greedy_divergence(
                texts.get("gen_spliced", ""), texts.get("gen_scratch", "")
            ),
            "divergence_char_spliced_vs_noisy": greedy_divergence(
                texts.get("gen_spliced", ""), texts.get("gen_noisy", "")
            ),
        },
        "native": raw,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("comparisons", "greedy_match")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
