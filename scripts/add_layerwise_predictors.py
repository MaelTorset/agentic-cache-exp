"""Re-run only the attention pass and add depth-resolved predictors.

The aggregate attention signal sums every layer together, which can hide a
signal that lives at one end of the network: early layers are largely
positional, late layers carry the semantic routing. This re-runs each case's
(cheap) attention plan with per-layer capture and merges the depth-sliced
predictors into the existing ablation records.

The ablation pass is untouched -- the measured damage is unchanged, only the
predictor columns grow.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agentic_cache_lab.span_ablation import build_attention_plan, segment_ranges
from run_span_ablation import run_native_runner


def layer_slice_mass(mass_by_layer: list[list[float]], p0: int, p1: int, lo: float, hi: float) -> float:
    """Attention the span received, summed over a fractional slice of depth."""
    n_layers = len(mass_by_layer)
    start = int(n_layers * lo)
    stop = max(start + 1, int(n_layers * hi))
    return sum(sum(layer[p0:p1]) for layer in mass_by_layer[start:stop])


def main() -> None:
    model = os.environ.get("ACL_MODEL_PATH", "/data/llama/models/Qwen3-4B-Q4_K_M.gguf")
    runner = Path(os.environ.get("ACL_NATIVE_RUNNER", str(ROOT / "build" / "native-probes" / "semantic-kv-runner")))
    cases_path = Path(os.environ.get("ACL_CASES", str(ROOT / "benchmark-results" / "span-ablation-cases.json")))
    ablation_path = Path(os.environ.get("ACL_ABLATION", str(ROOT / "benchmark-results" / "span-ablation.json")))
    threads = os.environ.get("ACL_THREADS", "10")
    ctx = os.environ.get("ACL_CTX", "16384")
    batch = os.environ.get("ACL_BATCH", "2048")
    seqs = os.environ.get("ACL_SEQS", "4")

    cases = {case["id"]: case for case in json.loads(cases_path.read_text(encoding="utf-8"))}
    payload = json.loads(ablation_path.read_text(encoding="utf-8"))

    slices = {
        "early_layers": (0.0, 0.25),
        "mid_layers": (0.25, 0.75),
        "late_layers": (0.75, 1.0),
        "final_layer": (0.96, 1.0),
    }

    for index, record in enumerate(payload["records"], start=1):
        case = cases[record["case_id"]]
        native = run_native_runner(
            runner=runner, model=model, plan=build_attention_plan(case),
            threads=threads, ctx=ctx, batch=batch, seqs=seqs, dump_attention=True,
        )
        attention = native["attention"]
        mass_by_layer = attention["mass_by_layer"]
        p0, p1 = segment_ranges(native)["span"]
        span_tokens = max(p1 - p0, 1)
        n_layers = max(len(mass_by_layer), 1)

        for name, (lo, hi) in slices.items():
            start = int(n_layers * lo)
            stop = max(start + 1, int(n_layers * hi))
            # Normalise by the rows contributing to this slice, so slices of
            # different depth are on the same scale, then by span length so the
            # figure is a mean weight per span token.
            rows = attention["queries_total"] / n_layers * (stop - start) * attention["n_heads"]
            mass = layer_slice_mass(mass_by_layer, p0, p1, lo, hi)
            record[f"attention_{name}_per_token"] = (mass / rows / span_tokens) if rows else 0.0

        print(
            f"[{index}/{len(payload['records'])}] {record['case_id']}: "
            f"late={record['attention_late_layers_per_token']:.6f} "
            f"final={record['attention_final_layer_per_token']:.6f}",
            flush=True,
        )

    payload["metadata"]["layerwise_predictors"] = sorted(slices)
    ablation_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"updated {ablation_path}")


if __name__ == "__main__":
    main()
