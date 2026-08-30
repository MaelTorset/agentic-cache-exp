"""Correlate cheap predictors against the measured cost of forgetting a span.

Reads the ablation records and asks one question: does any signal computable
*before* deleting a span predict how much deleting it actually hurt?

Damage is measured as ``1 - cosine_similarity`` between the spliced state's
pre-generation logits and the clean reference's, so larger means worse. Greedy
divergence is reported alongside but is not the primary target: it saturates
(either the generations match or they do not), which throws away the gradation
the correlation needs.

The two trivial baselines -- span length and distance to the end of the context
-- are the bar to clear. A predictor that does not beat them has only
rediscovered the geometry of the context, not anything about its content.

Spearman is computed directly (rank-transform, then Pearson on ranks) to keep
the script dependency-free; ties get average ranks.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PREDICTORS = (
    ("attention_mass_per_token", "attention per span token, all layers (hypothesis)"),
    ("attention_late_layers_per_token", "attention per span token, last quarter of layers"),
    ("attention_final_layer_per_token", "attention per span token, final layer"),
    ("attention_mid_layers_per_token", "attention per span token, middle layers"),
    ("attention_early_layers_per_token", "attention per span token, first quarter of layers"),
    ("attention_mass_per_row", "attention per softmax row (length-confounded)"),
    ("attention_mass_raw", "raw attention mass"),
    ("span_tokens", "span length [baseline]"),
    ("span_token_share", "span share of context [baseline]"),
    ("distance_to_end", "distance to context end [baseline]"),
)

BASELINES = {"span_tokens", "span_token_share", "distance_to_end"}


def rank(values: list[float]) -> list[float]:
    """Average ranks, so tied values do not fabricate ordering."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        shared = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denominator = (sum(a * a for a in dx) ** 0.5) * (sum(b * b for b in dy) ** 0.5)
    if denominator == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rank(xs), rank(ys))


def group_means(records: list[dict], key: str, value: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for record in records:
        buckets.setdefault(str(record[key]), []).append(float(record[value]))
    return {name: sum(values) / len(values) for name, values in sorted(buckets.items())}


def main() -> None:
    source = Path(os.environ.get("ACL_ABLATION", str(ROOT / "benchmark-results" / "span-ablation.json")))
    output = Path(os.environ.get("ACL_OUTPUT", str(ROOT / "benchmark-results" / "span-ablation-analysis.json")))
    threshold = float(os.environ.get("ACL_RHO_THRESHOLD", "0.5"))

    payload = json.loads(source.read_text(encoding="utf-8"))
    all_records = payload["records"]

    # A span with no surviving tokens after it cannot have contaminated
    # anything, so its damage is structurally zero and tells us nothing about
    # whether attention predicts damage. Drop those before correlating.
    records = [record for record in all_records if record.get("contaminable", True)]
    dropped = len(all_records) - len(records)
    if dropped:
        print(f"dropped {dropped} degenerate case(s) with an empty suffix", file=sys.stderr)
    if len(records) < 30:
        print(f"WARNING: only {len(records)} records; the kill criterion assumes at least 30.", file=sys.stderr)

    damage = [1.0 - float(record["cosine_similarity"]) for record in records]

    correlations = {}
    for name, label in PREDICTORS:
        if any(name not in record for record in records):
            continue  # layerwise columns are added by a separate pass
        values = [float(record[name]) for record in records]
        correlations[name] = {
            "label": label,
            "spearman_rho": round(spearman(values, damage), 4),
            "is_baseline": name in BASELINES,
        }

    best_baseline = max(
        (abs(stats["spearman_rho"]) for name, stats in correlations.items() if stats["is_baseline"]),
        default=0.0,
    )
    candidates = {name: stats for name, stats in correlations.items() if not stats["is_baseline"]}
    best_name = max(candidates, key=lambda name: abs(candidates[name]["spearman_rho"]))
    best_rho = candidates[best_name]["spearman_rho"]

    passes_threshold = abs(best_rho) >= threshold
    beats_baselines = abs(best_rho) > best_baseline
    verdict = "PASS" if (passes_threshold and beats_baselines) else "FAIL"

    usable = [record for record in records if record.get("greedy_usable", True)]
    analysis = {
        "records": len(records),
        "greedy_usable_records": len(usable),
        "greedy_exact_match_rate": (
            round(sum(1 for r in usable if r["greedy_exact_match"]) / len(usable), 4) if usable else None
        ),
        "damage_metric": "1 - cosine_similarity (spliced vs clean pre-generation logits)",
        "damage_min": round(min(damage), 6),
        "damage_median": round(sorted(damage)[len(damage) // 2], 6),
        "damage_max": round(max(damage), 6),
        "correlations": correlations,
        "best_candidate": best_name,
        "best_candidate_rho": best_rho,
        "best_baseline_abs_rho": round(best_baseline, 4),
        "threshold": threshold,
        "passes_threshold": passes_threshold,
        "beats_baselines": beats_baselines,
        "verdict": verdict,
        "damage_by_span_kind": {k: round(v, 6) for k, v in group_means(records, "span_kind", "cosine_similarity").items()},
        "attention_by_span_kind": {
            k: round(v, 6) for k, v in group_means(records, "span_kind", "attention_mass_per_token").items()
        },
    }

    output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"records: {analysis['records']}  (greedy usable: {analysis['greedy_usable_records']})")
    print(f"damage 1-cos: min={analysis['damage_min']} median={analysis['damage_median']} max={analysis['damage_max']}")
    print("\nSpearman rho vs damage:")
    for name, stats in sorted(correlations.items(), key=lambda kv: -abs(kv[1]["spearman_rho"])):
        tag = "[baseline]" if stats["is_baseline"] else ""
        print(f"  {stats['spearman_rho']:+.4f}  {name:28s} {tag}")
    print(f"\nbest candidate: {best_name} rho={best_rho:+.4f}")
    print(f"best baseline |rho|: {best_baseline:.4f}")
    print(f"threshold {threshold} passed: {passes_threshold}; beats baselines: {beats_baselines}")
    print(f"VERDICT: {verdict}")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
