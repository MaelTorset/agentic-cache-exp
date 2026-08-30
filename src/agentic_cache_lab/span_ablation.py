"""Ground truth for selective forgetting: what does deleting a span actually cost?

Each case is a context split into four segments::

    prefix | span | suffix | query

``span`` is the candidate for removal. ``query`` is what the agent asks next,
and is where divergence is measured.

Two sequences are built:

- ``clean``   -- prefix, suffix, query. The reference: what the model does when
  it never saw the span at all.
- ``spliced`` -- prefix, span, suffix, then the span's KV is removed and the
  survivors are RoPE-shifted left by ``len(span)``, then query is evaluated.

Both end with identical tokens at identical positions, so their pre-generation
logits are directly comparable. They differ only in that the spliced suffix was
*computed while the span was in context* -- the contamination this measures.

Segments are tokenized independently by the runner, so the clean and spliced
sequences share byte-identical prefix/suffix/query token sequences; there is no
tokenizer-boundary drift between the two conditions.

A separate attention plan evaluates the full context without splicing, so the
per-position attention mass can be read for the span's range.
"""

from __future__ import annotations

CLEAN_SEQ = 0
SPLICED_SEQ = 1
ATTENTION_SEQ = 0


def _segments(case: dict) -> list[dict]:
    return [
        {"id": "prefix", "text": case["prefix"], "add_bos": True},
        {"id": "span", "text": case["span"]},
        {"id": "suffix", "text": case["suffix"]},
        {"id": "query", "text": case["query"]},
    ]


def build_attention_plan(case: dict, top_k: int = 20) -> dict:
    """Evaluate the full context, unspliced, so attention mass can be read.

    No removal happens here: every position keeps its original index, so
    ``attention.mass[p]`` lines up with the ``p0``/``p1`` range the ``eval`` ops
    report for each segment.
    """
    return {
        "config": {"top_k": top_k, "suppress_logs": True},
        "metadata": {"mode": "span_attention", "case_id": case["id"]},
        "segments": _segments(case),
        "ops": [
            {"op": "eval", "seq": ATTENTION_SEQ, "segment": "prefix", "pos": 0},
            {"op": "eval", "seq": ATTENTION_SEQ, "segment": "span", "start_after_segment": "prefix"},
            {"op": "eval", "seq": ATTENTION_SEQ, "segment": "suffix", "start_after_segment": "span"},
            {"op": "eval", "seq": ATTENTION_SEQ, "segment": "query", "start_after_segment": "suffix", "logits": True},
        ],
    }


def build_ablation_plan(case: dict, generate_tokens: int = 64, top_k: int = 20) -> dict:
    """Clean reference versus spliced state, compared and generated from."""
    ops: list[dict] = [
        # clean: the span was never evaluated
        {"op": "eval", "seq": CLEAN_SEQ, "segment": "prefix", "pos": 0},
        {"op": "eval", "seq": CLEAN_SEQ, "segment": "suffix", "start_after_segment": "prefix"},
        {"op": "eval", "seq": CLEAN_SEQ, "segment": "query", "start_after_segment": "suffix", "logits": True},
        # spliced: full context, then forget the span
        {"op": "eval", "seq": SPLICED_SEQ, "segment": "prefix", "pos": 0},
        {"op": "eval", "seq": SPLICED_SEQ, "segment": "span", "start_after_segment": "prefix"},
        {"op": "eval", "seq": SPLICED_SEQ, "segment": "suffix", "start_after_segment": "span"},
        {"op": "remove", "seq": SPLICED_SEQ, "segment": "span"},
        # close the hole: shift everything from the span's end leftwards by len(span)
        {
            "op": "shift",
            "seq": SPLICED_SEQ,
            "start_after_segment": "span",
            "p1": -1,
            "delta_segment": "span",
            "direction": "negative",
        },
        {"op": "eval", "seq": SPLICED_SEQ, "segment": "query", "start_after_segment": "suffix", "logits": True},
        {"op": "compare", "left": CLEAN_SEQ, "right": SPLICED_SEQ, "label": "spliced_vs_clean"},
    ]
    if generate_tokens > 0:
        ops.append({"op": "generate", "seq": CLEAN_SEQ, "label": "gen_clean", "max_tokens": generate_tokens})
        ops.append({"op": "generate", "seq": SPLICED_SEQ, "label": "gen_spliced", "max_tokens": generate_tokens})

    return {
        "config": {"top_k": top_k, "suppress_logs": True},
        "metadata": {"mode": "span_ablation", "case_id": case["id"]},
        "segments": _segments(case),
        "ops": ops,
    }


def segment_ranges(native: dict) -> dict[str, tuple[int, int]]:
    """Extract each segment's [p0, p1) from the runner's per-op output.

    Only the first eval of a segment is recorded, which is the unspliced
    position range in an attention run.
    """
    ranges: dict[str, tuple[int, int]] = {}
    for op in native.get("ops", []):
        if op.get("op") != "eval":
            continue
        segment = str(op.get("segment", ""))
        if segment and segment not in ranges:
            ranges[segment] = (int(op["p0"]), int(op["p1"]))
    return ranges


def first_divergence_index(left: str, right: str) -> int:
    """Character index where two greedy generations first differ.

    Returns -1 when one is a prefix of the other and they never contradict.
    """
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) == len(right):
        return -1
    return min(len(left), len(right))


def summarize_case(case: dict, attention_native: dict, ablation_native: dict) -> dict:
    """Merge one case's two runner outputs into a single flat record."""
    ranges = segment_ranges(attention_native)
    span_p0, span_p1 = ranges["span"]
    attention = attention_native["attention"]
    mass = attention["mass"]

    # Normalise by the number of (query, head, layer) rows so the figure is a
    # mean attention weight rather than a count that grows with context size.
    rows = float(attention["queries_total"] * attention["n_heads"])
    span_mass = float(sum(mass[span_p0:span_p1]))
    context_mass = float(sum(mass))

    comparison = dict(ablation_native["comparisons"]["spliced_vs_clean"])
    generations = ablation_native.get("generations", {})
    clean_generation = dict(generations.get("gen_clean", {}))
    spliced_generation = dict(generations.get("gen_spliced", {}))
    clean_text = str(clean_generation.get("text", ""))
    spliced_text = str(spliced_generation.get("text", ""))
    clean_tokens = int(clean_generation.get("tokens", 0))

    span_tokens = span_p1 - span_p0
    total_tokens = max(int(ranges["query"][1]), 1)
    suffix_p0, suffix_p1 = ranges["suffix"]
    suffix_tokens = suffix_p1 - suffix_p0

    return {
        "case_id": case["id"],
        "span_kind": case.get("span_kind", "unknown"),
        "span_position": case.get("span_position", "unknown"),
        # --- measured damage (what we are trying to predict) ---
        "cosine_similarity": comparison["cosine_similarity"],
        "top_k_overlap": comparison["top_k_overlap"],
        "top_token_match": comparison["top_token_match"],
        "mean_abs_diff": comparison["mean_abs_diff"],
        "greedy_divergence_index": first_divergence_index(clean_text, spliced_text),
        "greedy_exact_match": clean_text == spliced_text,
        # A model that emits end-of-generation immediately produces two empty
        # strings, which compare equal for reasons that have nothing to do with
        # the splice. Correlating against that is correlating against noise, so
        # flag it rather than silently scoring it as "no damage".
        "greedy_usable": clean_tokens > 0,
        "greedy_tokens_clean": clean_tokens,
        # --- candidate predictors (cheap, computable before removal) ---
        # Note: total context mass equals `rows` exactly (every softmax row sums
        # to 1), so a share-of-context figure would be identical to the per-row
        # one. The two useful normalisations are per-row (how much of the whole
        # context's attention the span absorbed) and per-token (how heavily an
        # individual span token was attended), which decouples the signal from
        # span length.
        "attention_mass_raw": span_mass,
        "attention_mass_per_row": span_mass / rows if rows else 0.0,
        "attention_mass_per_token": (span_mass / rows / span_tokens) if rows and span_tokens else 0.0,
        # --- trivial baselines: any predictor must beat these to matter ---
        "span_tokens": span_tokens,
        "span_token_share": span_tokens / total_tokens,
        "distance_to_end": total_tokens - span_p1,
        # Tokens that were computed while the span was still visible, and so
        # are the only ones the splice can have contaminated. When this is zero
        # the case is degenerate: removal is exactly free for structural
        # reasons, not because the span was unimportant.
        "suffix_tokens": suffix_tokens,
        "contaminable": suffix_tokens > 0,
        "span_p0": span_p0,
        "span_p1": span_p1,
        "context_tokens": total_tokens,
        "generation_clean": clean_text,
        "generation_spliced": spliced_text,
    }
