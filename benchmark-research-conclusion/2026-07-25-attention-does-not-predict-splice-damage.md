# Attention Does Not Predict Splice Damage (Negative Result)

Date: 2026-07-25
Model: Qwen3-4B-Q4_K_M (CPU, 10 threads, ctx 16384, batch 2048).
Scripts: `scripts/run_span_ablation.py`, `scripts/add_layerwise_predictors.py`,
`scripts/analyze_span_ablation.py`.
Raw data: `benchmark-results/span-ablation.json`,
`benchmark-results/span-ablation-analysis.json`.

## The question

Selective forgetting -- deleting a span from the middle of a KV cache and
RoPE-shifting the survivors -- is cheap but not exact. Earlier probes showed the
damage varies wildly with *what* is removed: the branch-adjacent splice diverged
within ~21 characters (`2026-07-23-kv-splice-shift-only-probe.md`) while removing
bulky dead file reads was nearly benign (`2026-07-24-dead-reads-forgetting.md`).

That suggested a hypothesis worth testing, because it is the difference between
a curiosity and a usable mechanism:

> The damage from removing a span is predicted by how much the surviving tokens
> attended to it. If so, a harness can decide *before* deleting whether a span
> is free to forget.

Kill criterion, fixed before any measurement: if the best candidate predictor
scores |Spearman rho| < 0.5 against measured damage across at least 30 cases,
**and** fails to beat the trivial baselines (span length, distance to the end of
the context), then selective forgetting is not steerable and the line of work
stops.

## Method

36 cases over the shopbug fixture, factorial across three axes: span kind (dead
file read / relevant file read / neutral filler / decision note) x position
(early / middle / late) x length (~40 / ~160 / ~380 words).

Each case is `prefix | span | suffix | query`, measured twice:

- **clean** -- `prefix + suffix + query` evaluated from scratch: what the model
  does having never seen the span.
- **spliced** -- `prefix + span + suffix`, then the span's KV is removed, the
  survivors are shifted left by `len(span)`, and `query` is evaluated.

Both end with identical tokens at identical positions. They differ only in that
the spliced suffix was computed while the span was visible. Damage is
`1 - cosine_similarity` between the two pre-generation logit vectors.

Attention was captured through llama.cpp's public `cb_eval` hook, reading the
`kq_soft_max-<il>` tensors (which requires disabling flash attention, since it
fuses the softmax away). Validation: total captured mass equals
`queries x heads x layers` exactly, and padded positions are exactly zero, so
every softmax row is accounted for once.

## Result: the criterion fails

```text
Spearman rho vs damage (36 cases)

  +0.4606  distance_to_end                    [baseline]
  +0.3743  attention_mass_raw
  +0.3393  attention_mass_per_row
  +0.3199  span_tokens                        [baseline]
  +0.3024  span_token_share                   [baseline]
  -0.2252  attention_late_layers_per_token
  -0.1894  attention_final_layer_per_token
  +0.0355  attention_mid_layers_per_token
  +0.0296  attention_early_layers_per_token
  +0.0013  attention_mass_per_token
```

No attention variant reaches 0.5, and none beats `distance_to_end`. The two
attention figures that correlate at all (`raw`, `per_row`) are not normalised by
span length, so they are largely restating `span_tokens`.

Attention per token is flat across span kinds, which is the hypothesis failing
at its most basic: it cannot tell the file the answer depends on from filler.

```text
                 mean damage (1-cos)   mean attention per token
dead_read              0.01913                 0.000920
filler                 0.00647                 0.000879
decision               0.00647                 0.000861
relevant_read          0.00477                 0.000912
```

## Why it fails: attention is re-encoding position

Conditioning shows the correlation is borrowed, not owned. Holding **length**
constant, attention per token looks like a decent predictor. Holding
**position** constant, it collapses and flips sign:

```text
                        within length bucket      within position bucket
                        short  medium   long      early  middle    late
attention_per_token     +0.61   +0.49  +0.59      -0.34   -0.15   -0.04
attention_late_layers   +0.03   -0.28  +0.17      -0.32   -0.34   -0.56
```

A predictor whose sign depends on what you condition on is tracking a
confounder. Position drives damage for a mechanical reason -- a span early in
the context has more surviving tokens computed after it, so more of the state
was contaminated -- and attention mass correlates with position. Once position
is held fixed, the residual attention signal carries nothing.

That is the finding: **the cost of forgetting a span is governed by how much
context was computed after it, not by how much the model attended to it.**

## Secondary result worth keeping

Splicing is rarely exact. Across the 36 cases only **19.4%** reproduced the
clean greedy output exactly; the other 80% diverged. Damage spans a 27x range
(`1-cos` from 0.0012 to 0.0334), so the operation is neither uniformly safe nor
uniformly destructive -- it is just not steerable by this signal.

## Caveats, including one that limits the cross-kind comparison

- **The by-kind table is confounded.** For `relevant_read` cases the span *is*
  the evidence, so the suffix holds only filler; for every other kind the suffix
  also carries the anchor file, to keep the question answerable. Those cases
  therefore have more contaminable content downstream. The by-kind damage
  ordering should not be read as a semantic effect. The correlation result does
  not rest on it -- attention per token is flat across kinds regardless.
- One model, one fixture repo, CPU only, single run per case. 36 cases is the
  pre-registered minimum, not a comfortable sample; the within-bucket
  conditional analyses run on n=12 each and are indicative only.
- Only shift-only repair was tested (0% recompute). A CacheBlend-style selective
  recompute could change the damage profile, but that is a different question:
  it would make splicing cheaper to *repair*, not more *predictable*.
- Attention was aggregated over heads. A per-head signal could in principle
  survive where the head-mean does not, though nothing in these results suggests
  a direction worth chasing.

## Verdict

The kill criterion was written before the measurements and it fails on both
clauses. Selective forgetting is not steerable by attention, and the best
available predictor of splice damage is a geometric property of the context that
requires no model introspection at all.

Per the criterion, this line of work stops here.
