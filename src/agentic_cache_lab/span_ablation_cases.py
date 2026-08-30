"""Build the (context, span) pairs the ablation harness measures.

The agent task is fixed across every case: find the auth-cookie bug in the
shopbug fixture repo. What varies is the span offered for removal, along the
three axes that should drive how much damage removing it does:

- **kind**     -- dead file read, relevant file read, neutral filler, or a
                  decision/instruction note.
- **position** -- how much context sits before the span versus after it.
- **length**   -- how many times the span's content is repeated.

Only ``relevant_read`` spans carry the information the query needs, so if
attention mass means anything it should separate them from the rest.

Position matters for a mechanical reason as well as a semantic one: a span near
the end has fewer surviving tokens after it, so less of the context was computed
while it was visible. That is exactly why ``distance_to_end`` is carried as a
trivial baseline -- a predictor that only recovers position has found nothing.
"""

from __future__ import annotations

from pathlib import Path

FILLER_SENTENCE = (
    "Project convention: keep changes minimal, respect the existing style, run the "
    "test suite before concluding, and never touch unrelated modules. "
)

DECISION_NOTE = (
    "Session note: the active branch is auth cookie generation. QR onboarding, "
    "billing, and analytics are parked and should not be modified. "
)

TASK_FRAMING = (
    "You are an agent debugging a small TypeScript repository.\n"
    "A test is failing: buildSessionCookie uses a localhost-compatible cookie in "
    "development. Expected SameSite=Lax without Secure. Observed SameSite=None "
    "without Secure, which browsers reject on local HTTP.\n"
)

QUERY = "\nWhich file must be changed to fix the failing test? Answer with the path only."

DEAD_FILES = (
    "backend/src/billing/invoice.ts",
    "backend/src/analytics/events.ts",
    "frontend/src/qr/scanner.tsx",
)

RELEVANT_FILES = (
    "backend/src/auth/cookie.ts",
    "backend/tests/auth-cookie.test.ts",
)

# Repeat counts for the three length buckets.
LENGTHS = {"short": 1, "medium": 3, "long": 6}

# How the non-span context is split around the span. Each entry is
# (filler repeats before the span, filler repeats after the span).
#
# The trailing count is never zero. With no surviving tokens between the span
# and the query there is nothing that was computed while the span was visible,
# so removing it is exactly free and the case measures the absence of a
# mechanism rather than a cheap position. Those cases score zero damage against
# a non-zero predictor and would drag every correlation towards nothing.
POSITIONS = {"early": (0, 6), "middle": (3, 3), "late": (6, 1)}


def _read(repo_root: Path, relative_path: str) -> str:
    return f"\n// {relative_path}\n" + (repo_root / relative_path).read_text(encoding="utf-8")


def _span_text(repo_root: Path, kind: str, repeats: int) -> str:
    if kind == "dead_read":
        sources = DEAD_FILES
    elif kind == "relevant_read":
        sources = RELEVANT_FILES
    elif kind == "filler":
        return FILLER_SENTENCE * (repeats * 3)
    elif kind == "decision":
        return DECISION_NOTE * (repeats * 3)
    else:
        raise ValueError(f"unknown span kind: {kind}")

    # Cycle through the available files so a longer span is genuinely more
    # material rather than the same file pasted twice.
    return "".join(_read(repo_root, sources[i % len(sources)]) for i in range(repeats))


def build_cases(repo_root: Path) -> list[dict]:
    """Return one case per (kind, position, length) combination."""
    # Anchor context the query actually needs, so a case is never unanswerable.
    anchor = _read(repo_root, "backend/src/auth/cookie.ts")

    cases: list[dict] = []
    for kind in ("dead_read", "relevant_read", "filler", "decision"):
        for position, (before, after) in POSITIONS.items():
            for length, repeats in LENGTHS.items():
                span = _span_text(repo_root, kind, repeats)

                prefix = TASK_FRAMING + FILLER_SENTENCE * before
                suffix = FILLER_SENTENCE * after
                if kind != "relevant_read":
                    # Keep the answer derivable when the span itself is not the
                    # evidence; otherwise every non-relevant case would measure
                    # an unanswerable question instead of a forgetting cost.
                    suffix = anchor + suffix

                cases.append(
                    {
                        "id": f"{kind}__{position}__{length}",
                        "span_kind": kind,
                        "span_position": position,
                        "span_length": length,
                        "prefix": prefix,
                        "span": span,
                        "suffix": suffix,
                        "query": QUERY,
                    }
                )
    return cases
