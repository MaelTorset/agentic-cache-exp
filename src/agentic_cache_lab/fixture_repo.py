from __future__ import annotations

from pathlib import Path

from .models import AgentEvent, EventKind


DEFAULT_SESSION_READS = (
    "frontend/src/qr/scanner.tsx",
    "frontend/src/qr/qr-route.ts",
    "backend/src/billing/invoice.ts",
    "backend/src/analytics/events.ts",
    "backend/src/auth/cookie.ts",
    "backend/tests/auth-cookie.test.ts",
)


def build_fixture_repo_events(repo_root: Path, reads: tuple[str, ...] = DEFAULT_SESSION_READS) -> list[AgentEvent]:
    """Build a deterministic agent-like trace from a small repo fixture."""
    events = [
        AgentEvent(
            kind=EventKind.USER,
            source="conversation",
            text=(
                "We are debugging a small TypeScript repo. The agent first inspects QR, "
                "billing, and analytics files, then switches to an auth cookie failure."
            ),
            created_at=1,
        )
    ]

    timestamp = 2
    for relative_path in reads:
        path = repo_root / relative_path
        events.append(
            AgentEvent(
                kind=EventKind.FILE_READ,
                source=relative_path,
                text=path.read_text(encoding="utf-8"),
                created_at=timestamp,
            )
        )
        timestamp += 1

        if "qr/" in relative_path:
            events.append(
                AgentEvent(
                    kind=EventKind.DECISION,
                    source="session-notes",
                    text="Temporary branch: QR onboarding uses camera permission and manual-code fallback logic.",
                    created_at=timestamp,
                )
            )
            timestamp += 1

    events.extend(
        [
            AgentEvent(
                kind=EventKind.ERROR,
                source="test-output/auth-cookie.test.ts",
                text=(
                    "FAIL buildSessionCookie uses a localhost-compatible cookie in development. "
                    "Expected SameSite=Lax without Secure. Observed SameSite=None without Secure, "
                    "which browsers reject on local HTTP."
                ),
                created_at=timestamp,
            ),
            AgentEvent(
                kind=EventKind.DECISION,
                source="session-notes",
                text=(
                    "Active branch: auth cookie generation. The fix belongs in "
                    "backend/src/auth/cookie.ts. QR, billing, and analytics are unrelated unless "
                    "the task switches back."
                ),
                created_at=timestamp + 1,
            ),
            AgentEvent(
                kind=EventKind.USER,
                source="conversation",
                text=(
                    "Now solve the auth cookie bug, then be ready to return to the QR branch "
                    "without rereading the shared session context."
                ),
                created_at=timestamp + 2,
            ),
        ]
    )
    return events

