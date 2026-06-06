from __future__ import annotations

from .models import AgentEvent, EventKind


def build_long_context_events(noise_blocks: int = 36) -> list[AgentEvent]:
    """Create a deterministic trace where only a small auth slice remains useful."""
    events = [
        AgentEvent(
            kind=EventKind.USER,
            source="conversation",
            text="We investigated QR onboarding, billing, search, and authentication in one long session.",
            created_at=1,
        ),
        AgentEvent(
            kind=EventKind.FILE_READ,
            source="backend/src/auth/session.ts",
            text=(
                "Auth middleware reads the session cookie, validates the JWT, loads the user, "
                "and rejects /api/me when the cookie is missing. Local HTTP must not use "
                "SameSite=None without Secure."
            ),
            created_at=2,
        ),
        AgentEvent(
            kind=EventKind.ERROR,
            source="logs/auth.log",
            text=(
                "GET /api/me failed with 401 after login. Browser did not store the session cookie "
                "because Set-Cookie used SameSite=None without Secure on localhost."
            ),
            created_at=3,
        ),
    ]

    for index in range(noise_blocks):
        topic = "qr" if index % 2 == 0 else "frontend"
        source = f"frontend/src/features/{topic}/noise_{index}.tsx"
        repeated = " ".join(
            [
                f"Noise block {index} for {topic}.",
                "This contains implementation details for scanner camera state, marketing UI copy,",
                "layout variants, unrelated route metadata, and exploratory notes.",
            ]
            * 10
        )
        events.append(
            AgentEvent(
                kind=EventKind.FILE_READ,
                source=source,
                text=repeated,
                created_at=4 + index,
            )
        )

    events.extend(
        [
            AgentEvent(
                kind=EventKind.DECISION,
                source="notes",
                text="Decision: active task is auth cookie delivery. QR and frontend noise should be omitted.",
                created_at=100,
            ),
            AgentEvent(
                kind=EventKind.USER,
                source="conversation",
                text="Now focus only on authentication cookies and explain the likely fix.",
                created_at=101,
            ),
        ]
    )
    return events
