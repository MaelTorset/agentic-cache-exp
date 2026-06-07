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


def build_long_coding_task_events(noise_files: int = 12) -> list[AgentEvent]:
    """Create a coding-shaped trace with many irrelevant file reads and one auth bug."""
    events = [
        AgentEvent(
            kind=EventKind.USER,
            source="conversation",
            text=(
                "We are debugging a monorepo. The agent explores QR onboarding, billing, analytics, "
                "frontend settings, and auth before solving a failing cookie test."
            ),
            created_at=1,
        ),
    ]

    topics = ["qr", "frontend", "billing", "analytics"]
    for index in range(noise_files):
        topic = topics[index % len(topics)]
        source = f"app/src/{topic}/noise_file_{index}.ts"
        text = " ".join(
            [
                f"Unrelated {topic} implementation file {index}.",
                "It contains component state, event names, copy variants, helper functions,",
                "and integration details that do not affect authentication cookies.",
            ]
            * 8
        )
        events.append(
            AgentEvent(
                kind=EventKind.FILE_READ,
                source=source,
                text=text,
                created_at=2 + index,
            )
        )

    events.extend(
        [
            AgentEvent(
                kind=EventKind.FILE_READ,
                source="backend/src/auth/cookie.ts",
                text=(
                    "export function buildSessionCookie(token, env) { "
                    "const secure = env === 'production'; "
                    "return `session=${token}; Path=/; HttpOnly; SameSite=None${secure ? '; Secure' : ''}`; "
                    "} "
                    "Bug: in local HTTP, SameSite=None without Secure is rejected by modern browsers."
                ),
                created_at=100,
            ),
            AgentEvent(
                kind=EventKind.FILE_READ,
                source="backend/tests/auth-cookie.test.ts",
                text=(
                    "Test failure: login returns 200 but /api/me remains 401 in local development. "
                    "Expected Set-Cookie to use SameSite=Lax without Secure when env is development. "
                    "Expected production cookies to keep SameSite=None with Secure."
                ),
                created_at=101,
            ),
            AgentEvent(
                kind=EventKind.ERROR,
                source="logs/auth-cookie-test.log",
                text=(
                    "FAIL auth-cookie.test.ts: browser rejected session cookie. "
                    "Observed Set-Cookie: session=abc; Path=/; HttpOnly; SameSite=None. "
                    "Reason: SameSite=None requires Secure, but local test uses http://localhost."
                ),
                created_at=102,
            ),
            AgentEvent(
                kind=EventKind.DECISION,
                source="notes",
                text=(
                    "Decision: active coding task is backend auth cookie generation. "
                    "Forget QR, frontend, billing, and analytics files unless the task switches back."
                ),
                created_at=103,
            ),
            AgentEvent(
                kind=EventKind.USER,
                source="conversation",
                text=(
                    "Now solve only the auth cookie bug. Identify the file to patch, root cause, "
                    "and exact behavior change."
                ),
                created_at=104,
            ),
        ]
    )
    return events
