from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import AgentEvent, EventKind


def load_jsonl(path: Path) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            kind = EventKind(raw["kind"])
            text = str(raw["text"])
        except KeyError as exc:
            raise ValueError(f"{path}:{line_number} missing required field {exc}") from exc
        events.append(
            AgentEvent(
                kind=kind,
                text=text,
                source=str(raw.get("source", "conversation")),
                created_at=float(raw.get("created_at", line_number)),
                metadata={str(k): str(v) for k, v in raw.get("metadata", {}).items()},
            )
        )
    return events


def dump_jsonl(events: Iterable[AgentEvent], path: Path) -> None:
    lines = []
    for event in events:
        lines.append(
            json.dumps(
                {
                    "kind": event.kind.value,
                    "source": event.source,
                    "text": event.text,
                    "created_at": event.created_at,
                    "metadata": event.metadata,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
