from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResult:
    text: str
    latency_ms: float
    usage: dict[str, object]


class EchoClient:
    """Offline client for benchmark plumbing before a real local model is attached."""

    def complete(self, prompt: str, max_tokens: int | None = None, seed: int | None = None) -> LLMResult:
        started = time.perf_counter()
        lines = prompt.splitlines()
        preview = "\n".join(lines[-8:])
        return LLMResult(
            text=f"[echo]\n{preview}",
            latency_ms=(time.perf_counter() - started) * 1000,
            usage={"prompt_chars": len(prompt)},
        )


class OpenAICompatibleClient:
    """Minimal client for local vLLM/SGLang servers exposing /v1/chat/completions."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "local",
        timeout_seconds: float = 300,
        temperature: float = 0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def complete(self, prompt: str, max_tokens: int | None = None, seed: int | None = None) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM server returned HTTP {exc.code}: {detail}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        message = raw["choices"][0]["message"]
        text = message.get("content") or message.get("reasoning_content") or ""
        return LLMResult(text=text, latency_ms=latency_ms, usage=raw.get("usage", {}))
