from __future__ import annotations

import argparse
import json
from pathlib import Path

from .event_log import load_jsonl
from .harness import run_model_harness
from .models import estimate_tokens
from .packer import PackConfig, PromptPacker
from .segment_store import SegmentStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic context cache experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Compare raw history with routed context")
    benchmark.add_argument("--trace", type=Path, required=True)
    benchmark.add_argument("--query", default="Fix the authentication cookie bug. Omit unrelated QR scanner context.")
    benchmark.add_argument("--objective", default="Resolve the active coding task with minimal useful context.")
    benchmark.add_argument("--max-prompt-tokens", type=int, default=320)
    benchmark.add_argument("--json", action="store_true", help="Emit JSON only")

    harness = subparsers.add_parser("model-harness", help="Run raw vs routed prompts against a local model server")
    harness.add_argument("--trace", type=Path, required=True)
    harness.add_argument("--query", default="Fix the authentication cookie bug. Omit unrelated QR scanner context.")
    harness.add_argument("--objective", default="Resolve the active coding task with minimal useful context.")
    harness.add_argument("--max-prompt-tokens", type=int, default=320)
    harness.add_argument("--base-url", default="http://127.0.0.1:8080")
    harness.add_argument("--model", default="local-model")
    harness.add_argument("--api-key", default="local")
    harness.add_argument("--runs", type=int, default=3)
    harness.add_argument("--warmup", type=int, default=1)
    harness.add_argument("--max-output-tokens", type=int, default=32)
    harness.add_argument("--timeout-seconds", type=float, default=300)
    harness.add_argument("--echo", action="store_true", help="Use the offline echo client instead of a model server")

    args = parser.parse_args()
    if args.command == "benchmark":
        result = run_benchmark(args.trace, args.query, args.objective, args.max_prompt_tokens)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print_report(result)
    elif args.command == "model-harness":
        result = run_model_harness(
            trace_path=args.trace,
            query=args.query,
            objective=args.objective,
            max_prompt_tokens=args.max_prompt_tokens,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            runs=args.runs,
            warmup=args.warmup,
            max_output_tokens=args.max_output_tokens,
            echo=args.echo,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, indent=2, sort_keys=True))


def run_benchmark(trace_path: Path, query: str, objective: str, max_prompt_tokens: int) -> dict[str, object]:
    events = load_jsonl(trace_path)
    store = SegmentStore()
    segments = store.add_events(events)
    packer = PromptPacker(config=PackConfig(max_prompt_tokens=max_prompt_tokens))
    packed = packer.pack(segments, query=query, objective=objective)

    raw_history = "\n\n".join(f"{event.kind.value.upper()} {event.source}\n{event.text}" for event in events)
    raw_prompt = f"{raw_history}\n\nUser query:\n{query}"
    raw_tokens = estimate_tokens(raw_prompt)
    saved_tokens = raw_tokens - packed.token_estimate
    saved_ratio = saved_tokens / raw_tokens if raw_tokens else 0.0

    return {
        "trace": str(trace_path),
        "events": len(events),
        "segments": len(segments),
        "raw_tokens_estimate": raw_tokens,
        "routed_tokens_estimate": packed.token_estimate,
        "stable_prefix_tokens_estimate": packed.stable_token_estimate,
        "saved_tokens_estimate": saved_tokens,
        "saved_ratio": round(saved_ratio, 4),
        "included_segments": [
            {
                "id": item.segment.id,
                "source": item.segment.source,
                "kind": item.segment.kind.value,
                "labels": list(item.segment.labels),
                "tokens": item.segment.token_count,
                "score": round(item.score, 4),
            }
            for item in packed.included
        ],
        "omitted_segments": [
            {
                "id": segment.id,
                "source": segment.source,
                "kind": segment.kind.value,
                "labels": list(segment.labels),
                "tokens": segment.token_count,
            }
            for segment in packed.omitted
        ],
    }


def print_report(result: dict[str, object]) -> None:
    print("Agentic Cache Lab benchmark")
    print(f"trace: {result['trace']}")
    print(f"events: {result['events']} | segments: {result['segments']}")
    print(f"raw tokens estimate: {result['raw_tokens_estimate']}")
    print(f"routed tokens estimate: {result['routed_tokens_estimate']}")
    print(f"stable prefix tokens estimate: {result['stable_prefix_tokens_estimate']}")
    print(f"saved tokens estimate: {result['saved_tokens_estimate']} ({result['saved_ratio']:.1%})")
    print("\nincluded segments:")
    for item in result["included_segments"]:
        labels = ",".join(item["labels"])
        print(f"- {item['source']} [{item['kind']}/{labels}] score={item['score']} tokens={item['tokens']}")
    print("\nomitted segments:")
    for item in result["omitted_segments"]:
        labels = ",".join(item["labels"])
        print(f"- {item['source']} [{item['kind']}/{labels}] tokens={item['tokens']}")


if __name__ == "__main__":
    main()
