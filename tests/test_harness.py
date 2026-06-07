from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_cache_lab.harness import build_prompt_candidates, run_model_harness, sample_result
from agentic_cache_lab.llm_client import LLMResult


ROOT = Path(__file__).resolve().parents[1]


class HarnessTest(unittest.TestCase):
    def test_build_prompt_candidates_returns_all_benchmark_modes(self) -> None:
        prompts = build_prompt_candidates(
            trace_path=ROOT / "examples" / "repo_debug_session.jsonl",
            query="Fix auth cookies. Omit QR scanner context.",
            objective="Measure routing.",
            max_prompt_tokens=320,
        )

        self.assertEqual([prompt.mode for prompt in prompts], ["raw", "routed", "routed_critical", "oracle_relevant"])
        self.assertGreater(prompts[0].token_estimate, prompts[1].stable_prefix_token_estimate)

    def test_echo_harness_runs_without_model_server(self) -> None:
        result = run_model_harness(
            trace_path=ROOT / "examples" / "repo_debug_session.jsonl",
            query="Fix auth cookies. Omit QR scanner context.",
            objective="Measure routing.",
            max_prompt_tokens=320,
            base_url="http://127.0.0.1:8080",
            model="local-model",
            api_key="local",
            runs=1,
            warmup=0,
            max_output_tokens=16,
            echo=True,
            timeout_seconds=1,
        )

        self.assertEqual(result["runs"], 1)
        self.assertEqual(len(result["samples"]), 4)
        self.assertIn("raw", result["summary"])
        self.assertIn("routed", result["summary"])
        self.assertIn("routed_critical", result["summary"])
        self.assertIn("oracle_relevant", result["summary"])

    def test_sample_result_prefers_runtime_completion_tokens(self) -> None:
        prompts = build_prompt_candidates(
            trace_path=ROOT / "examples" / "repo_debug_session.jsonl",
            query="Fix auth cookies.",
            objective="Measure routing.",
            max_prompt_tokens=320,
        )
        sample = sample_result(
            1,
            prompts[0],
            LLMResult(
                text="",
                latency_ms=1000,
                usage={
                    "completion_tokens": 12,
                    "prompt_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": 75},
                },
            ),
        )

        self.assertEqual(sample["output_tokens_estimate"], 12)
        self.assertEqual(sample["output_tokens_per_second_estimate"], 12.0)
        self.assertEqual(sample["runtime_prompt_tokens"], 100)
        self.assertEqual(sample["cached_prompt_tokens"], 75)
        self.assertEqual(sample["runtime_prompt_cache_hit_ratio"], 0.75)


if __name__ == "__main__":
    unittest.main()
