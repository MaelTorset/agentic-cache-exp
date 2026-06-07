from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_cache_lab.bug_resolution_benchmark import (
    build_bug_resolution_prompts,
    build_native_generation_plan,
    run_bug_resolution_quality_benchmark,
    score_bug_answer,
)
from agentic_cache_lab.fixture_repo import build_fixture_repo_events


ROOT = Path(__file__).resolve().parents[1]


class BugResolutionBenchmarkTest(unittest.TestCase):
    def test_score_bug_answer_rewards_expected_fix(self) -> None:
        score = score_bug_answer(
            "Patch backend/src/auth/cookie.ts. In development/test use SameSite=Lax without Secure. "
            "In production keep SameSite=None with Secure."
        )

        self.assertEqual(score["score"], 4)
        self.assertTrue(score["file_hit"])
        self.assertTrue(score["cause_hit"])
        self.assertTrue(score["dev_lax_hit"])
        self.assertTrue(score["prod_secure_hit"])

    def test_build_bug_resolution_prompts_compares_full_and_routed(self) -> None:
        events = build_fixture_repo_events(ROOT / "examples" / "fixtures" / "shopbug-repo")
        prompts = build_bug_resolution_prompts(events)

        self.assertEqual([prompt.mode for prompt in prompts], ["full_noise", "routed_prompt"])
        self.assertGreater(prompts[0].token_estimate, prompts[1].token_estimate)

    def test_quality_benchmark_echo_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_bug_resolution_quality_benchmark(
                repo_root=ROOT / "examples" / "fixtures" / "shopbug-repo",
                base_url="http://127.0.0.1:8082",
                model="local",
                runs=1,
                warmup=0,
                max_output_tokens=8,
                temperature=0.2,
                seed_base=1,
                output_trace=Path(directory) / "trace.jsonl",
                echo=True,
                timeout_seconds=1,
            )

        self.assertEqual(result["benchmark"], "fixture_repo_bug_resolution_quality")
        self.assertEqual(len(result["samples"]), 2)
        self.assertIn("full_noise", result["summary"])
        self.assertIn("routed_prompt", result["summary"])

    def test_native_generation_plan_contains_generate_ops(self) -> None:
        events = build_fixture_repo_events(ROOT / "examples" / "fixtures" / "shopbug-repo")
        plan = build_native_generation_plan(events, max_tokens=12)

        self.assertEqual(plan["metadata"]["mode"], "native_kv_branch_generation")
        self.assertEqual(plan["metadata"]["max_tokens"], 12)
        self.assertEqual([op["op"] for op in plan["ops"]].count("generate"), 2)
        self.assertEqual([op["label"] for op in plan["ops"] if op["op"] == "generate"], ["kv_branch_auth", "scratch_auth"])


if __name__ == "__main__":
    unittest.main()
