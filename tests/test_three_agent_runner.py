from __future__ import annotations

import unittest
from pathlib import Path


class ThreeAgentRunnerTests(unittest.TestCase):
    def test_runner_exposes_configurable_agent_count(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src" / "run_pi_two_agents.py").read_text()
        self.assertIn("agent_count: int", source)
        self.assertIn("build_cleanup(agent_count, env_seed=seed)", source)
        self.assertIn("range(agent_count)", source)
        self.assertIn('parser.add_argument("--agents", type=int, default=2)', source)

    def test_runner_accepts_explicit_provider_model_and_run_label(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src" / "run_pi_two_agents.py").read_text()
        self.assertIn("provider: str", source)
        self.assertIn("model: str", source)
        self.assertIn("run_label: str", source)
        self.assertIn('parser.add_argument("--provider"', source)
        self.assertIn('parser.add_argument("--model"', source)
        self.assertIn('parser.add_argument("--run-label"', source)
        self.assertIn('"provider": provider', source)
        self.assertIn('"model": model', source)

    def test_runner_accepts_per_agent_model_assignments(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src" / "run_pi_two_agents.py").read_text()
        self.assertIn("agent_models: list[str] | None", source)
        self.assertIn("agent_models[index]", source)
        self.assertIn('parser.add_argument("--agent-models"', source)
        self.assertIn('"model": agent_models[index]', source)

    def test_runner_pins_env_seed_and_logs_clean_events(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src" / "run_pi_two_agents.py").read_text()
        self.assertIn("build_cleanup(agent_count, env_seed=seed)", source)
        self.assertIn("normalise_event_item(event_item)", source)
        self.assertIn('"player_cleaned_count"', source)
        self.assertIn('"env_seed": seed', source)
        self.assertIn('not isinstance(event_item, dict)', source)
        self.assertIn("def json_safe", source)
        self.assertIn("bytes, bytearray", source)

    def test_runner_persists_partial_trajectory_and_video_after_provider_failure(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src" / "run_pi_two_agents.py").read_text()
        self.assertIn("except Exception as error:", source)
        self.assertIn('"status": "failed" if failure else "completed"', source)
        self.assertIn('"completed_steps": len(records)', source)
        self.assertIn("if failure is not None:", source)


if __name__ == "__main__":
    unittest.main()
