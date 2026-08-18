from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pi_client import build_pi_command, decide, raise_for_pi_error


class PiClientTests(unittest.TestCase):
    def test_command_uses_local_model_persistent_session_and_no_tools(self) -> None:
        command = build_pi_command(Path("/tmp/session.jsonl"), "choose an action")
        self.assertIn("--session", command)
        self.assertIn("/tmp/session.jsonl", command)
        self.assertIn("--provider", command)
        self.assertIn("cleanup-local", command)
        self.assertIn("--model", command)
        self.assertIn("Qwen3-VL-8B-Instruct", command)
        self.assertIn("--no-tools", command)
        self.assertIn("--print", command)
        self.assertNotIn("bash", command)

    def test_command_can_target_codex_oauth_model_without_changing_policy_prompt(self) -> None:
        command = build_pi_command(
            Path("/tmp/session.jsonl"),
            "choose an action",
            provider="openai-codex",
            model="gpt-5.6",
        )
        self.assertIn("openai-codex", command)
        self.assertIn("gpt-5.6", command)
        self.assertIn("--no-tools", command)
        self.assertIn("--no-context-files", command)
        self.assertIn("--no-extensions", command)
        self.assertIn("--no-skills", command)
        self.assertIn("--system-prompt", command)
        self.assertIn("--print", command)

    def test_provider_error_reports_stderr_without_echoing_the_prompt(self) -> None:
        error = subprocess.CalledProcessError(1, ["pi", "--print", "sensitive prompt"], stderr="Codex overloaded")
        with self.assertRaisesRegex(RuntimeError, "Codex overloaded") as raised:
            raise_for_pi_error(error)
        self.assertNotIn("sensitive prompt", str(raised.exception))

    def test_default_timeout_allows_slow_codex_response(self) -> None:
        self.assertEqual(decide.__defaults__[0], 300.0)


if __name__ == "__main__":
    unittest.main()
