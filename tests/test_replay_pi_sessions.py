from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_pi_sessions import load_complete_actions


def assistant_message(action: str) -> str:
    return json.dumps({
        "type": "message",
        "message": {"role": "assistant", "content": [{"type": "text", "text": json.dumps({"action": action})}]},
    })


class ReplayPiSessionTests(unittest.TestCase):
    def test_uses_only_the_common_completed_prefix_across_agents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for agent, actions in enumerate((("FORWARD", "TURN_LEFT", "NOOP"), ("STEP_LEFT", "FIRE_CLEAN"), ("BACKWARD", "TURN_RIGHT"))):
                sessions = root / f"pi_agent_{agent}_sessions"
                sessions.mkdir()
                (sessions / "context_000.jsonl").write_text("\n".join(assistant_message(action) for action in actions) + "\n")
            self.assertEqual(
                load_complete_actions(root, agent_count=3),
                [["FORWARD", "TURN_LEFT"], ["STEP_LEFT", "FIRE_CLEAN"], ["BACKWARD", "TURN_RIGHT"]],
            )


if __name__ == "__main__":
    unittest.main()
