from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cheap_talk import MAX_MESSAGE_CHARS, deliver_messages, parse_directed_messages
from run_pi_two_agents import build_policy_prompt


class CheapTalkTests(unittest.TestCase):
    def test_parser_keeps_distinct_messages_for_two_recipients(self) -> None:
        result = parse_directed_messages(
            [
                {"to": 1, "text": "I will clean."},
                {"to": 2, "text": "Please harvest."},
            ],
            sender=0,
            agent_count=3,
        )

        self.assertEqual(result["messages"], [{"to": 1, "text": "I will clean."}, {"to": 2, "text": "Please harvest."}])
        self.assertEqual(result["dropped"], [])

    def test_parser_drops_self_duplicate_and_overlength_messages(self) -> None:
        result = parse_directed_messages(
            [
                {"to": 0, "text": "self"},
                {"to": 1, "text": "first"},
                {"to": 1, "text": "duplicate"},
                {"to": 2, "text": "x" * (MAX_MESSAGE_CHARS + 1)},
            ],
            sender=0,
            agent_count=3,
        )

        self.assertEqual(result["messages"], [{"to": 1, "text": "first"}])
        self.assertEqual(len(result["dropped"]), 3)

    def test_messages_are_delivered_only_on_the_following_turn(self) -> None:
        current_inbox = [[], [], []]
        next_inbox = [[], [], []]
        deliver_messages(
            sender=0,
            step=8,
            messages=[{"to": 2, "text": "Clean now."}],
            next_inbox=next_inbox,
        )

        self.assertEqual(current_inbox[2], [])
        self.assertEqual(next_inbox[2], [{"from": 0, "sent_at_step": 8, "text": "Clean now."}])

    def test_cheap_talk_prompt_has_only_prior_inbox_and_directed_schema(self) -> None:
        grid = {"valid_actions": ["NOOP", "FIRE_CLEAN"], "agent_state": {"previous_action": "NOOP"}}
        inbox = [{"from": 2, "sent_at_step": 4, "text": "I will harvest."}]

        prompt, _ = build_policy_prompt("supervisor_cheap_talk", grid, [], [], incoming_messages=inbox, agent_index=0, agent_count=3)

        self.assertIn("INBOX_FROM_PREVIOUS_STEP_JSON", prompt)
        self.assertIn("CHEAP_TALK_ALLOWED_RECIPIENTS_JSON", prompt)
        self.assertIn("[1, 2]", prompt)
        self.assertIn('"sent_at_step": 4', prompt)
        self.assertIn('"messages"', prompt)
        self.assertIn('"to"', prompt)
        self.assertNotIn("global_active_dirt_cells", prompt)
        self.assertNotIn('Return exactly one JSON object and nothing else:\n\n{"action":"ONE_VALID_ACTION"}', prompt)
        self.assertNotIn("additional keys", prompt)
    def test_required_broadcast_prompt_requires_one_public_message_per_step(self) -> None:
        grid = {"valid_actions": ["NOOP", "FIRE_CLEAN"], "agent_state": {"previous_action": "NOOP"}}

        prompt, _ = build_policy_prompt("supervisor_required_broadcast", grid, [], [], incoming_messages=[], agent_index=0, agent_count=3)

        self.assertIn("MUST send one concise broadcast on every decision step", prompt)
        self.assertIn("same text to every listed recipient", prompt)
        self.assertIn('"to": 1', prompt)
        self.assertIn('"to": 2', prompt)


if __name__ == "__main__":
    unittest.main()
