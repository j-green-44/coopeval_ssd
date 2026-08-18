from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mediation import (
    close_epoch,
    parse_mediation_choice,
    participant_fairness_summary,
    validate_mediator_plan,
)


class MediationTests(unittest.TestCase):
    def test_invalid_choice_retains_existing_enrolment(self) -> None:
        result = parse_mediation_choice('{"mediation_choice":"JOIN"}', was_enrolled=True, is_initial_epoch=False)

        self.assertFalse(result["valid"])
        self.assertTrue(result["enrolled"])
        self.assertIn("CONTINUE", result["error"])

    def test_two_participant_plan_requires_clean_and_harvest(self) -> None:
        raw = json.dumps(
            {
                "valid_for_steps": 50,
                "assignments": [
                    {"agent_index": 0, "role": "CLEAN", "objective": "clean"},
                    {"agent_index": 2, "role": "CLEAN", "objective": "clean"},
                ],
            }
        )

        result = validate_mediator_plan(raw, participants={0, 2}, interval_steps=50)

        self.assertFalse(result["valid"])
        self.assertIn("CLEAN", result["error"])
        self.assertIn("HARVEST", result["error"])

    def test_plan_rejects_assignment_to_nonparticipant(self) -> None:
        raw = json.dumps(
            {
                "valid_for_steps": 50,
                "assignments": [
                    {"agent_index": 0, "role": "CLEAN", "objective": "clean"},
                    {"agent_index": 1, "role": "HARVEST", "objective": "harvest"},
                ],
            }
        )

        result = validate_mediator_plan(raw, participants={0, 2}, interval_steps=50)

        self.assertFalse(result["valid"])
        self.assertIn("participants", result["error"])

    def test_close_epoch_records_assignments_and_realised_outcomes(self) -> None:
        epoch = {
            "epoch": 0,
            "start_step": 0,
            "participants": [0, 2],
            "assignments": [
                {"agent_index": 0, "role": "CLEAN"},
                {"agent_index": 2, "role": "HARVEST"},
            ],
        }
        outcomes = {
            0: {"reward": 2.0, "confirmed_clean_removals": 15},
            2: {"reward": 18.0, "confirmed_clean_removals": 0},
        }

        closed = close_epoch(epoch, end_step=49, outcomes=outcomes)

        self.assertEqual(closed["end_step"], 49)
        self.assertEqual(
            closed["realised_outcomes"],
            [
                {"agent_index": 0, "assigned_role": "CLEAN", "reward": 2.0, "confirmed_clean_removals": 15},
                {"agent_index": 2, "assigned_role": "HARVEST", "reward": 18.0, "confirmed_clean_removals": 0},
            ],
        )

    def test_fairness_summary_uses_all_completed_assignments(self) -> None:
        ledger = [
            {
                "epoch": 0,
                "assignments": [
                    {"agent_index": 0, "role": "CLEAN"},
                    {"agent_index": 2, "role": "HARVEST"},
                ],
                "realised_outcomes": [
                    {"agent_index": 0, "assigned_role": "CLEAN", "reward": 2.0, "confirmed_clean_removals": 15},
                    {"agent_index": 2, "assigned_role": "HARVEST", "reward": 18.0, "confirmed_clean_removals": 0},
                ],
            },
            {
                "epoch": 1,
                "assignments": [
                    {"agent_index": 0, "role": "HARVEST"},
                    {"agent_index": 2, "role": "FLEX"},
                ],
                "realised_outcomes": [
                    {"agent_index": 0, "assigned_role": "HARVEST", "reward": 12.0, "confirmed_clean_removals": 0},
                    {"agent_index": 2, "assigned_role": "FLEX", "reward": 3.0, "confirmed_clean_removals": 4},
                ],
            },
        ]

        summary = participant_fairness_summary(ledger, participants=[0, 2])

        self.assertEqual(
            summary,
            [
                {"agent_index": 0, "mediated_intervals": 2, "clean_intervals": 1, "harvest_intervals": 1, "flex_intervals": 0, "reward_while_mediated": 14.0, "confirmed_clean_removals_while_mediated": 15, "last_assigned_role": "HARVEST"},
                {"agent_index": 2, "mediated_intervals": 2, "clean_intervals": 0, "harvest_intervals": 1, "flex_intervals": 1, "reward_while_mediated": 21.0, "confirmed_clean_removals_while_mediated": 4, "last_assigned_role": "FLEX"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
