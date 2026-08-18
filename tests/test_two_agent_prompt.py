from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_pi_two_agents import (
    MEDIATION_INTERVAL_STEPS,
    MINIMAL_NO_AIM_INSTRUCTIONS,
    SUPERVISOR_STRATEGY_INSTRUCTIONS,
    build_mediator_prompt,
    build_policy_prompt,
)


class TwoAgentPromptTests(unittest.TestCase):
    def test_directional_task_prior_is_orientation_relative_without_global_coordinates(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src" / "run_pi_two_agents.py").read_text()
        self.assertIn("The water is generally in the global north of the map.", source)
        self.assertIn("the top is your current forward direction, not necessarily global north", source)
        self.assertIn("facing EAST: water is generally to your left", source)
        self.assertIn("Apples are generally south of the water.", source)
        self.assertIn("Walls block movement; river, sand, and grass are traversable terrain.", source)
        self.assertIn("Walk into the river to reach and clean live dirt.", source)
        self.assertNotIn("Walls and river block movement.", source)
        self.assertNotIn("water_world_coordinate", source)

    def test_minimal_no_aim_prompt_preserves_grid_and_memory_without_strategy_advice(self) -> None:
        grid = {
            "schema_version": "cleanup_grid_context_v1",
            "frame": 0,
            "self": {"row": 5, "column": 5, "orientation": "NORTH"},
            "view": {"height": 11, "width": 11},
            "cells": [],
            "valid_actions": ["NOOP", "FORWARD"],
            "agent_state": {"previous_action": "NOOP", "cumulative_reward": 0.0},
            "local_affordances": {"movement_affordances": {"forward": "free"}},
        }
        history = [{"step": 0, "action": "FORWARD", "reward": 0.0, "outcome": {"movement_outcome": "blocked"}}]

        prompt, policy_grid = build_policy_prompt("minimal_no_aim", grid, history)

        self.assertIn(MINIMAL_NO_AIM_INSTRUCTIONS, prompt)
        self.assertIn("GRID_CONTEXT_JSON", prompt)
        self.assertIn("POLICY_VISIBLE_RECENT_HISTORY_JSON", prompt)
        self.assertIn('"movement_outcome": "blocked"', prompt)
        self.assertNotIn("maximise", prompt.lower())
        self.assertNotIn("when ready_to_shoot", prompt.lower())
        self.assertNotIn("prefer movement", prompt.lower())
        self.assertNotIn("walk into the river", prompt.lower())
        self.assertNotIn("do not assume", prompt.lower())
        self.assertEqual(policy_grid, grid)

    def test_supervisor_strategy_prompt_uses_controller_json_action_schema(self) -> None:
        grid = {"valid_actions": ["NOOP", "FIRE_CLEAN"], "agent_state": {"previous_action": "NOOP"}}
        prompt, policy_grid = build_policy_prompt("supervisor_strategy", grid, [])

        self.assertIn(SUPERVISOR_STRATEGY_INSTRUCTIONS, prompt)
        self.assertIn('{"action":"ONE_VALID_ACTION"}', prompt)
        self.assertNotIn("exactly one integer", prompt.lower())
        self.assertIn("GRID_CONTEXT_JSON", prompt)
        self.assertEqual(policy_grid, grid)

    def test_repetition_prompt_exposes_only_supplied_past_partner_history(self) -> None:
        grid = {"valid_actions": ["NOOP", "FIRE_CLEAN"], "agent_state": {"previous_action": "NOOP"}}
        partner_history = [{"step": 4, "agents": [{"agent_index": 1, "action": "FIRE_CLEAN", "reward": 0.0, "confirmed_clean_removals": 1}]}]

        prompt, policy_grid = build_policy_prompt("supervisor_repetition", grid, [], partner_history)

        self.assertIn(SUPERVISOR_STRATEGY_INSTRUCTIONS, prompt)
        self.assertIn("PARTNER_VISIBLE_RECENT_HISTORY_JSON", prompt)
        self.assertIn('"confirmed_clean_removals": 1', prompt)
        self.assertNotIn("global_active_dirt_cells", prompt)
        self.assertEqual(policy_grid, grid)

    def test_mediation_agent_and_mediator_prompts_keep_assignments_causal(self) -> None:
        grid = {"valid_actions": ["NOOP", "FIRE_CLEAN"], "agent_state": {"previous_action": "NOOP"}}
        assignment = {"epoch": 1, "role": "HARVEST", "objective": "Prioritise locally observable apples.", "start_step": 50, "end_step": 99}
        prompt, _ = build_policy_prompt("supervisor_mediation", grid, [], [], mediation_assignment=assignment, mediation_status={"enrolled": True})
        mediator_prompt = build_mediator_prompt(
            participants=[0, 2],
            ledger=[{"epoch": 0, "assignments": [{"agent_index": 0, "role": "CLEAN"}]}],
            fairness_summary=[{"agent_index": 0, "clean_intervals": 1}],
        )

        self.assertIn("MEDIATION_ASSIGNMENT_JSON", prompt)
        self.assertIn('"role": "HARVEST"', prompt)
        self.assertIn("MEDIATION_ASSIGNMENT_LEDGER_JSON", mediator_prompt)
        self.assertIn("MEDIATION_PARTICIPANT_FAIRNESS_SUMMARY_JSON", mediator_prompt)
        self.assertIn(str(MEDIATION_INTERVAL_STEPS), mediator_prompt)
        self.assertNotIn("global_active_dirt_cells", mediator_prompt)
        self.assertNotIn("GRID_CONTEXT_JSON", mediator_prompt)


if __name__ == "__main__":
    unittest.main()
