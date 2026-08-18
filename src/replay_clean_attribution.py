"""Replay Cleanup actions and attribute dirt removal to clean beams causally.

Supports either a completed trajectory.jsonl run or a session-only partial run.
For session-only runs, it uses the common completed action prefix exactly like
replay_pi_sessions.py.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MELTINGPOT = Path("/home/jack/phd/meltingpot")
TINY_VLM = Path("/home/jack/phd/tiny_cooperative_vlm/src")

import sys
sys.path[:0] = [
    str(ROOT / "src"),
    str(MELTINGPOT / "meltingpot_semantic_dataset" / "src"),
    str(TINY_VLM),
]

from meltingpot_semantic_dataset.environment import build_cleanup
from meltingpot_semantic_dataset.schema import load_schema
from run_pi_two_agents import normalise_event_item
from replay_pi_sessions import load_complete_actions


def _world_offset(orientation: int, forward: int, right: int) -> tuple[int, int]:
    if orientation == 0:  # N
        return -forward, right
    if orientation == 1:  # E
        return right, forward
    if orientation == 2:  # S
        return forward, -right
    if orientation == 3:  # W
        return -right, -forward
    raise ValueError(f"invalid Lab2D orientation code: {orientation}")


def beam_offsets(beam_length: int, beam_radius: int) -> set[tuple[int, int]]:
    """Approximate the Cleanup clean-beam footprint in local forward/right coords.

    Mirrors the Lua Cleaner/Tagger ray construction:
    - forward rays for each lateral offset in [-r, r] with length L - abs(offset)
    - one short side ray to the immediate left/right of length r
    """
    offsets: set[tuple[int, int]] = set()
    for right in range(-beam_radius, beam_radius + 1):
        max_forward = beam_length - abs(right)
        for forward in range(1, max_forward + 1):
            offsets.add((forward, right))
    for lateral in range(1, beam_radius + 1):
        offsets.add((0, -lateral))
        offsets.add((0, lateral))
    return offsets


def world_beam_cells(
    position_xy: tuple[int, int],
    orientation: int,
    world_shape: tuple[int, int],
    *,
    beam_length: int,
    beam_radius: int,
) -> set[tuple[int, int]]:
    x, y = position_xy
    height, width = world_shape
    cells: set[tuple[int, int]] = set()
    for forward, right in beam_offsets(beam_length, beam_radius):
        dy, dx = _world_offset(orientation, forward, right)
        yy, xx = y + dy, x + dx
        if 0 <= yy < height and 0 <= xx < width:
            cells.add((yy, xx))
    return cells


def load_actions(run_dir: Path, agent_count: int) -> list[list[str]]:
    trajectory = run_dir / "trajectory.jsonl"
    if trajectory.exists():
        rows = [json.loads(line) for line in trajectory.read_text(encoding="utf-8").splitlines() if line.strip()]
        actions_by_agent = [[] for _ in range(agent_count)]
        for row in rows:
            for agent in range(agent_count):
                actions_by_agent[agent].append(str(row["agents"][agent]["decision"]["action"]).upper())
        return actions_by_agent
    return load_complete_actions(run_dir, agent_count)


def removed_dirt_cells(pre_dirt: np.ndarray, post_dirt: np.ndarray) -> set[tuple[int, int]]:
    removed = np.argwhere((pre_dirt.astype(bool)) & (~post_dirt.astype(bool)))
    return {tuple(int(v) for v in item) for item in removed}


def find_agent_positions(ids: np.ndarray, orientations: np.ndarray, agent_count: int) -> tuple[list[tuple[int, int]], list[int]]:
    positions: list[tuple[int, int]] = []
    facing: list[int] = []
    for agent in range(agent_count):
        found = np.argwhere(ids == (agent + 1))
        if len(found) != 1:
            raise RuntimeError(f"expected exactly one position for agent {agent + 1}, found {len(found)}")
        y, x = (int(v) for v in found[0])
        positions.append((x, y))
        facing.append(int(orientations[y, x]))
    return positions, facing


def summarise_agent(step_rows: Iterable[dict]) -> dict:
    rows = list(step_rows)
    action_counts = Counter(row["action"] for row in rows)
    fire_rows = [row for row in rows if row["action"] == "FIRE_CLEAN"]
    return {
        "steps": len(rows),
        "actions": dict(action_counts),
        "fire_clean_actions": len(fire_rows),
        "strict_successes": sum(row["strict_success"] for row in fire_rows),
        "beam_removed_any": sum(row["beam_removed_any"] for row in fire_rows),
        "player_cleaned_events": sum(row["player_cleaned_event_count"] for row in fire_rows),
        "ambiguous_only_fires": sum((not row["strict_success"]) and row["shared_removed_count"] > 0 for row in fire_rows),
        "misses": sum((row["action"] == "FIRE_CLEAN") and row["removed_count"] == 0 for row in rows),
        "total_removed_cells_unambiguous": sum(row["removed_count"] for row in fire_rows),
        "total_removed_cells_shared": sum(row["shared_removed_count"] for row in fire_rows),
        "first_strict_success_steps": [row["step"] for row in fire_rows if row["strict_success"]][:20],
    }


def replay_attribution(
    run_dir: Path,
    agent_count: int,
    *,
    seed: int = 39,
    beam_length: int = 3,
    beam_radius: int = 1,
) -> Path:
    schema = load_schema()
    dirt_index = schema.index("dirt_live")
    actions_by_agent = load_actions(run_dir, agent_count)
    steps = min(len(actions) for actions in actions_by_agent)
    if steps == 0:
        raise ValueError("no actions available to replay")

    env = build_cleanup(agent_count, env_seed=seed)
    per_step_rows: list[dict] = []
    try:
        timestep = env.reset()
        for step in range(steps):
            pre_world = timestep.observation[0]["WORLD.SEMANTIC_GRID"]
            ids = timestep.observation[0]["WORLD.AGENT_ID_GRID"]
            orientations = timestep.observation[0]["WORLD.AGENT_ORIENTATION_GRID"]
            positions, facing = find_agent_positions(ids, orientations, agent_count)
            pre_dirt = pre_world[:, :, dirt_index]

            step_actions = [actions_by_agent[agent][step] for agent in range(agent_count)]
            beam_cells_by_agent: list[set[tuple[int, int]]] = []
            for agent, action in enumerate(step_actions):
                if action == "FIRE_CLEAN":
                    beam_cells_by_agent.append(
                        world_beam_cells(
                            positions[agent],
                            facing[agent],
                            pre_dirt.shape,
                            beam_length=beam_length,
                            beam_radius=beam_radius,
                        )
                    )
                else:
                    beam_cells_by_agent.append(set())

            from cleanup_vlm.actions import discover_action_mapping
            mapping = discover_action_mapping()
            timestep = env.step([mapping.index(action) for action in step_actions])
            post_world = timestep.observation[0]["WORLD.SEMANTIC_GRID"]
            post_dirt = post_world[:, :, dirt_index]
            removed = removed_dirt_cells(pre_dirt, post_dirt)

            all_beam_coverage: dict[tuple[int, int], list[int]] = {}
            for agent, cells in enumerate(beam_cells_by_agent):
                for cell in cells:
                    all_beam_coverage.setdefault(cell, []).append(agent)

            step_events = [(event_name, normalise_event_item(event_item)) for event_name, event_item in env.events()]
            cleaned_counts = {agent: 0 for agent in range(agent_count)}
            for event_name, event_item in step_events:
                if event_name != "player_cleaned" or not isinstance(event_item, dict):
                    continue
                player_index = int(event_item.get("player_index", -1))
                zero_based = player_index - 1
                if 0 <= zero_based < agent_count:
                    cleaned_counts[zero_based] += 1

            for agent, action in enumerate(step_actions):
                beam_cells = beam_cells_by_agent[agent]
                in_beam_pre = {cell for cell in beam_cells if pre_dirt[cell]}
                removed_in_beam = removed & beam_cells
                unambiguous_removed = {cell for cell in removed_in_beam if all_beam_coverage.get(cell) == [agent]}
                shared_removed = removed_in_beam - unambiguous_removed
                per_step_rows.append({
                    "step": step,
                    "agent_index": agent,
                    "action": action,
                    "position_xy": list(positions[agent]),
                    "orientation": facing[agent],
                    "beam_cells": [list(cell) for cell in sorted(beam_cells)],
                    "pre_dirt_in_beam": [list(cell) for cell in sorted(in_beam_pre)],
                    "removed_cells_in_beam": [list(cell) for cell in sorted(removed_in_beam)],
                    "strict_removed_cells": [list(cell) for cell in sorted(unambiguous_removed)],
                    "shared_removed_cells": [list(cell) for cell in sorted(shared_removed)],
                    "beam_had_live_dirt_pre_step": bool(in_beam_pre),
                    "beam_removed_any": bool(removed_in_beam),
                    "strict_success": bool(unambiguous_removed),
                    "removed_count": len(unambiguous_removed),
                    "shared_removed_count": len(shared_removed),
                    "player_cleaned_event_count": cleaned_counts[agent],
                    "global_removed_cells_this_step": [list(cell) for cell in sorted(removed)],
                })
    finally:
        env.close()

    output_dir = run_dir / "replay_clean_attribution"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "per_step_agent_attribution.jsonl"
    detail_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in per_step_rows), encoding="utf-8")

    summary = {
        "status": "completed",
        "seed": seed,
        "agent_count": agent_count,
        "replayed_steps": steps,
        "beam_length": beam_length,
        "beam_radius": beam_radius,
        "source": "trajectory.jsonl" if (run_dir / "trajectory.jsonl").exists() else "pi session common prefix",
        "agents": {},
    }
    for agent in range(agent_count):
        summary["agents"][str(agent)] = summarise_agent(row for row in per_step_rows if row["agent_index"] == agent)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--beam-length", type=int, default=3)
    parser.add_argument("--beam-radius", type=int, default=1)
    args = parser.parse_args()
    print(replay_attribution(args.run_dir, args.agents, seed=args.seed, beam_length=args.beam_length, beam_radius=args.beam_radius))


if __name__ == "__main__":
    main()
