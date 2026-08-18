"""Persistent-session Pi policy rollout over exact egocentric Cleanup grids."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MELTINGPOT = Path("/home/jack/phd/meltingpot")
TINY_VLM = Path("/home/jack/phd/tiny_cooperative_vlm/src")
sys.path[:0] = [str(ROOT / "src"), str(MELTINGPOT / "meltingpot_semantic_dataset" / "src"), str(TINY_VLM)]

from action_outcome import build_action_outcome
from cleanup_vlm.actions import discover_action_mapping
from cleanup_vlm.video import write_video
from grid_context import build_grid_message
from grid_prompt import build_request_text
from meltingpot_semantic_dataset.egocentric import ViewGeometry, world_to_egocentric_semantics
from meltingpot_semantic_dataset.environment import build_cleanup
from meltingpot_semantic_dataset.schema import load_schema
from pi_client import decide

# Keep the per-turn prompt small enough for a 16k server context. Persistent
# policy-visible state is carried as structured recent history, while Pi session
# files rotate every few calls instead of accumulating full 11x11 grids forever.
PI_GRID_INSTRUCTIONS = """You control one Cleanup avatar to maximise its cumulative reward. You receive only the exact current 11x11 egocentric symbolic grid and policy-visible action/reward history. Walls and river block movement. Apples are individual rewards; FIRE_CLEAN affects only pollution directly ahead. If an action did not change the local situation or reward, explore with a justified different action. Return JSON only with action, public_message, and intent."""
SESSION_DECISIONS = 3
HISTORY_LIMIT = 12


def parse_action(raw: str, valid_actions: set[str]) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        action = str(value["action"]).upper()
        if action not in valid_actions:
            raise ValueError(f"unsupported action: {action}")
        return {"action": action, "public_message": str(value.get("public_message", ""))[:160], "intent": str(value.get("intent", ""))[:160]}
    except Exception as error:
        return {"action": "NOOP", "public_message": "", "intent": "fallback after invalid Pi output", "parse_error": str(error)[:300]}


def run(seed: int, steps: int, output: Path, record_video: bool) -> Path:
    from run_grid_context import INSTRUCTIONS
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / f"pi_seed_{seed}_steps_{steps}"
    run_dir.mkdir(parents=True, exist_ok=True)
    session_dir = run_dir / "pi_sessions"; session_dir.mkdir(exist_ok=True)
    schema, geometry = load_schema(), ViewGeometry(5, 5, 9, 1)
    mapping, valid_actions = discover_action_mapping(), set(discover_action_mapping().name_to_index)
    env = build_cleanup(2)
    records: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    try:
        timestep = env.reset()
        previous_action, previous_reward, cumulative, previous_outcome = "NOOP", 0.0, 0.0, {}
        for step in range(steps):
            if record_video:
                frames.append(np.array(timestep.observation[0]["WORLD.RGB"], copy=True))
            world = timestep.observation[0]
            semantic, ids, orientations = world["WORLD.SEMANTIC_GRID"], world["WORLD.AGENT_ID_GRID"], world["WORLD.AGENT_ORIENTATION_GRID"]
            found = np.argwhere(ids == 1)
            if len(found) != 1:
                raise RuntimeError("controlled avatar is unavailable in world state")
            y, x = (int(v) for v in found[0]); orientation = int(orientations[y, x])
            local = world_to_egocentric_semantics(semantic, (x, y), orientation, geometry)
            local_ids = world_to_egocentric_semantics(ids[:, :, None], (x, y), orientation, geometry)[:, :, 0]
            local_orientations = world_to_egocentric_semantics(orientations[:, :, None], (x, y), orientation, geometry)[:, :, 0]
            grid = build_grid_message(semantic=local, channel_names=schema.names, agent_ids=local_ids, orientations=local_orientations, self_id=1, frame=step, previous_action=previous_action, previous_reward=previous_reward, cumulative_reward=cumulative, previous_outcome=previous_outcome, ready_to_shoot=bool(world.get("READY_TO_SHOOT", False)), valid_actions=tuple(sorted(valid_actions)), orientation_codes=schema.orientation_codes)
            recent = [{"step": row["decision_step"], "action": row["decision"]["action"], "reward": row["reward"], "intent": row["decision"].get("intent", "")} for row in records[-HISTORY_LIMIT:]]
            prompt = build_request_text(PI_GRID_INSTRUCTIONS, grid) + "\n\nPOLICY_VISIBLE_RECENT_HISTORY_JSON:\n" + json.dumps(recent, sort_keys=True)
            session = session_dir / f"context_{step // SESSION_DECISIONS:03d}.jsonl"
            raw, latency_ms = decide(session, prompt)
            decision = parse_action(raw, valid_actions)
            action = decision["action"]
            timestep = env.step([mapping.index(action), mapping.index("NOOP")])
            next_world = timestep.observation[0]
            next_ids = next_world["WORLD.AGENT_ID_GRID"]
            next_found = np.argwhere(next_ids == 1)
            if len(next_found) != 1:
                raise RuntimeError("controlled avatar is unavailable after action")
            next_y, next_x = (int(v) for v in next_found[0])
            next_orientation = int(next_world["WORLD.AGENT_ORIENTATION_GRID"][next_y, next_x])
            next_local = world_to_egocentric_semantics(next_world["WORLD.SEMANTIC_GRID"], (next_x, next_y), next_orientation, geometry)
            dirt_index = schema.names.index("dirt_live")
            previous_outcome = build_action_outcome(
                action, moved=(next_x, next_y) != (x, y), local_view_changed=not np.array_equal(local, next_local),
                dirt_before=int(local[:, :, dirt_index].sum()), dirt_after=int(next_local[:, :, dirt_index].sum()),
            )
            previous_action, previous_reward = action, float(timestep.reward[0] or 0.0)
            cumulative += previous_reward
            records.append({"seed": seed, "decision_step": step, "grid_context": grid, "decision": decision, "raw_pi_response": raw, "latency_ms": latency_ms, "reward": previous_reward, "cumulative_reward": cumulative})
    finally:
        env.close()
    if record_video:
        video = run_dir / "videos" / "episode_000.lossless.mkv"; video.parent.mkdir(parents=True, exist_ok=True); write_video(frames, video, fps=10)
    (run_dir / "trajectory.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"status": "completed", "seed": seed, "steps": steps, "agent": "Pi bounded persistent context", "model": "Qwen3-VL-8B-Instruct via vLLM", "decision_count": len(records), "total_reward": cumulative, "policy_input": "egocentric symbolic grid plus policy-visible recent history only", "pi_session_rotation": {"decisions_per_session": SESSION_DECISIONS, "history_limit": HISTORY_LIMIT}}, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=14); parser.add_argument("--steps", type=int, default=200); parser.add_argument("--output", type=Path, default=ROOT / "outputs"); parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args(); print(run(args.seed, args.steps, args.output, args.record_video))


if __name__ == "__main__":
    main()
