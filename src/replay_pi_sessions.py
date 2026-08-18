"""Replay the common completed action prefix from Pi session logs into a review video."""
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


def load_complete_actions(run_dir: Path, agent_count: int) -> list[list[str]]:
    """Extract valid JSON action responses, truncating to every agent's shared prefix."""
    action_lists: list[list[str]] = []
    for agent in range(agent_count):
        actions: list[str] = []
        sessions = run_dir / f"pi_agent_{agent}_sessions"
        for session in sorted(sessions.glob("context_*.jsonl")):
            for line in session.read_text(encoding="utf-8").splitlines():
                record: dict[str, Any] = json.loads(line)
                message = record.get("message", {})
                if record.get("type") != "message" or message.get("role") != "assistant":
                    continue
                for part in message.get("content", []):
                    if part.get("type") != "text":
                        continue
                    try:
                        action = str(json.loads(part["text"])["action"]).upper()
                    except (KeyError, TypeError, json.JSONDecodeError):
                        continue
                    actions.append(action)
        action_lists.append(actions)
    completed_steps = min((len(actions) for actions in action_lists), default=0)
    if completed_steps == 0:
        raise ValueError("No common completed action prefix found in Pi session logs")
    return [actions[:completed_steps] for actions in action_lists]


def replay(run_dir: Path, agent_count: int, fps: int = 10) -> Path:
    sys.path[:0] = [
        str(ROOT / "src"),
        str(MELTINGPOT / "meltingpot_semantic_dataset" / "src"),
        str(TINY_VLM),
    ]
    from cleanup_vlm.actions import discover_action_mapping
    from cleanup_vlm.video import write_video
    from meltingpot_semantic_dataset.environment import build_cleanup

    actions_by_agent = load_complete_actions(run_dir, agent_count)
    steps = len(actions_by_agent[0])
    mapping = discover_action_mapping()
    valid_actions = set(mapping.name_to_index)
    invalid = sorted({action for actions in actions_by_agent for action in actions if action not in valid_actions})
    if invalid:
        raise ValueError(f"Session logs contain unsupported actions: {invalid}")

    frames: list[np.ndarray] = []
    env = build_cleanup(agent_count)
    try:
        timestep = env.reset()
        for step in range(steps):
            frames.append(np.array(timestep.observation[0]["WORLD.RGB"], copy=True))
            timestep = env.step([mapping.index(actions_by_agent[agent][step]) for agent in range(agent_count)])
    finally:
        env.close()

    video = run_dir / "videos" / f"replay_steps_000_to_{steps - 1:03d}.lossless.mkv"
    video.parent.mkdir(parents=True, exist_ok=True)
    write_video(frames, video, fps=fps)
    manifest = {
        "status": "completed",
        "artifact_type": "deterministic replay of incomplete rollout",
        "source": "Pi session action responses",
        "agent_count": agent_count,
        "replayed_steps": steps,
        "seed": 39,
        "video": str(video.relative_to(run_dir)),
        "fps": fps,
    }
    (run_dir / "replay_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    print(replay(args.run_dir, args.agents, args.fps))


if __name__ == "__main__":
    main()
