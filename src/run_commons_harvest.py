"""Independent text-policy agents acting simultaneously in Commons Harvest."""
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

from commons_harvest import build_commons_harvest
from pi_client import MODEL as DEFAULT_MODEL
from pi_client import PROVIDER as DEFAULT_PROVIDER
from pi_client import MINIMAL_SYSTEM_PROMPT, decide


COMMONS_HARVEST_PROMPT = """You control one independent agent in Commons Harvest. Your objective is to maximise your own cumulative reward by collecting apples, while preserving future opportunity. Apples regrow based on nearby surviving apples: clearing a local patch completely can permanently destroy that patch's regrowth. Exercise restraint when the local apple supply looks sparse; explore other local areas instead of repeatedly exhausting the same patch. Other agents act simultaneously and may compete for apples. You receive only a compact summary of your current local RGB view and your own past actions/rewards. It contains no global map, world coordinates, other agents' hidden state, or future information. The visual labels are approximate and advisory: use action outcomes to correct your assumptions. FIRE_ZAP is available but does not create apples or reward; use it only if you have a local strategic reason. Return JSON only."""
SUPERVISOR_STRATEGY_PROMPT_PATH = ROOT / "configs" / "prompts" / "commons_harvest_supervisor_strategy.txt"
SUPERVISOR_STRATEGY_PROMPT = SUPERVISOR_STRATEGY_PROMPT_PATH.read_text(encoding="utf-8")


def summarise_local_rgb(rgb: np.ndarray, tile_size: int = 8) -> dict[str, Any]:
    """Convert the agent's local RGB image into a deliberately coarse local cue set."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB observation must have shape [height, width, 3]")
    height, width, _ = rgb.shape
    if height % tile_size or width % tile_size:
        raise ValueError("RGB dimensions must divide into whole tiles")
    counts = {"apple_like_cells": 0, "grass_like_cells": 0, "wall_like_cells": 0, "other_avatar_like_cells": 0}
    for row in range(0, height, tile_size):
        for column in range(0, width, tile_size):
            tile = rgb[row : row + tile_size, column : column + tile_size].astype(np.int16)
            red = (tile[:, :, 0] > 160) & (tile[:, :, 1] < 140) & (tile[:, :, 2] < 140)
            green = (tile[:, :, 1] > 110) & (tile[:, :, 1] > tile[:, :, 0] + 25) & (tile[:, :, 1] > tile[:, :, 2] + 25)
            dark = (tile.max(axis=2) < 115) & (tile.sum(axis=2) > 15)
            blue = (tile[:, :, 2] > 125) & (tile[:, :, 2] > tile[:, :, 0] + 35)
            if int(red.sum()) >= 3:
                counts["apple_like_cells"] += 1
            if int(green.sum()) >= 6:
                counts["grass_like_cells"] += 1
            if int(dark.sum()) >= 20:
                counts["wall_like_cells"] += 1
            if int(blue.sum()) >= 5:
                counts["other_avatar_like_cells"] += 1
    return {"view_tiles": {"height": height // tile_size, "width": width // tile_size}, **counts}


def build_commons_harvest_prompt(*, visual_summary: dict[str, Any], history: list[dict[str, Any]], ready_to_shoot: bool, valid_actions: list[str], policy_mode: str = "detailed") -> str:
    if policy_mode == "detailed":
        instructions = COMMONS_HARVEST_PROMPT
    elif policy_mode == "supervisor_strategy":
        instructions = SUPERVISOR_STRATEGY_PROMPT
    else:
        raise ValueError(f"unsupported Commons Harvest policy mode: {policy_mode}")
    return (
        f"{instructions}\n\n"
        f"LOCAL_RGB_SUMMARY_JSON:\n{json.dumps(visual_summary, sort_keys=True)}\n\n"
        f"POLICY_VISIBLE_RECENT_HISTORY_JSON:\n{json.dumps(history, sort_keys=True)}\n\n"
        f"READY_TO_SHOOT: {json.dumps(ready_to_shoot)}\n"
        f"VALID_ACTIONS_JSON:\n{json.dumps(valid_actions)}\n\n"
        'Return exactly: {"action":"ONE_VALID_ACTION"}.'
    )


def parse_action(raw: str, valid_actions: set[str]) -> dict[str, str]:
    try:
        action = str(json.loads(raw)["action"]).upper()
        if action not in valid_actions:
            raise ValueError(f"unsupported action: {action}")
        return {"action": action}
    except Exception as error:
        return {"action": "NOOP", "parse_error": str(error)[:300]}


def run(seed: int, steps: int, output: Path, record_video: bool, agent_count: int = 2, provider: str = DEFAULT_PROVIDER, model: str = DEFAULT_MODEL, run_label: str = "commons_harvest_smoke", policy_mode: str = "detailed") -> Path:
    if not run_label or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in run_label):
        raise ValueError("run_label must contain only letters, digits, underscores, and hyphens")
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / f"{run_label}_{agent_count}_agents_seed_{seed}_steps_{steps}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sessions = [run_dir / f"pi_agent_{index}_sessions" for index in range(agent_count)]
    for session in sessions:
        session.mkdir(exist_ok=True)
    env, mapping = build_commons_harvest(agent_count, env_seed=seed)
    valid_actions = set(mapping.name_to_index)
    histories: list[list[dict[str, Any]]] = [[] for _ in range(agent_count)]
    rewards = [0.0] * agent_count
    records: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    failure: Exception | None = None
    try:
        timestep = env.reset()
        for step in range(steps):
            if record_video:
                frames.append(np.array(timestep.observation[0]["WORLD.RGB"], copy=True))
            prompts: list[str] = []
            decisions: list[dict[str, str]] = []
            raw_responses: list[str] = []
            latencies: list[float] = []
            summaries: list[dict[str, Any]] = []
            for index in range(agent_count):
                observation = timestep.observation[index]
                summary = summarise_local_rgb(np.asarray(observation["RGB"]))
                prompt = build_commons_harvest_prompt(
                    visual_summary=summary,
                    history=histories[index][-12:],
                    ready_to_shoot=bool(observation["READY_TO_SHOOT"]),
                    valid_actions=sorted(valid_actions),
                    policy_mode=policy_mode,
                )
                raw, latency = decide(sessions[index] / f"context_{step // 3:03d}.jsonl", prompt, provider=provider, model=model, system_prompt=MINIMAL_SYSTEM_PROMPT)
                summaries.append(summary); prompts.append(prompt); raw_responses.append(raw); latencies.append(latency)
                decisions.append(parse_action(raw, valid_actions))
            timestep = env.step([mapping.index(item["action"]) for item in decisions])
            agents: list[dict[str, Any]] = []
            for index, decision in enumerate(decisions):
                reward = float(timestep.reward[index] or 0.0)
                rewards[index] += reward
                history_item = {"step": step, "action": decision["action"], "reward": reward}
                histories[index].append(history_item)
                agents.append({"agent_index": index, "local_rgb_summary": summaries[index], "decision": decision, "raw_pi_response": raw_responses[index], "latency_ms": latencies[index], "reward": reward, "cumulative_reward": rewards[index]})
            records.append({"seed": seed, "env_seed": seed, "decision_step": step, "agents": agents})
    except Exception as error:
        failure = error
    finally:
        env.close()
    if record_video and frames:
        from cleanup_vlm.video import write_video
        video = run_dir / "videos" / "episode_000.lossless.mkv"
        video.parent.mkdir(parents=True, exist_ok=True)
        write_video(frames, video, fps=10)
    (run_dir / "trajectory.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    summary = {"status": "failed" if failure else "completed", "game": "commons_harvest__open", "seed": seed, "env_seed": seed, "steps": steps, "completed_steps": len(records), "agents": agent_count, "provider": provider, "model": model, "run_label": run_label, "policy_mode": policy_mode, "total_reward_by_agent": rewards, "total_reward": sum(rewards), "failure": str(failure) if failure else None, "policy_input": "compact summary derived from each agent's current local RGB observation plus own causal action/reward history; no world RGB, world coordinates, global apple counts, or partner hidden state", "observation_caveat": "apple/grass/wall/avatar labels are palette-based local RGB heuristics, not simulator semantic ground truth"}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if failure:
        raise failure
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-label", default="commons_harvest_smoke")
    parser.add_argument("--policy-mode", choices=("detailed", "supervisor_strategy"), default="detailed")
    args = parser.parse_args()
    print(run(args.seed, args.steps, args.output, args.record_video, args.agents, args.provider, args.model, args.run_label, args.policy_mode))


if __name__ == "__main__":
    main()
