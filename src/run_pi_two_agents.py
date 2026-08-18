"""Two independent bounded Pi agents acting simultaneously in Cleanup."""
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
from affordances import build_local_affordances
from cleanup_vlm.actions import discover_action_mapping
from cleanup_vlm.video import write_video
from grid_context import build_grid_message
from grid_prompt import build_request_text
from mediation import close_epoch, parse_mediation_choice, participant_fairness_summary, validate_mediator_plan
from meltingpot_semantic_dataset.egocentric import ViewGeometry, world_to_egocentric_semantics
from meltingpot_semantic_dataset.environment import build_cleanup
from meltingpot_semantic_dataset.schema import load_schema
from pi_client import MODEL as DEFAULT_MODEL
from pi_client import PROVIDER as DEFAULT_PROVIDER
from pi_client import MINIMAL_SYSTEM_PROMPT, decide

PI_GRID_INSTRUCTIONS = """You control one Cleanup avatar to maximise your cumulative reward. Other independent agents act at the same time. The water is generally in the global north of the map. Your grid is egocentric: the top is your current forward direction, not necessarily global north. Use your reported orientation: facing NORTH, water is generally ahead; facing EAST: water is generally to your left; facing SOUTH, water is generally behind; facing WEST, water is generally to your right. Apples are generally south of the water. You receive only your current exact 11x11 egocentric symbolic grid and your own policy-visible action/reward history. Local affordances are advisory, not an action gate: when ready_to_shoot and local live dirt is directly ahead, FIRE_CLEAN; after no_visible_target_change, reposition or turn before firing again. Prefer movement actions marked free; when blocked, choose a different free movement or turn. Walls block movement; river, sand, and grass are traversable terrain. Walk into the river to reach and clean live dirt. Apples are individual rewards; FIRE_CLEAN affects only pollution directly ahead. Do not assume the other agent's intention. Return JSON only with action, public_message, and intent."""
MINIMAL_NO_AIM_INSTRUCTIONS = """You control one avatar.
The observation is an egocentric symbolic grid. The top of the grid is the direction your avatar is facing."""
SUPERVISOR_STRATEGY_PROMPT_PATH = ROOT / "configs" / "prompts" / "cleanup_supervisor_strategy.txt"
SUPERVISOR_STRATEGY_INSTRUCTIONS = SUPERVISOR_STRATEGY_PROMPT_PATH.read_text(encoding="utf-8").strip()
REPETITION_CONDITION_INSTRUCTIONS = """This is a repeated-game coordination condition. PARTNER_VISIBLE_RECENT_HISTORY_JSON contains only past, public records of other agents' actions, rewards, and simulator-confirmed cleaning removals. Use this evidence to adapt to repeated contribution or free-riding. It contains no current-step actions, future information, or global environment state."""
MEDIATION_INTERVAL_STEPS = 50
MEDIATION_CONDITION_INSTRUCTIONS = """This is an opt-in third-party mediation condition. At review points you may voluntarily join, continue, or leave mediation. If enrolled, you receive only your own time-bounded high-level role assignment. You retain control of primitive actions and must follow local observations and valid_actions; do not assume unseen state."""
MEDIATOR_INSTRUCTIONS = """You are a bounded third-party mediator for Cleanup. Assign high-level roles only to the current participants. Use the complete causal assignment ledger and fairness summary to avoid repeatedly assigning costly CLEAN work to the same participant when another eligible participant can rotate into it. Do not control primitive actions, infer unseen state, or assign non-participants. Return only the requested JSON."""
SESSION_DECISIONS = 3
HISTORY_LIMIT = 12


def build_mediation_choice_prompt(was_enrolled: bool, is_initial_epoch: bool) -> str:
    allowed = ["JOIN", "LEAVE"] if (is_initial_epoch or not was_enrolled) else ["CONTINUE", "LEAVE"]
    return (
        "You may voluntarily participate in a bounded third-party mediation service. "
        "Your choice changes only whether the mediator may assign you a high-level role in the next interval. "
        f"Return exactly one JSON object with mediation_choice in {allowed}: "
        '{"mediation_choice":"ONE_ALLOWED_CHOICE"}'
    )


def build_mediator_prompt(participants: list[int], ledger: list[dict[str, Any]], fairness_summary: list[dict[str, Any]]) -> str:
    return (
        f"{MEDIATOR_INSTRUCTIONS}\n\n"
        f"MEDIATION_INTERVAL_STEPS: {MEDIATION_INTERVAL_STEPS}\n"
        f"CURRENT_PARTICIPANTS_JSON:\n{json.dumps(participants, sort_keys=True)}\n\n"
        f"MEDIATION_ASSIGNMENT_LEDGER_JSON:\n{json.dumps(ledger, sort_keys=True)}\n\n"
        f"MEDIATION_PARTICIPANT_FAIRNESS_SUMMARY_JSON:\n{json.dumps(fairness_summary, sort_keys=True)}\n\n"
        "Return exactly: {\"valid_for_steps\":50,\"assignments\":[{\"agent_index\":0,\"role\":\"CLEAN|HARVEST|FLEX\",\"objective\":\"...\",\"fairness_basis\":\"...\"}]}."
    )


def build_policy_prompt(
    policy_mode: str,
    grid: dict[str, Any],
    history: list[dict[str, Any]],
    partner_history: list[dict[str, Any]] | None = None,
    mediation_assignment: dict[str, Any] | None = None,
    mediation_status: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if policy_mode == "minimal_no_aim":
        policy_grid = grid
        prompt = (
            f"{MINIMAL_NO_AIM_INSTRUCTIONS}\n\n"
            f"GRID_CONTEXT_JSON:\n{json.dumps(policy_grid, sort_keys=True)}\n\n"
            f"POLICY_VISIBLE_RECENT_HISTORY_JSON:\n{json.dumps(history, sort_keys=True)}\n\n"
            "Select exactly one action from valid_actions. Return JSON only: "
            '{"action":"ONE_VALID_ACTION"}.'
        )
        return prompt, policy_grid
    if policy_mode == "supervisor_strategy":
        prompt = (
            f"{SUPERVISOR_STRATEGY_INSTRUCTIONS}\n\n"
            f"GRID_CONTEXT_JSON:\n{json.dumps(grid, sort_keys=True)}\n\n"
            f"POLICY_VISIBLE_RECENT_HISTORY_JSON:\n{json.dumps(history, sort_keys=True)}"
        )
        return prompt, grid
    if policy_mode in {"supervisor_repetition", "supervisor_mediation"}:
        prompt = (
            f"{SUPERVISOR_STRATEGY_INSTRUCTIONS}\n\n"
            f"{REPETITION_CONDITION_INSTRUCTIONS}\n\n"
            + (f"{MEDIATION_CONDITION_INSTRUCTIONS}\n\n" if policy_mode == "supervisor_mediation" else "")
            + f"GRID_CONTEXT_JSON:\n{json.dumps(grid, sort_keys=True)}\n\n"
            f"POLICY_VISIBLE_RECENT_HISTORY_JSON:\n{json.dumps(history, sort_keys=True)}\n\n"
            f"PARTNER_VISIBLE_RECENT_HISTORY_JSON:\n{json.dumps(partner_history or [], sort_keys=True)}"
        )
        if policy_mode == "supervisor_mediation":
            prompt += f"\n\nMEDIATION_STATUS_JSON:\n{json.dumps(mediation_status or {'enrolled': False}, sort_keys=True)}"
            if mediation_assignment is not None:
                prompt += f"\n\nMEDIATION_ASSIGNMENT_JSON:\n{json.dumps(mediation_assignment, sort_keys=True)}"
        return prompt, grid
    if policy_mode != "detailed":
        raise ValueError(f"unsupported policy_mode: {policy_mode}")
    return build_request_text(PI_GRID_INSTRUCTIONS, grid) + "\n\nPOLICY_VISIBLE_RECENT_HISTORY_JSON:\n" + json.dumps(history, sort_keys=True), grid


def parse_action(raw: str, valid_actions: set[str]) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        action = str(value["action"]).upper()
        if action not in valid_actions:
            raise ValueError(f"unsupported action: {action}")
        return {"action": action, "public_message": str(value.get("public_message", ""))[:160], "intent": str(value.get("intent", ""))[:160]}
    except Exception as error:
        return {"action": "NOOP", "public_message": "", "intent": "fallback after invalid Pi output", "parse_error": str(error)[:300]}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    return value


def normalise_event_item(value: Any) -> Any:
    """Decode Lab2D's compact event payloads into JSON-safe Python values."""
    if isinstance(value, (list, tuple)) and value and value[0] == b"dict":
        pairs = value[1:]
        if len(pairs) % 2 == 0:
            decoded: dict[str, Any] = {}
            for key, item in zip(pairs[::2], pairs[1::2]):
                name = key.decode("utf-8") if isinstance(key, bytes) else str(key)
                decoded[name] = normalise_event_item(item)
            return decoded
    if isinstance(value, (list, tuple)):
        return [normalise_event_item(item) for item in value]
    return value


def local_state(world: dict[str, Any], avatar_id: int, schema: Any, geometry: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], int]:
    ids, orientations = world["WORLD.AGENT_ID_GRID"], world["WORLD.AGENT_ORIENTATION_GRID"]
    found = np.argwhere(ids == avatar_id)
    if len(found) != 1:
        raise RuntimeError(f"avatar {avatar_id} is unavailable")
    y, x = (int(v) for v in found[0]); orientation = int(orientations[y, x])
    semantic = world["WORLD.SEMANTIC_GRID"]
    return (
        world_to_egocentric_semantics(semantic, (x, y), orientation, geometry),
        world_to_egocentric_semantics(ids[:, :, None], (x, y), orientation, geometry)[:, :, 0],
        world_to_egocentric_semantics(orientations[:, :, None], (x, y), orientation, geometry)[:, :, 0],
        (x, y), orientation,
    )


def run(
    seed: int,
    steps: int,
    output: Path,
    record_video: bool,
    agent_count: int = 2,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    run_label: str = "pi",
    policy_mode: str = "detailed",
) -> Path:
    if not 1 <= agent_count <= 7:
        raise ValueError("agent_count must be in [1, 7]")
    if not run_label or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in run_label):
        raise ValueError("run_label must contain only letters, digits, underscores, and hyphens")
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / f"{run_label}_{agent_count}_agents_seed_{seed}_steps_{steps}"; run_dir.mkdir(parents=True, exist_ok=True)
    session_dirs = [run_dir / f"pi_agent_{index}_sessions" for index in range(agent_count)]
    for directory in session_dirs: directory.mkdir(exist_ok=True)
    schema, geometry = load_schema(), ViewGeometry(5, 5, 9, 1)
    mapping = discover_action_mapping(); valid_actions = set(mapping.name_to_index)
    dirt_index = schema.names.index("dirt_live")
    env = build_cleanup(agent_count, env_seed=seed)
    records: list[dict[str, Any]] = []; frames: list[np.ndarray] = []
    states = [{"previous_action": "NOOP", "previous_reward": 0.0, "cumulative": 0.0, "previous_outcome": {}, "history": []} for _ in range(agent_count)]
    public_history: list[dict[str, Any]] = []
    mediation_state: dict[str, Any] = {
        "enrolled": [False] * agent_count,
        "active_epoch": None,
        "active_assignments": {},
        "ledger": [],
        "interval_outcomes": {},
        "events": [],
        "choice_fallbacks": 0,
        "validation_fallbacks": 0,
    }
    mediator_session_dir = run_dir / "mediator_sessions"
    if policy_mode == "supervisor_mediation":
        mediator_session_dir.mkdir(exist_ok=True)
    failure: Exception | None = None
    try:
        timestep = env.reset()
        for step in range(steps):
            review_event: dict[str, Any] | None = None
            if policy_mode == "supervisor_mediation" and step % MEDIATION_INTERVAL_STEPS == 0:
                review_event = {"step": step, "epoch": step // MEDIATION_INTERVAL_STEPS, "choices": []}
                active_epoch = mediation_state["active_epoch"]
                if active_epoch is not None:
                    mediation_state["ledger"].append(close_epoch(active_epoch, step - 1, mediation_state["interval_outcomes"]))
                    mediation_state["active_epoch"] = None
                    mediation_state["active_assignments"] = {}
                    mediation_state["interval_outcomes"] = {}
                for index in range(agent_count):
                    was_enrolled = bool(mediation_state["enrolled"][index])
                    choice_prompt = build_mediation_choice_prompt(was_enrolled, is_initial_epoch=(step == 0))
                    choice_session = run_dir / f"mediation_choice_agent_{index}.jsonl"
                    raw_choice, choice_latency = decide(choice_session, choice_prompt, provider=provider, model=model, system_prompt=MINIMAL_SYSTEM_PROMPT)
                    parsed_choice = parse_mediation_choice(raw_choice, was_enrolled, is_initial_epoch=(step == 0))
                    if not parsed_choice["valid"]:
                        mediation_state["choice_fallbacks"] += 1
                    mediation_state["enrolled"][index] = parsed_choice["enrolled"]
                    review_event["choices"].append({"agent_index": index, "raw": raw_choice, "latency_ms": choice_latency, **parsed_choice})
                participants = [index for index, enrolled in enumerate(mediation_state["enrolled"]) if enrolled]
                review_event["participants"] = participants
                if len(participants) >= 2:
                    fairness_summary = participant_fairness_summary(mediation_state["ledger"], participants)
                    mediator_prompt = build_mediator_prompt(participants, mediation_state["ledger"], fairness_summary)
                    mediator_session = mediator_session_dir / f"epoch_{step // MEDIATION_INTERVAL_STEPS:03d}.jsonl"
                    raw_plan, mediator_latency = decide(mediator_session, mediator_prompt, provider=provider, model=model, system_prompt=MINIMAL_SYSTEM_PROMPT)
                    plan = validate_mediator_plan(raw_plan, set(participants), MEDIATION_INTERVAL_STEPS)
                    review_event["mediator"] = {"raw": raw_plan, "latency_ms": mediator_latency, "validation": plan, "ledger": mediation_state["ledger"], "fairness_summary": fairness_summary}
                    if plan["valid"]:
                        assignments = {
                            item["agent_index"]: {**item, "epoch": step // MEDIATION_INTERVAL_STEPS, "start_step": step, "end_step": min(step + MEDIATION_INTERVAL_STEPS - 1, steps - 1)}
                            for item in plan["assignments"]
                        }
                        mediation_state["active_assignments"] = assignments
                        mediation_state["active_epoch"] = {
                            "epoch": step // MEDIATION_INTERVAL_STEPS,
                            "start_step": step,
                            "participants": participants,
                            "assignments": list(assignments.values()),
                        }
                        mediation_state["interval_outcomes"] = {index: {"reward": 0.0, "confirmed_clean_removals": 0} for index in participants}
                    else:
                        mediation_state["validation_fallbacks"] += 1
                else:
                    review_event["mediator"] = {"status": "ABSTAIN", "reason": "fewer than two opted-in participants"}
                mediation_state["events"].append(review_event)
            if record_video: frames.append(np.array(timestep.observation[0]["WORLD.RGB"], copy=True))
            pre_worlds = [timestep.observation[index] for index in range(agent_count)]
            global_pre = pre_worlds[0]["WORLD.SEMANTIC_GRID"]
            evaluation_before = {"global_active_dirt_cells": int(global_pre[:, :, dirt_index].sum()), "global_clean_river_cells": int(global_pre[:, :, schema.names.index("dirt_inactive")].sum())}
            pre_locals = [local_state(pre_worlds[index], index + 1, schema, geometry) for index in range(agent_count)]
            decisions: list[dict[str, Any]] = []; raws: list[str] = []; latencies: list[float] = []; grids: list[dict[str, Any]] = []
            # Both decisions use the same pre-step state. Agent 1 never sees agent 0's current action.
            for index in range(agent_count):
                local, local_ids, local_orientations, _, _ = pre_locals[index]
                state = states[index]
                affordances = build_local_affordances(local, schema.names, local_ids, self_id=index + 1, ready_to_shoot=bool(pre_worlds[index].get("READY_TO_SHOOT", False)))
                grid = build_grid_message(semantic=local, channel_names=schema.names, agent_ids=local_ids, orientations=local_orientations, self_id=index + 1, frame=step, previous_action=state["previous_action"], previous_reward=state["previous_reward"], cumulative_reward=state["cumulative"], previous_outcome=state["previous_outcome"], local_affordances=affordances, ready_to_shoot=bool(pre_worlds[index].get("READY_TO_SHOOT", False)), valid_actions=tuple(sorted(valid_actions)), orientation_codes=schema.orientation_codes)
                history = state["history"][-HISTORY_LIMIT:]
                partner_history = [
                    {"step": item["step"], "agents": [agent for agent in item["agents"] if agent["agent_index"] != index]}
                    for item in public_history[-HISTORY_LIMIT:]
                ]
                active_assignment = mediation_state["active_assignments"].get(index) if policy_mode == "supervisor_mediation" else None
                mediation_status = {"enrolled": bool(mediation_state["enrolled"][index]), "active_assignment": active_assignment is not None}
                prompt, policy_grid = build_policy_prompt(
                    policy_mode,
                    grid,
                    history,
                    partner_history if policy_mode in {"supervisor_repetition", "supervisor_mediation"} else None,
                    mediation_assignment=active_assignment,
                    mediation_status=mediation_status,
                )
                session = session_dirs[index] / f"context_{step // SESSION_DECISIONS:03d}.jsonl"
                mediated_modes = {"minimal_no_aim", "supervisor_strategy", "supervisor_repetition", "supervisor_mediation"}
                raw, latency = decide(session, prompt, provider=provider, model=model, system_prompt=MINIMAL_SYSTEM_PROMPT) if policy_mode in mediated_modes else decide(session, prompt, provider=provider, model=model)
                grids.append(policy_grid); raws.append(raw); latencies.append(latency); decisions.append(parse_action(raw, valid_actions))
            timestep = env.step([mapping.index(decision["action"]) for decision in decisions])
            step_events = [(event_name, normalise_event_item(event_item)) for event_name, event_item in env.events()]
            player_cleaned_counts = {index: 0 for index in range(agent_count)}
            for event_name, event_item in step_events:
                if event_name != "player_cleaned" or not isinstance(event_item, dict):
                    continue
                player_index = int(event_item.get("player_index", -1))
                zero_based = player_index - 1
                if 0 <= zero_based < agent_count:
                    player_cleaned_counts[zero_based] += 1
            post_worlds = [timestep.observation[index] for index in range(agent_count)]
            global_post = post_worlds[0]["WORLD.SEMANTIC_GRID"]
            evaluation_after = {"global_active_dirt_cells": int(global_post[:, :, dirt_index].sum()), "global_clean_river_cells": int(global_post[:, :, schema.names.index("dirt_inactive")].sum())}
            step_agents = []; public_step_agents = []
            for index in range(agent_count):
                post_local, _, _, post_position, _ = local_state(post_worlds[index], index + 1, schema, geometry)
                pre_local, _, _, pre_position, _ = pre_locals[index]
                state = states[index]; action = decisions[index]["action"]
                reward = float(timestep.reward[index] or 0.0)
                outcome = build_action_outcome(action, moved=post_position != pre_position, local_view_changed=not np.array_equal(pre_local, post_local), dirt_before=int(pre_local[:, :, dirt_index].sum()), dirt_after=int(post_local[:, :, dirt_index].sum()))
                state["previous_action"] = action; state["previous_reward"] = reward; state["cumulative"] += reward; state["previous_outcome"] = outcome
                state["history"].append({"step": step, "action": action, "reward": reward, "intent": decisions[index].get("intent", ""), "outcome": outcome})
                active_assignment = mediation_state["active_assignments"].get(index) if policy_mode == "supervisor_mediation" else None
                if index in mediation_state["interval_outcomes"]:
                    mediation_state["interval_outcomes"][index]["reward"] += reward
                    mediation_state["interval_outcomes"][index]["confirmed_clean_removals"] += player_cleaned_counts[index]
                step_agents.append({"agent_index": index, "grid_context": grids[index], "decision": decisions[index], "raw_pi_response": raws[index], "latency_ms": latencies[index], "reward": reward, "cumulative_reward": state["cumulative"], "outcome": outcome, "event_metrics": {"player_cleaned_count": player_cleaned_counts[index]}, "mediation": {"enrolled": bool(mediation_state["enrolled"][index]), "assignment": active_assignment}})
                public_step_agents.append({"agent_index": index, "action": action, "reward": reward, "confirmed_clean_removals": player_cleaned_counts[index]})
            records.append(json_safe({"seed": seed, "env_seed": seed, "decision_step": step, "evaluation_only": {"before": evaluation_before, "after": evaluation_after, "active_dirt_delta": evaluation_after["global_active_dirt_cells"] - evaluation_before["global_active_dirt_cells"]}, "events": [{"name": event_name, "item": event_item} for event_name, event_item in step_events], "mediation_review": review_event, "agents": step_agents}))
            public_history.append({"step": step, "agents": public_step_agents})
        if policy_mode == "supervisor_mediation" and mediation_state["active_epoch"] is not None:
            mediation_state["ledger"].append(close_epoch(mediation_state["active_epoch"], len(records) - 1, mediation_state["interval_outcomes"]))
            mediation_state["active_epoch"] = None
    except Exception as error:
        failure = error
    finally:
        env.close()
    if record_video:
        video = run_dir / "videos" / "episode_000.lossless.mkv"; video.parent.mkdir(parents=True, exist_ok=True); write_video(frames, video, fps=10)
    (run_dir / "trajectory.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    totals = [state["cumulative"] for state in states]
    policy_input = {
        "minimal_no_aim": "egocentric symbolic grid plus own past outcomes/history; no stated objective or strategy instructions",
        "supervisor_strategy": "egocentric symbolic grid plus own past outcomes/history; explicit supervisor-supplied long-horizon individual-reward strategy",
        "supervisor_repetition": "egocentric symbolic grid plus own past outcomes/history and a 12-step causal public record of other agents' actions, rewards, and simulator-confirmed cleaning removals",
        "supervisor_mediation": "supervisor repetition context plus voluntary 50-step third-party high-level role mediation; mediator sees only complete causal assignment/outcome ledger and assigns current opted-in participants",
    }.get(policy_mode, "each agent: egocentric symbolic grid plus own past outcomes/history only")
    summary = {"status": "failed" if failure else "completed", "seed": seed, "env_seed": seed, "steps": steps, "completed_steps": len(records), "agents": agent_count, "provider": provider, "model": model, "run_label": run_label, "policy_mode": policy_mode, "decision_count_per_agent": len(records), "total_reward_by_agent": totals, "total_reward": sum(totals), "failure": str(failure) if failure else None, "policy_input": policy_input, "pi_session_rotation": {"decisions_per_session": SESSION_DECISIONS, "history_limit": HISTORY_LIMIT}}
    if policy_mode == "supervisor_mediation":
        summary["mediation"] = {"interval_steps": MEDIATION_INTERVAL_STEPS, "review_events": mediation_state["events"], "assignment_ledger": mediation_state["ledger"], "participation_choice_fallbacks": mediation_state["choice_fallbacks"], "mediator_validation_fallbacks": mediation_state["validation_fallbacks"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if failure is not None:
        raise failure
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-label", default="pi")
    parser.add_argument("--policy-mode", choices=("detailed", "minimal_no_aim", "supervisor_strategy", "supervisor_repetition", "supervisor_mediation"), default="detailed")
    args = parser.parse_args()
    print(run(args.seed, args.steps, args.output, args.record_video, args.agents, args.provider, args.model, args.run_label, args.policy_mode))


if __name__ == "__main__": main()
