"""Run tiny text-only Cleanup experiments using an egocentric grid message."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MELTINGPOT = Path("/home/jack/phd/meltingpot")
TINY_VLM = Path("/home/jack/phd/tiny_cooperative_vlm/src")
sys.path[:0] = [str(ROOT / "src"), str(MELTINGPOT / "meltingpot_semantic_dataset" / "src"), str(TINY_VLM)]

from grid_context import build_grid_message
from grid_prompt import build_request_text
from cleanup_vlm.video import write_video


DETAILED_PROMPT_PATH = TINY_VLM.parent / "configs" / "prompts" / "cleanup_agent.txt"
INSTRUCTIONS = DETAILED_PROMPT_PATH.read_text(encoding="utf-8") + """

GRID INTERFACE OVERRIDE
For this experiment, you do NOT receive an RGB image. Instead you receive a current, exact 11×11 egocentric symbolic grid. `self` gives your local row, column, and facing direction. Each cell specifies terrain and objects. Treat wall and river as blocked for movement. Use the grid and your action/reward history as the sole evidence; do not infer unlisted objects or global state. The JSON grid below is the current local observation.
"""


def save_evaluation_video(frames: list[np.ndarray], output: Path, fps: int = 10) -> None:
    """Write human-only WORLD.RGB frames with the established lossless writer."""
    write_video(frames, output, fps=fps)


def _load_text_model(model_path: str, device: str) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    resolved = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if resolved == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.bfloat16 if resolved == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(resolved)
    model.eval()
    return processor, model, resolved


def _decide(processor: Any, model: Any, device: str, grid: dict[str, Any], valid_actions: set[str]) -> tuple[dict[str, Any], str, float]:
    import torch

    text = build_request_text(INSTRUCTIONS, grid)
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], padding=True, return_tensors="pt").to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    raw = processor.batch_decode(generated[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0].strip()
    try:
        decision = json.loads(raw)
        if set(decision) - {"action", "public_message", "intent"}:
            raise ValueError("unexpected decision fields")
        action = str(decision["action"]).upper()
        if action not in valid_actions:
            raise ValueError(f"unsupported action: {action}")
        decision = {
            "action": action,
            "public_message": str(decision.get("public_message", ""))[:160],
            "intent": str(decision.get("intent", ""))[:160],
        }
    except Exception as error:
        decision = {"action": "NOOP", "public_message": "", "intent": "fallback after invalid model output", "parse_error": str(error)[:300]}
    return decision, raw, (time.perf_counter() - started) * 1000


def run(seed: int, steps: int, output: Path, model_path: str, device: str, record_video: bool = False) -> Path:
    # Load Torch/Qwen before importing Melting Pot/Lab2D. Importing the simulator
    # first initializes TensorFlow CUDA stubs and can make a healthy RTX 5090
    # look unavailable to Torch in the same process.
    processor, model, resolved_device = _load_text_model(model_path, device)
    from cleanup_vlm.actions import discover_action_mapping
    from meltingpot_semantic_dataset.egocentric import ViewGeometry, world_to_egocentric_semantics
    from meltingpot_semantic_dataset.environment import build_cleanup
    from meltingpot_semantic_dataset.schema import load_schema

    output.mkdir(parents=True, exist_ok=True)
    schema, geometry = load_schema(), ViewGeometry(5, 5, 9, 1)
    mapping = discover_action_mapping()
    valid_actions = set(mapping.name_to_index)
    env = build_cleanup(2)
    records: list[dict[str, Any]] = []
    world_frames: list[np.ndarray] = []
    try:
        timestep = env.reset()
        previous_action, previous_reward, cumulative = "NOOP", 0.0, 0.0
        for step in range(steps):
            if record_video:
                world_frames.append(np.array(timestep.observation[0]["WORLD.RGB"], copy=True))
            world = timestep.observation[0]
            semantic, ids, orientations = world["WORLD.SEMANTIC_GRID"], world["WORLD.AGENT_ID_GRID"], world["WORLD.AGENT_ORIENTATION_GRID"]
            found = np.argwhere(ids == 1)
            if len(found) != 1:
                raise RuntimeError("controlled avatar is unavailable in local world state")
            y, x = (int(value) for value in found[0])
            orientation = int(orientations[y, x])
            local = world_to_egocentric_semantics(semantic, (x, y), orientation, geometry)
            local_ids = world_to_egocentric_semantics(ids[:, :, None], (x, y), orientation, geometry)[:, :, 0]
            local_orientations = world_to_egocentric_semantics(orientations[:, :, None], (x, y), orientation, geometry)[:, :, 0]
            grid = build_grid_message(semantic=local, channel_names=schema.names, agent_ids=local_ids, orientations=local_orientations, self_id=1, frame=step, previous_action=previous_action, previous_reward=previous_reward, cumulative_reward=cumulative, ready_to_shoot=bool(timestep.observation[0].get("READY_TO_SHOOT", False)), valid_actions=tuple(sorted(valid_actions)), orientation_codes=schema.orientation_codes)
            decision, raw, latency_ms = _decide(processor, model, resolved_device, grid, valid_actions)
            action = decision["action"]
            timestep = env.step([mapping.index(action), mapping.index("NOOP")])
            previous_action, previous_reward = action, float(timestep.reward[0] or 0.0)
            cumulative += previous_reward
            records.append({"seed": seed, "decision_step": step, "grid_context": grid, "decision": decision, "raw_model_response": raw, "latency_ms": latency_ms, "reward": previous_reward, "cumulative_reward": cumulative})
    finally:
        env.close()
    run_dir = output / f"seed_{seed}_steps_{steps}"
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path: str | None = None
    if record_video:
        video = run_dir / "videos" / "episode_000.lossless.mkv"
        video.parent.mkdir(parents=True, exist_ok=True)
        save_evaluation_video(world_frames, video, fps=10)
        video_path = str(video.relative_to(run_dir))
    (run_dir / "trajectory.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"status": "completed", "seed": seed, "steps": steps, "model": model_path, "device": resolved_device, "decision_count": len(records), "total_reward": cumulative, "policy_input": "egocentric symbolic grid only", "evaluation_video": video_path}, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--model", default="/home/jack/phd/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--record-video", action="store_true", help="Save human-only WORLD.RGB evaluation video as lossless FFV1 MKV.")
    args = parser.parse_args()
    print(run(args.seed, args.steps, args.output, args.model, args.device, args.record_video))


if __name__ == "__main__":
    main()
