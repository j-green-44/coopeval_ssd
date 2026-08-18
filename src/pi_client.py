"""Minimal persistent Pi client for the local vLLM server.

Pi is deliberately run without its coding tools. Its persisted session is the
long-running memory; Cleanup remains the only action surface.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

PI = "/home/jack/.local/bin/pi"
PROVIDER = "cleanup-local"
MODEL = "Qwen3-VL-8B-Instruct"
SYSTEM_PROMPT = """You are a bounded policy agent inside DeepMind Melting Pot Cleanup.
You have persistent session memory. Maintain a concise running account of attempted actions and their observed outcomes. You receive egocentric symbolic grids, never a global map. Never request or use filesystem, shell, network, or coding tools. On every decision, inspect the newest grid and history. If a repeated action did not change reward or position, choose a different locally justified action. Return JSON only: {\"action\":\"ONE_VALID_ACTION\",\"public_message\":\"\",\"intent\":\"\"}."""
MINIMAL_SYSTEM_PROMPT = "Return only the requested JSON action."


def build_pi_command(
    session_path: Path,
    prompt: str,
    *,
    provider: str = PROVIDER,
    model: str = MODEL,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[str]:
    return [
        PI, "--provider", provider, "--model", model, "--session", str(session_path),
        "--no-tools", "--no-context-files", "--no-extensions", "--no-skills",
        "--system-prompt", system_prompt, "--print", prompt,
    ]


def raise_for_pi_error(error: subprocess.CalledProcessError) -> None:
    detail = (error.stderr or error.stdout or "Pi exited without an error message").strip()
    raise RuntimeError(f"Pi provider request failed: {detail}") from error


def decide(
    session_path: Path,
    prompt: str,
    timeout_s: float = 300.0,
    *,
    provider: str = PROVIDER,
    model: str = MODEL,
    system_prompt: str | None = None,
) -> tuple[str, float]:
    import time
    started = time.perf_counter()
    try:
        result = subprocess.run(
            build_pi_command(
                session_path,
                prompt,
                provider=provider,
                model=model,
                system_prompt=SYSTEM_PROMPT if system_prompt is None else system_prompt,
            ), check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s,
        )
    except subprocess.CalledProcessError as error:
        raise_for_pi_error(error)
    return result.stdout.strip(), (time.perf_counter() - started) * 1000
