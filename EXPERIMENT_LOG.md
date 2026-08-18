# Experiment log

## 2026-08-05 — Grid-context Cleanup LLM project setup

- Status: completed (two-step feasibility smoke)
- Objective: pause RGB-based perception work and test whether an LLM can act in Melting Pot Cleanup when given a structured egocentric grid description of its environment.
- Hypothesis: correct, policy-visible symbolic context will separate language/planning/action-selection limitations from the current RGB grounding failure.
- Scope/data: local Melting Pot Cleanup environment and its existing simulator semantic-export path. No visual model training or image input is part of this experiment.
- Inputs: `/home/jack/phd/meltingpot/meltingpot_semantic_dataset/` semantic exporter, the Cleanup environment, and local `/home/jack/phd/models/Qwen3-VL-8B-Instruct`.
- Method/config: `src/grid_context.py` serialises only the current agent-visible 11×11 egocentric grid, own local state, and valid actions into `cleanup_grid_context_v1`; `src/run_grid_context.py` loads text-only Qwen and validates its returned action before stepping the environment. A second Cleanup agent is `NOOP` to isolate the controlled policy.
- Command: `PYTHONPATH=/home/jack/phd/meltingpot:/home/jack/phd/meltingpot/.venv311/lib/python3.11/site-packages:/home/jack/phd/cleanup_grid_context_llm/src:/home/jack/phd/meltingpot/meltingpot_semantic_dataset/src:/home/jack/phd/tiny_cooperative_vlm/src python src/run_grid_context.py --seed <7|8> --steps 2 --output outputs` after activating `/home/jack/phd/meltingpot/.venv-train`.
- Code state: not versioned.
- Outputs: `outputs/seed_7_steps_2/` and `outputs/seed_8_steps_2/`, each with `trajectory.jsonl` and `summary.json`. The runner now also supports `--record-video`, saving human-only `WORLD.RGB` at `videos/episode_000.lossless.mkv` using the established FFV1/bgr0 writer from `tiny_cooperative_vlm`.
- Results: two initial real two-decision runs completed on CPU with two valid structured Qwen responses each. All four actions were `FORWARD`; no fallback or parse error occurred. Per-decision latency was 39.5–42.2 s. A CUDA repair then moved Qwen/Torch loading before the Melting Pot/Lab2D import, preventing TensorFlow CUDA stubs from poisoning Torch's device discovery. The forced-CUDA two-step verification (`seed_10`) completed with `device: "cuda"`, valid actions, and 1.35 s then 0.61 s decision latency.
- Findings: the end-to-end symbolic policy path works: exact local grid -> text-only local Qwen -> validated JSON action -> live Cleanup step. The initial view mainly contained walls/grass/inactive apples, and Qwen's `FORWARD`/"explore" responses were coherent but not evidence of cleaning or coordination competence. We then reused the prior detailed VLM policy prompt verbatim from `tiny_cooperative_vlm/configs/prompts/cleanup_agent.txt`, with a terminal grid-interface override that replaces the RGB modality. The 10-step CUDA probe (`seed_13`) nevertheless selected `FORWARD` ten times, each claiming it was moving toward apples. This confirms that prompt detail alone does not ground the symbolic grid into action-relative movement constraints.
- Caveats: this is oracle-context evaluation, not an RGB/VLM comparison. It is only four decisions across two two-step rollouts, with a passive partner and CPU inference. Simulator semantic state is restricted to the egocentric view, but the policy receives exact symbolic labels, so its result is an upper-bound feasibility signal rather than a deployable perception result.
- Decision: retain as a working feasibility baseline.
- Next action: run a small controlled 20–50 decision comparison across symbolic-grid Qwen, deterministic mock cleaner, and NOOP, with per-action/reward/cleaning metrics and explicit seeds.

## 2026-08-05 — Pi bounded-context 200-step Cleanup rollout

- Status: completed
- Objective: test whether a local Pi-managed, persistent policy-visible action/history context improves symbolic-grid Cleanup behaviour over direct one-shot Qwen prompting.
- Hypothesis: Exploratory.
- Scope/data: one live two-player Cleanup episode, seed 17, 200 controlled-agent decisions. The partner remains `NOOP`; this is not a cooperative-policy evaluation.
- Inputs: local vLLM server hosting `/home/jack/phd/models/Qwen3-VL-8B-Instruct`; Pi 0.83.0; exact current egocentric 11×11 symbolic grid; own action/reward history only.
- Method/config: `src/run_pi_context.py` uses Pi with coding tools, context files, extensions, and skills disabled. To prevent full-grid session accumulation exceeding the 16k server context, Pi session files rotate every 3 decisions and each request carries the latest grid plus the last 12 policy-visible `(step, action, reward, intent)` records. All model output is parsed and validated against the live action mapping before stepping Cleanup.
- Command: `source /home/jack/phd/meltingpot/.venv-train/bin/activate && PYTHONPATH=/home/jack/phd/meltingpot:/home/jack/phd/meltingpot/.venv311/lib/python3.11/site-packages:/home/jack/phd/cleanup_grid_context_llm/src:/home/jack/phd/meltingpot/meltingpot_semantic_dataset/src:/home/jack/phd/tiny_cooperative_vlm/src python src/run_pi_context.py --seed 17 --steps 200 --record-video --output outputs`
- Code state: not versioned; added `src/pi_client.py`, `src/run_pi_context.py`, focused `tests/test_pi_client.py`, and project-local `agent_harness/` tool-contract workspace.
- Outputs: `outputs/pi_seed_17_steps_200/trajectory.jsonl`, `summary.json`, `pi_sessions/`, and `videos/episode_000.lossless.mkv` (200 frames, FFV1/bgr0, 240×168).
- Results: 200/200 valid decisions, 0 parse fallbacks, total reward 0.0, and 0 positive-reward decisions. Action count: `STEP_RIGHT` 68, `STEP_LEFT` 67, `FORWARD` 60, `FIRE_CLEAN` 5. Median/p95/max model latency: 700.5/838.9/933.9 ms.
- Findings: Pi session/history management eliminated the earlier direct-session context-window failure and generated varied actions rather than the prior `FORWARD × 200` collapse. However, varied movement and occasional cleaning did not produce reward. The policy repeatedly described collecting inactive apples at local coordinates, indicating that its symbolic object semantics and movement/reward grounding remain inadequate.
- Caveats: exact symbolic labels make this an oracle-perception condition. The bounded rolling context is not yet the planned live Pi tool-calling bridge (`observe_grid`, `recent_history`, `act`); it is a persistent-session policy condition. Single seed, passive partner, and no human video review yet.
- Decision: retain as a working Pi-context baseline; reject any claim that persistent text context alone improves Cleanup reward.
- Next action: inspect the 200-frame review video and trajectory jointly, then implement the simulator-only tool bridge with explicit local movement affordances and position-change outcomes.

## 2026-08-05 — Pi context input filtering and five-decision session smoke

- Status: completed
- Objective: remove non-actionable latent semantic labels from the policy input and expand the Pi session window without again exceeding the 16k vLLM context.
- Method/config: `grid_context.py` now suppresses `apple_inactive`, `dirt_inactive`, and `spawn_point`; live apples and live dirt remain visible. `run_pi_context.py` rotates sessions every 5 decisions (was 3), retaining the bounded 12-record action/reward history.
- Command: same Pi runner command, smoke configuration `--seed 18 --steps 5 --output outputs`.
- Outputs: `outputs/pi_seed_18_steps_5/`.
- Results: all 7 tests passed; live 5-decision smoke completed with one Pi session file and no forbidden semantic labels present in any serialized policy grid. Actions: `FORWARD`, `STEP_RIGHT`, `FIRE_CLEAN`, `STEP_LEFT`, `NOOP`.
- Caveats: this is a configuration smoke, not a reward evaluation. Five decisions establishes only that the expanded session remains inside the current context limit.
- Decision: use this filtered five-decision context configuration for the next controlled rollout.
- Next action: rerun a 200-step Pi rollout with the filtered input, then compare action/reward and wall-escape behaviour against seed 17.

## 2026-08-05 — Pi action-outcome 200-step Cleanup rollout

- Status: completed
- Objective: test whether explicit past movement/cleaning outcomes enable the Pi policy to recover after blocked movement rather than repeating lateral actions at a wall.
- Scope/data: one live seed-21, 200-decision Cleanup rollout with a `NOOP` partner.
- Method/config: `src/action_outcome.py` emits only post-action, policy-visible local evidence: `movement_outcome` (`moved`/`blocked`/`not_applicable`), local-view change, visible live-dirt count before/after, and cleaning effect. The Pi session rotation was returned to 3 decisions; inactive apples/dirt and spawn points stayed hidden.
- Outputs: `outputs/pi_seed_21_steps_200/trajectory.jsonl`, `summary.json`, `pi_sessions/`, and 200-frame `videos/episode_000.lossless.mkv` (FFV1/bgr0, 240×168).
- Results: 200/200 valid decisions, 0 parse fallbacks, reward 0.0. Actions: `FORWARD` 60, `STEP_RIGHT` 52, `FIRE_CLEAN` 46, `STEP_LEFT` 23, `TURN_RIGHT` 14, `TURN_LEFT` 5. Of 199 prior-action outcomes: 124 moved, 11 blocked, 64 not applicable. Of 46 cleaning actions, 10 reported one or more visible targets removed, 33 reported no visible target change, and 2 showed more visible dirt.
- Findings: unlike seed 19 (41 invalid-output `NOOP`s and repeated `STEP_RIGHT`/`FIRE_CLEAN`), the policy reacted to most blocked outcomes with turns or different lateral actions. For example, after blocks at steps 64, 68, 83, 86, and 94 it selected `TURN_RIGHT`/`TURN_LEFT` with explicit recovery intent. It therefore learned the local operational fact needed to leave a wall, but still did not translate cleaning/navigation into individual apple reward.
- Caveats: single seed, passive partner, exact symbolic grid, and model-generated intent are evidence of chosen policy text rather than ground-truth rationale. No reward remains the decisive task metric.
- Decision: retain explicit outcome feedback and three-decision rotation; it materially improves recovery behaviour and eliminates parse fallbacks, but does not establish successful Cleanup play.
- Next action: add a compact local action-affordance summary (which relative moves are currently traversable and which facing directions contain visible live dirt) and compare several fixed seeds against this seed-21 baseline.

## 2026-08-05 — Two independent Pi agents, 200-step Cleanup rollout

- Status: completed
- Objective: determine whether an independently controlled second agent increases visible pollution removal and enables productive Cleanup play.
- Scope/data: one live 200-decision two-agent rollout, seed 23. Both agents use independent Pi sessions and their own local symbolic input/history; neither receives the other agent's current action.
- Method/config: `src/run_pi_two_agents.py` obtains both decisions from the same pre-step state and applies them simultaneously. Both policies retain the filtered grid, three-decision rotation, own outcome feedback, and twelve-record own history.
- Outputs: `outputs/pi_two_agents_seed_23_steps_200/trajectory.jsonl`, `summary.json`, per-agent `pi_agent_*_sessions/`, and `videos/episode_000.lossless.mkv` (200 frames, FFV1/bgr0, 240×168).
- Results: each agent made 200 decisions; rewards were 0.0/0.0. Agent 0: 17 cleaning actions with visible target removal, 37 clean actions with no visible change, 5 blocked movements, 7 parse fallbacks, 981.9 ms median latency. Agent 1: 20 cleaning actions with visible target removal, 45 no-effect clean actions, 3 blocked movements, 8 parse fallbacks, 1012.9 ms median latency.
- Findings: the second agent raised verified visible cleaning removals to 37 combined events and both agents explored/moved (120 moved outcomes each). However, without coordination or local cleaning affordances they still repeatedly fired on unchanged targets, and no individual apple reward occurred.
- Caveats: one seed, exact symbolic oracle context, passive/no-message coordination protocol, and a raw visible-dirt decrement is not a global cleanliness or apple-regrowth metric.
- Decision: retain two-agent simultaneous stepping as a valid harness capability, but do not claim that it improves Cleanup reward.
- Next action: add local affordances and a compact shared public cleaning-status message, then compare one-agent versus two-agent runs over fixed seeds.

## 2026-08-05 — Two-agent local-affordance and throughput run

- Status: completed
- Objective: test whether local move/beam affordances suppress no-target cleaning and sustain river cleanliness.
- Scope/data: one seed-25, 200-decision two-agent rollout.
- Method/config: each policy received current local `movement_affordances` plus a forward clean-beam target indicator. Evaluation-only trajectory fields logged global active dirt and clean river cells; these fields were verified absent from policy input.
- Outputs: `outputs/pi_two_agents_seed_25_steps_200/trajectory.jsonl`, `summary.json`, per-agent sessions, and 200-frame FFV1 video.
- Results: rewards 0.0/0.0. Agent 0 made 10 clean attempts, all no-effect; agent 1 made 24 clean attempts, 23 no-effect and 0 verified removals. Evaluation-only active dirt rose from 79 to 147 cells (minimum 79, maximum 147, mean 106.65).
- Findings: this run did not sustain cleaning. The policy mostly navigated/turned and did not acquire effective beam alignment. The local affordance input alone is insufficient as currently phrased and/or its beam-ray approximation needs calibration against the real simulator beam geometry before being used as a policy constraint.
- Caveats: seed-25 differs from seed-23, so this is not a controlled numerical comparison; it nevertheless proves no cleaning benefit in this realization. Global dirt fields are evaluator-only.
- Decision: do not use the present affordance prompt as a hard cleaning gate.
- Next action: calibrate the local beam-affordance helper against simulator action outcomes and run a scripted local-cleaner capacity baseline before another LLM comparison.

## 2026-08-05 — Directional-prior two-agent, 300-step Cleanup rollout

- Status: completed
- Objective: test whether a prompt-only water-north / apples-south task-layout prior improves river discovery and sustained cleaning.
- Scope/data: seed 27, 300 decisions per agent; no global coordinates or target locations exposed.
- Method/config: added the exact directional prior “Water is generally north, at the top of your egocentric screen. Apples generally appear south of the water.” Both policies otherwise retained local affordances, own outcome history, and evaluator-only global dirt telemetry.
- Outputs: `outputs/pi_two_agents_seed_27_steps_300/trajectory.jsonl`, `summary.json`, session transcripts, and a 300-frame FFV1 video.
- Results: rewards 0.0/0.0; no positive reward events. Agent 0: 25 clean attempts, 0 verified removals, 25 no-effect. Agent 1: 20 clean attempts, 0 verified removals, 20 no-effect. Active river dirt increased 79→147 cells (min 79, mean 121.53). The river was never below the 0.4 depletion threshold (about 59 of 147 river cells).
- Findings: the directional prior did not restore productive cleaning. Agents moved extensively (194 and 186 successful movement outcomes) but did not obtain beam-target alignment, confirming the present local clean-affordance geometry/prompt policy is the blocking issue rather than episode length.
- Decision: keep the directional hint as an allowed task prior if desired, but pause LLM rollouts until beam calibration and a scripted capacity baseline are established.
- Next action: implement calibrated simulator-aligned beam target detection and two scripted reactive cleaners; use evaluator-only dirt trajectories to measure the achievable cleaning ceiling.

## 2026-08-05 — Orientation-relative navigation consistency check

- Status: completed
- Objective: test the corrected global-north/orientation-relative task prior across four new two-agent 50-step episodes.
- Scope/data: seeds 29–32; two independent Pi agents per episode; 400 total policy decisions.
- Outputs: `outputs/pi_two_agents_seed_{29,30,31,32}_steps_50/`, each with trajectory, summary, sessions, and FFV1 review video.
- Results: all 8 agent episodes observed river in their local grid. Seven had river visible at step 0; the other two first saw it at steps 3 and 12. Verified local dirt removals: 15 total (seed 29: 3, seed 30: 0, seed 31: 5, seed 32: 7). One agent (seed 29 agent 0) had 12 parse fallbacks; all other agents had 0.
- Findings: river-region observation is consistent under the corrected frame instruction (8/8), but cleaning alignment remains variable and is not yet a sustained-throughput solution.
- Decision: retain the orientation-relative wording; do not treat it as evidence of successful Cleanup policy until calibrated cleaning and reward results follow.
- Next action: inspect the seed-29 parse fallback source and calibrate beam geometry before a controlled multi-seed cleaning comparison.

## 2026-08-05 — River-traversability four-episode check

- Status: completed
- Objective: test whether correcting the false “river blocks movement” instruction and affordance improves local cleaning.
- Scope/data: seeds 34–37; two agents × 50 decisions each; 400 total policy decisions.
- Method/config: river was made traversable in both prompt and local movement affordance; only walls, map boundaries, and occupied cells block movement.
- Outputs: `outputs/pi_two_agents_seed_{34,35,36,37}_steps_50/`, each with trajectory, summary, sessions, and FFV1 video.
- Results: 7/8 agents observed river within 3 steps; seed 34 agent 0 did not observe river in 50 steps. Verified local dirt removals: seed 34: 7, seed 35: 12, seed 36: 8, seed 37: 12; total 39. This compares with 15 over the preceding four orientation-only episodes, but seeds differ and this is not a controlled comparison. Parse fallbacks: seed 35 agent 0=5, seed 36 agent 1=4, seed 37 agent 1=8; all others=0.
- Findings: the correction restored frequent effective cleaning for river-reaching agents. The remaining problem is inconsistent navigation for some spawns and JSON reliability, not belief that river terrain is impassable.
- Decision: keep river traversable in policy and affordances.
- Next action: repeat the old seeds 29–32 with the river correction for a controlled before/after comparison, and diagnose session parse fallbacks.

## 2026-08-05 — Three-agent 600-step Cleanup rollout

- Status: completed
- Objective: test whether a third independent cleaner can hold river pollution low enough for apple regrowth.
- Scope/data: seed 39, three independent agents, 600 decisions per agent (1,800 total policy decisions).
- Outputs: `outputs/pi_3_agents_seed_39_steps_600/trajectory.jsonl`, `summary.json`, three per-agent session directories, and `videos/episode_000.lossless.mkv` (600-frame FFV1).
- Results: rewards 0.0/0.0/0.0. Verified local removals: agent 0=63, agent 1=62, agent 2=44 (169 combined). Active dirt started at 79, reached minimum 56, ended 99 (mean 81.82); it was below the approximate 59-cell apple depletion threshold for 10 steps. Parse fallbacks: 18, 31, and 10 respectively.
- Findings: three cleaners materially improved river control versus prior runs and briefly crossed the apple-regrowth threshold, but did not remain below it long enough and no agent transitioned to harvest behaviour. The key next failure mode is role allocation: all agents remain primarily cleaners while apple availability is transient.
- Decision: retain three-agent capability and river-traversable policy; treat this as partial environmental-control success, not reward success.
- Next action: add a local role prompt/phase rule: at least one agent should leave the river and search south for live apples after local dirt has been reduced, while the other cleaners maintain the river. Also fix parse fallback reliability before a long comparison.

## 2026-08-12 — Local-model minimal no-aim prompt pilot (corrected memory-preserving condition)

- Status: completed. An earlier same-day `local_minimal_no_aim` run accidentally removed memory/history as well as behavioural instructions; it is retained only as a stateless-control artifact and is not the requested prompt comparison.
- Objective: observe three local LLM agents after removing stated objectives and strategy advice while preserving the established memory and policy-visible context.
- Scope/data: one 30-step Cleanup episode, environment seed 39, three independently controlled Qwen3-VL-8B-Instruct agents.
- Method/config: `--policy-mode minimal_no_aim`; the fixed instruction contains only avatar control, egocentric-grid semantics, valid-action selection, and JSON formatting. Each agent retains three-decision Pi sessions, the latest 12 own action/reward/outcome records, and the established grid payload including own reward/outcome state and local affordances. The prompt does not state a reward objective or advise cleaning, harvesting, movement, coordination, partner intention, map direction, or role allocation. Local vLLM used `VLLM_USE_FLASHINFER_SAMPLER=0` because FlashInfer warm-up otherwise required unavailable local NVCC.
- Command: `python src/run_pi_two_agents.py --seed 39 --agents 3 --steps 30 --record-video --provider cleanup-local --model Qwen3-VL-8B-Instruct --run-label local_no_aim_memory --policy-mode minimal_no_aim --output outputs` with the established project `PYTHONPATH` and Melting Pot environment.
- Outputs: `outputs/local_no_aim_memory_3_agents_seed_39_steps_30/`, including trajectory, summary, 30 rotated Pi session files, and a verified 30-frame lossless FFV1/bgr0 review video.
- Results: all 90 responses were valid, with zero parse fallbacks and zero reward. Agent 0 selected FORWARD=12, STEP_RIGHT=14, STEP_LEFT=4 and moved on all 30 steps. Agent 1 selected FORWARD=1, STEP_RIGHT=13, TURN_RIGHT=1, FIRE_CLEAN=15; 11 `player_cleaned` events were attributed to it. Agent 2 selected FORWARD=25, STEP_RIGHT=4, STEP_LEFT=1 and moved on all 30 steps. Global active dirt fell from 79 to 68. Median decision latency was 616.6, 745.3, and 723.2 ms.
- Findings: once memory and action outcomes were restored, the no-aim agents did not collapse into repeated FORWARD actions. One agent spontaneously adopted cleaning behaviour and removed 11 dirt cells, while the other two explored using movement actions. No reward was earned within the short horizon. This demonstrates materially different behaviour from the accidental stateless condition, but one episode is insufficient for a cooperation claim.
- Caveats: one seed, 30 steps, one model, and exact symbolic oracle observations. The retained local-affordance payload is an established derived input and can shape action selection; only the behavioural instruction wording was intentionally changed in this comparison.
- Decision: treat `local_no_aim_memory` as the requested no-stated-aim pilot; treat `local_minimal_no_aim` only as an accidental stateless ablation.
- Next action: repeat short fixed-condition episodes across seeds before interpreting consistency or emergent role allocation.

## 2026-08-12 — Local-model no-aim, memory-preserving 200-step rollout

- Status: completed.
- Objective: extend the corrected no-stated-aim, memory-preserving condition from 30 to 200 steps without changing the policy interface or environment seed.
- Scope/data: one 200-step Cleanup episode, environment seed 39, three independently controlled Qwen3-VL-8B-Instruct agents; 600 total policy decisions.
- Method/config: same `minimal_no_aim` condition as the corrected 30-step pilot: three-decision Pi sessions, latest 12 own action/reward/outcome records, full established local grid context and affordances, but no stated objective or strategy/coordination instructions.
- Outputs: `outputs/local_no_aim_memory_3_agents_seed_39_steps_200/`, including 200 trajectory records, summary, 201 rotated Pi session files, and a verified 200-frame FFV1/bgr0 review video.
- Results: all 600 responses were valid with zero parse fallbacks and zero reward. Agent 0: FORWARD=25, STEP_RIGHT=89, STEP_LEFT=86, moved on all 200 steps, no cleaning. Agent 1: FORWARD=12, STEP_RIGHT=99, STEP_LEFT=89, moved on all 200 steps, no cleaning. Agent 2: FORWARD=8, FIRE_CLEAN=65, STEP_RIGHT=66, STEP_LEFT=61; 9 simulator-attributed cleaning events and 56 no-effect FIRE_CLEAN steps. Global active dirt started at 79, reached a minimum of 78, and ended at 143; it never fell below 59. Median decision latency was 617.2, 597.9, and 605.4 ms.
- Findings: the longer rollout again produced an asymmetric role pattern—two agents moved continuously while one performed all cleaning—but the cleaner's throughput was too low to control pollution. The team earned no apples/reward and did not demonstrate successful common-resource management. The identity of the cleaner differed from the separate 30-step run, so role assignment is not yet stable evidence of deliberate coordination.
- Caveats: one 200-step trial, stochastic local-model responses, exact symbolic oracle observations, and local affordances remain policy-visible. Same seed fixes the environment but not necessarily model sampling. A role split visible in actions is not by itself proof that agents coordinated, especially without an explicit communication mechanism.
- Decision: retain as a completed long-horizon no-aim pilot; report environmental throughput and role asymmetry, not cooperation success.
- Next action: run repeated short fixed-condition trials and inspect agent rationales/session histories before attributing the role split to partner-aware reasoning.

## 2026-08-11 — Codex three-agent rollout attempt (alias resolved to GPT-5.6 Terra)

- Status: failed
- Objective: compare a Codex-hosted GPT-5.6 model with the prior local Qwen3-VL-8B policy under the same 600-step, three-agent Cleanup condition.
- Scope/data: seed 39; three simultaneously acting agents; same current egocentric 11×11 grid, local-affordance, own-history, session-rotation, action-validation, and prompt paths as the Qwen run. No global telemetry entered policy input.
- Method/config: `--provider openai-codex --model gpt-5.6 --run-label gpt56_codex`. Pi resolved the non-specific `gpt-5.6` model selector to `gpt-5.6-terra`, not GPT-5.6 Sol.
- Outputs: partial Pi sessions under `outputs/gpt56_codex_3_agents_seed_39_steps_600/`. The original run produced no trajectory, summary, or video, but the common completed session prefix was deterministically replayed into `videos/replay_steps_000_to_464.lossless.mkv` with `replay_manifest.json` (465 frames, 10 fps, lossless FFV1/bgr0).
- Results: the run completed 465 full environment steps (0–464) before agent 1's request at step 465 received three Codex responses stating that the servers were overloaded. Pi then exited non-zero. This is a provider-availability failure, not a policy result.
- Caveats: no valid 600-step comparison can be inferred. The old runner only wrote the trajectory and FFV1 video after a normal 600-step completion, so its in-memory frames were lost on the exception.
- Decision: reject this attempt as invalid for comparison.
- Next action: rerun from a fresh output directory with the explicit `gpt-5.6-sol` model ID after the provider smoke check; use the failure-resilient runner below.

## 2026-08-11 — GPT-5.6 Sol matched three-agent rollout

- Status: planned
- Objective: run the same seed-39, three-agent, 600-step protocol with the explicit GPT-5.6 Sol Codex model.
- Method/config: updated the runner so any provider failure preserves a partial `trajectory.jsonl`, `summary.json` with `status: failed` and completed-step count, and an FFV1 video before re-raising the error. Pi failure messages now retain only provider stderr, rather than echoing the full policy prompt. Full project suite: 16 tests passed. A direct `gpt-5.6-sol` one-decision JSON smoke passed.
- Command: `source /home/jack/phd/meltingpot/.venv-train/bin/activate && PYTHONPATH=/home/jack/phd/meltingpot:/home/jack/phd/meltingpot/.venv311/lib/python3.11/site-packages:/home/jack/phd/cleanup_grid_context_llm/src:/home/jack/phd/meltingpot/meltingpot_semantic_dataset/src:/home/jack/phd/tiny_cooperative_vlm/src python src/run_pi_two_agents.py --seed 39 --agents 3 --steps 600 --record-video --provider openai-codex --model gpt-5.6-sol --run-label gpt56_sol_codex --output outputs`
- Expected outputs: `outputs/gpt56_sol_codex_3_agents_seed_39_steps_600/`.
- Results: not run.
- Caveats: a failed/partial run remains an infrastructure artifact, not a policy comparison.
- Next action: launch only when ready; inspect `summary.json` before comparing it with the Qwen baseline.

## 2026-08-12 — Luna three-agent minimal no-aim 100-step rollout

- Status: completed
- Objective: rerun the recent three-agent Luna condition using the newly created minimal no-aim prompt, without overwriting the earlier detailed-prompt Luna artifact.
- Hypothesis: Exploratory
- Scope/data: one 100-step Cleanup episode, environment seed 39, three simultaneously acting agents; 300 total Codex policy decisions.
- Inputs: `openai-codex` / `gpt-5.6-luna`; current egocentric symbolic grid stripped of history, reward/outcome state, and derived affordances, with no stated objective.
- Method/config: `--policy-mode minimal_no_aim`; fixed prompt specifies avatar control and egocentric forward direction only, then asks for one valid JSON action. The executed artifact used a fresh one-decision Pi session per step and excluded `agent_state` and `local_affordances` from the supplied grid; it is therefore a stateless minimal-interface ablation.
- Command: `source /home/jack/phd/meltingpot/.venv-train/bin/activate && PYTHONPATH=/home/jack/phd/meltingpot:/home/jack/phd/meltingpot/.venv311/lib/python3.11/site-packages:/home/jack/phd/cleanup_grid_context_llm/src:/home/jack/phd/meltingpot/meltingpot_semantic_dataset/src:/home/jack/phd/tiny_cooperative_vlm/src python src/run_pi_two_agents.py --seed 39 --steps 100 --agents 3 --provider openai-codex --model gpt-5.6-luna --run-label gpt56_luna_minimal_noaim --policy-mode minimal_no_aim --record-video`
- Code state: not versioned; targeted prompt/client/event/runner tests passed (12 tests) immediately before launch.
- Outputs: `outputs/gpt56_luna_minimal_noaim_3_agents_seed_39_steps_100/`, including `trajectory.jsonl`, `summary.json`, per-agent Pi sessions, and `videos/episode_000.lossless.mkv`.
- Results: completed 100/100 steps with zero parse fallbacks and reward 0.0/0.0/0.0. Native simulator `player_cleaned` events: agent 0=0, agent 1=20, agent 2=0. Active dirt was 79 at start and 84 at end. Agent 1 fired 74 times; agent 2 fired 98 times without a confirmed removal.
- Findings: the minimal no-aim Luna condition completed reliably, unlike the immediately preceding detailed-prompt Luna attempt that stopped at 93 steps on a provider `fetch failed`. Behaviour was highly asymmetric: one agent found some effective cleaning, while another mostly fired ineffectively. The team did not sustain dirt control or earn reward.
- Caveats: one seeded episode, remote-provider model behaviour, exact symbolic oracle grid, and no stated task objective. This is a feasibility/behavioural ablation, not evidence of coordination or a fair general model comparison.
- Decision: retain as the completed minimal-prompt Luna artifact; do not infer task understanding from agent 1's 20 clean events.
- Next action: compare its video and native event timeline against the preserved detailed-prompt Luna prefix, then repeat short fixed-seed trials only if testing consistency is useful.

## 2026-08-12 — Luna three-agent minimal no-aim, memory-preserving 100-step rollout

- Status: completed
- Objective: repeat the stateless minimal no-aim Luna condition after restoring policy-visible own memory, while preserving the stateless and detailed-prompt artifacts.
- Hypothesis: Exploratory
- Scope/data: one 100-step Cleanup episode, environment seed 39, three simultaneously acting agents; 300 total Codex policy decisions.
- Inputs: `openai-codex` / `gpt-5.6-luna`; current egocentric grid, own `agent_state`, local affordances, and up to 12 own past action/reward/outcome records. The fixed instruction still contains no task objective or strategy advice.
- Method/config: `--policy-mode minimal_no_aim`; Pi sessions rotate every 3 decisions, preserving bounded agent-local context. Targeted prompt/client/event/runner tests passed (12 tests) before launch.
- Command: `source /home/jack/phd/meltingpot/.venv-train/bin/activate && PYTHONPATH=/home/jack/phd/meltingpot:/home/jack/phd/meltingpot/.venv311/lib/python3.11/site-packages:/home/jack/phd/cleanup_grid_context_llm/src:/home/jack/phd/meltingpot/meltingpot_semantic_dataset/src:/home/jack/phd/tiny_cooperative_vlm/src python src/run_pi_two_agents.py --seed 39 --steps 100 --agents 3 --provider openai-codex --model gpt-5.6-luna --run-label gpt56_luna_minimal_noaim_memory --policy-mode minimal_no_aim --record-video`
- Code state: not versioned.
- Outputs: `outputs/gpt56_luna_minimal_noaim_memory_3_agents_seed_39_steps_100/`, including 100 trajectory records, summary, Pi sessions, and a verified 100-frame 240×168 FFV1/bgr0 review video.
- Results: completed 100/100 steps, zero parse fallbacks, and reward 0.0/0.0/0.0. Native simulator `player_cleaned` events: agent 0=27, agent 1=12, agent 2=35 (74 total). Active dirt fell 79→30, first fell below 59 at step 36, and remained below 59 for 64/100 steps. Minimum active dirt was 30.
- Findings: restoring own past-outcome/history context substantially changed this one fixed-seed realization relative to the preceding stateless minimal run: native dirt removals rose 20→74 and active dirt ended 84→30. All three agents contributed confirmed cleaning. This demonstrates better local cleaning throughput, not successful Cleanup play: no reward occurred and this is one stochastic remote-model episode.
- Caveats: single seed and single repeat per condition; model sampling/provider state are not controlled by the environment seed. Exact symbolic oracle observations and policy-visible local affordances remain substantial assistance. Do not treat this as a causal estimate of memory's average effect without repeated matched trials.
- Decision: retain as a strong single-run indication that bounded own-history/outcome context is operationally useful under the no-aim prompt.
- Next action: review the video around steps 30–45 and run repeated short fixed-condition episodes before making a cross-condition claim.

## 2026-08-17 — GPT-5.6 Terra supervisor-strategy 30-step Cleanup pilot

- Status: completed
- Objective: test the supervisor-supplied explicit strategic Cleanup prompt after replacing its incompatible integer-only response requirement with the harness JSON action schema.
- Hypothesis: Exploratory.
- Scope/data: one 30-step Cleanup episode, environment seed 39, three independently controlled `openai-codex` / `gpt-5.6-terra` agents; 90 policy decisions.
- Inputs: current egocentric symbolic grid, established own action/reward/outcome history (up to 12 entries), local affordances, and the supervisor-supplied strategy prompt stored at `configs/prompts/cleanup_supervisor_strategy.txt`.
- Method/config: `--policy-mode supervisor_strategy`; each agent uses a three-decision Pi session rotation. The edited prompt requires exactly `{"action":"ONE_VALID_ACTION"}` and nothing else. Global evaluator telemetry remains absent from policy requests.
- Command: `source /home/jack/phd/meltingpot/.venv-train/bin/activate && PYTHONPATH=/home/jack/phd/meltingpot:/home/jack/phd/meltingpot/.venv311/lib/python3.11/site-packages:/home/jack/phd/cleanup_grid_context_llm/src:/home/jack/phd/meltingpot/meltingpot_semantic_dataset/src:/home/jack/phd/tiny_cooperative_vlm/src python src/run_pi_two_agents.py --seed 39 --agents 3 --steps 30 --record-video --provider openai-codex --model gpt-5.6-terra --run-label gpt56_terra_supervisor_strategy_retry --policy-mode supervisor_strategy --output outputs`
- Code state: not versioned; 7 targeted prompt/client tests passed before launch.
- Outputs: `outputs/gpt56_terra_supervisor_strategy_retry_3_agents_seed_39_steps_30/` (`trajectory.jsonl`, `summary.json`, per-agent Pi sessions, and 30-frame 240×168 FFV1/bgr0 review video).
- Results: completed 30/30 simultaneous steps; all 90 responses parsed as valid actions (0 fallbacks), and total reward was 0.0 (0.0 per agent). Native `player_cleaned` events were agent 0=13, agent 1=0, agent 2=13 (26 total). Agent 0 made 13 clean attempts (7 local visible-dirt reductions, 6 no-change); agent 2 made 15 (10 reductions, 5 no-change). Agent 1 moved on all 30 steps and did not clean. Evaluator-only active dirt fell 79→53 (minimum 53). Median decision latency: agent 0=7.24 s, agent 1=8.60 s, agent 2=6.76 s.
- Findings: the edited JSON format worked reliably with GPT-5.6 Terra; no integer/parser mismatch occurred. The prompt produced a cleaner/explorer split and materially reduced river pollution in this short realization, but no apples/reward appeared within 30 steps. This establishes output compatibility and short-horizon cleaning throughput, not successful Cleanup play or deliberate coordination.
- Caveats: one short fixed-seed remote-model pilot; the explicit strategic prompt is a substantial policy prior, symbolic labels are oracle perception, and a role split alone is not evidence of coordination.
- Decision: retain as the `supervisor_strategy` condition for repeated matched short trials; do not compare its reward to other conditions from this single episode.
- Next action: repeat the same 30-step protocol over fixed seeds with the same model and compare native cleaning events, dirt trajectory, reward, action mix, parsing, and latency against the memory-preserving no-aim condition.

## 2026-08-17 — GPT-5.6 Terra supervisor-strategy 300-step extension

- Status: completed
- Objective: extend the completed 30-step `supervisor_strategy` pilot under the identical seed, model, three-agent, causal-local observation, session rotation, and action-validation configuration, testing whether short-horizon cleaning translates into sustained river control and apple reward.
- Hypothesis: the cleaner/explorer split may sustain sufficient river cleanliness for regrowth, but reward remains uncertain.
- Scope/data: one 300-step Cleanup episode, environment seed 39, three simultaneous `openai-codex` / `gpt-5.6-terra` agents; 900 policy decisions.
- Inputs: `configs/prompts/cleanup_supervisor_strategy.txt`, current egocentric grid, own past action/reward/outcome history, and local affordances only. Global dirt telemetry is evaluator-only.
- Method/config: `--policy-mode supervisor_strategy`, three decisions per Pi session, 12 history records, lossless human-only review video.
- Command: `source /home/jack/phd/meltingpot/.venv-train/bin/activate && PYTHONPATH=/home/jack/phd/meltingpot:/home/jack/phd/meltingpot/.venv311/lib/python3.11/site-packages:/home/jack/phd/cleanup_grid_context_llm/src:/home/jack/phd/meltingpot/meltingpot_semantic_dataset/src:/home/jack/phd/tiny_cooperative_vlm/src python src/run_pi_two_agents.py --seed 39 --agents 3 --steps 300 --record-video --provider openai-codex --model gpt-5.6-terra --run-label gpt56_terra_supervisor_strategy --policy-mode supervisor_strategy --output outputs`
- Code state: not versioned; the same focused prompt/client tests passed for the preceding 30-step run.
- Outputs: `outputs/gpt56_terra_supervisor_strategy_3_agents_seed_39_steps_300/`, with `trajectory.jsonl`, `summary.json`, per-agent Pi sessions, and a verified 300-frame 240×168 FFV1/bgr0 review video.
- Results: completed 300/300 simultaneous steps and all 900 actions parsed correctly (0 fallbacks). Total individual reward was 106: agent 0=0, agent 1=106, agent 2=0. Agent 1's first reward occurred at step 53. Native `player_cleaned` events: agent 0=97, agent 1=0, agent 2=59 (156 total). Agent 0: 211 clean attempts, 60 local visible-dirt reductions, 124 no-change outcomes. Agent 2: 188 attempts, 45 reductions, 123 no-change. Agent 1 moved on 299/300 steps and harvested. Evaluator-only active dirt fell 79→52, minimum 34, mean 49.16; it first fell below 59 at step 39 and remained below 59 for 261/300 post-step states. Median/p95 decision latency (s): agent 0=7.09/11.33, agent 1=7.93/13.51, agent 2=7.19/12.78.
- Findings: this realization achieved the full maintainer/harvester pattern needed for Cleanup: two agents sustained enough cleaning for river recovery and one agent collected apples, producing 106 individual reward. Forensic review confirms that agent 1 performed zero clean actions and all recorded policy `public_message`/`intent` fields were empty; the policy-visible trajectories also contain no observed other-agent cells. Thus the division is not evidenced as an explicit negotiated plan. Behaviourally, agent 1 free-rode on the public good: it began orchard-oriented exploration from step 0 and captured all reward once apples appeared, while agents 0 and 2 remained locked into cleaning. The action split is strong functional role-specialisation evidence under this explicit prompt, but the agents have no communication channel and one episode cannot establish deliberate coordination or general performance.
- Caveats: one fixed seed, remote-model sampling/provider state, exact symbolic oracle observations, and an explicit strategy prior. Cleaner policies were inefficient: 247/399 cleaning attempts had no local visible-target change. The 30-step result alone did not predict reward because apple collection began only at step 53.
- Decision: retain `supervisor_strategy` as a promising explicit-policy condition; compare it against the memory-preserving no-aim condition using repeated matched seeds before claiming a prompt effect.
- Next action: run matched 300-step repeats across several seeds for both `supervisor_strategy` and `minimal_no_aim`, then report reward distribution, native cleaning throughput, dirt-control duration, action efficiency, parser reliability, and review-video failure cases.

## 2026-08-17 — CoopEval repetition: partner-visible-history 30-step smoke

- Status: running
- Objective: implement the first CoopEval mechanism family, repetition, by giving each agent a causal, public 12-step record of the other agents' past actions, rewards, and simulator-confirmed cleaning removals; test integration before a 300-step extension.
- Hypothesis: this makes repeated contribution/free-riding observable without exposing current actions, future data, global coordinates, or evaluator-only dirt telemetry.
- Scope/data: one 30-step Cleanup episode, seed 39, three `openai-codex` / `gpt-5.6-terra` agents; 90 policy decisions.
- Method/config: new `supervisor_repetition` mode retains the existing supervisor strategy, local grid, own history, local affordances, and three-decision session rotation. The added partner history is populated only after simultaneous environment stepping, then supplied on later decisions. Native `player_cleaned` provides contribution attribution.
- Command: `source /home/jack/phd/meltingpot/.venv-train/bin/activate && PYTHONPATH=/home/jack/phd/meltingpot:/home/jack/phd/meltingpot/.venv311/lib/python3.11/site-packages:/home/jack/phd/cleanup_grid_context_llm/src:/home/jack/phd/meltingpot/meltingpot_semantic_dataset/src:/home/jack/phd/tiny_cooperative_vlm/src python src/run_pi_two_agents.py --seed 39 --agents 3 --steps 30 --record-video --provider openai-codex --model gpt-5.6-terra --run-label gpt56_terra_supervisor_repetition --policy-mode supervisor_repetition --output outputs`
- Code state: not versioned; added a test-first partner-history prompt test. All 24 project tests passed after implementation.
- Outputs: `outputs/gpt56_terra_supervisor_repetition_3_agents_seed_39_steps_30/`.
- Results: completed all 30 steps; 90/90 parsed actions and 0 parser fallbacks. Agent 0 had 9 native cleaning removals from 15 cleaning attempts; agent 1 had 9 from 15; agent 2 had none from 0. Active dirt fell from 79 to 61. Reward was 0 for every agent, as expected before apple production; the mechanism was supplied in every inspected request and contained no `global_active_dirt_cells`. FFV1 review video verified: 30 frames, 240×168, `bgr0`.
- Caveats: this is an explicit public-history information mechanism, distinct from the current local-only strategy condition; it is not yet a reputation, mediator, or contract mechanism.
- Decision: operational smoke passed (full horizon, zero parse fallbacks, causal partner history serialized, and task-relevant cleaning). Extend to the matched 300-step run.
- Next action: run and review the matched 300-step extension.

## 2026-08-17 — CoopEval repetition: partner-visible-history 300-step extension

- Status: running
- Objective: extend the successful partner-visible-history smoke to a matched 300-step episode and assess whether repeated public contribution records change labour allocation or reward concentration.
- Scope/data: seed 39, three `openai-codex` / `gpt-5.6-terra` agents, 900 policy decisions, lossless FFV1 review video.
- Method/config: identical to the 30-step repetition smoke; `supervisor_repetition`, three-decision persistent-session rotation, 12-step own-history and partner-history windows, and native `player_cleaned` attribution.
- Command: `source /home/jack/phd/meltingpot/.venv-train/bin/activate && PYTHONPATH=/home/jack/phd/meltingpot:/home/jack/phd/meltingpot/.venv311/lib/python3.11/site-packages:/home/jack/phd/cleanup_grid_context_llm/src:/home/jack/phd/meltingpot/meltingpot_semantic_dataset/src:/home/jack/phd/tiny_cooperative_vlm/src python src/run_pi_two_agents.py --seed 39 --agents 3 --steps 300 --record-video --provider openai-codex --model gpt-5.6-terra --run-label gpt56_terra_supervisor_repetition --policy-mode supervisor_repetition --output outputs`
- Outputs: `outputs/gpt56_terra_supervisor_repetition_3_agents_seed_39_steps_300/`.
- Results: completed normally: 300/300 steps, 900/900 parsed actions, and 0 fallbacks. Total reward 86: agent 0 = 23, agent 1 = 0, agent 2 = 63. Jain reward fairness = 0.548. Native cleaning events: agent 0 = 37, agent 1 = 67, agent 2 = 17 (121 total). Active dirt: 79 start, 40 minimum, 89 end; 126/300 post-step states below 59 active dirt. First reward occurred at step 40. FFV1 video verified: 300 frames, 240×168, `bgr0`.
- Comparison with matched single `supervisor_strategy` rollout: reward distribution was less concentrated (`[23, 0, 63]`, Jain 0.548 versus `[0, 106, 0]`, Jain 0.333), but total reward was lower (86 versus 106), cleaning was lower (121 versus 156 native events), and river maintenance was worse (126 versus 261 states below 59; end dirt 89 versus 52).
- Behavioural reading: the initial two-maintainer/one-harvester pattern shifted part way through: agent 0 harvested 23 after contributing early cleaning, while agent 1 became the persistent zero-reward maintainer. The partner-history condition therefore reduced—but did not remove—reward concentration and did not establish equitable reciprocal role rotation.
- Decision: the repetition mechanism is operational and produces a meaningful behavioural change, but this single rollout does not show improved cooperative performance overall. Treat it as a feasibility result; repeat matched seeds before a mechanism claim. The next CoopEval family to implement should be reputation, using the same causal contribution ledger but a transparent rolling contribution score.
- Caveats: do not interpret a single seed as evidence that the repetition mechanism improves cooperation or fairness; environment seed is fixed but remote-model responses may vary between rollouts.

## 2026-08-18 — CoopEval mediation: opt-in high-level role allocation

- Status: planned
- Objective: evaluate third-party mediation without centralising primitive Cleanup control. Agents may voluntarily join mediation at 50-step review epochs; the mediator assigns only participant high-level roles (`CLEAN`, `HARVEST`, or `FLEX`).
- Hypothesis: retaining a complete causal assignment/outcome ledger lets the mediator rotate cleaning burden and reward opportunity more fairly than partner history alone, without reducing individual agents to puppets.
- Scope/data: planned fixed-seed 30-step plumbing smoke, then 100-step ledger/second-epoch validation before any 300-step run; three agents; local symbolic observations; lossless FFV1 review video.
- Method/config: `supervisor_mediation` retains the supervisor strategy and 12-step causal partner-history record. At each review every agent returns a structured `JOIN`/`CONTINUE`/`LEAVE` choice. With two or more participants, the mediator receives the complete prior mediation ledger — assignments, realised reward, and native `player_cleaned` counts — plus a derived per-participant fairness summary. It can assign current participants only. The mediator receives no grid, global dirt telemetry, global position, current-step action, or future outcome.
- Code state: not versioned. Implementation added `src/mediation.py`, `tests/test_mediation.py`, and `supervisor_mediation` support in `src/run_pi_two_agents.py`; full test suite passed (30 tests) after implementation.
- Command: `source /home/jack/phd/meltingpot/.venv-train/bin/activate && PYTHONPATH=/home/jack/phd/meltingpot:/home/jack/phd/meltingpot/.venv311/lib/python3.11/site-packages:/home/jack/phd/cleanup_grid_context_llm/src:/home/jack/phd/meltingpot/meltingpot_semantic_dataset/src:/home/jack/phd/tiny_cooperative_vlm/src python src/run_pi_two_agents.py --seed 39 --agents 3 --steps 30 --record-video --provider openai-codex --model gpt-5.6-terra --run-label gpt56_terra_supervisor_mediation_smoke --policy-mode supervisor_mediation --output outputs`
- Outputs: planned `outputs/gpt56_terra_supervisor_mediation_smoke_3_agents_seed_39_steps_30/`.
- Results: not run.
- Caveats: a 30-step smoke cannot test a role rotation because the first 50-step assignment interval has not completed. Require a 100-step validation to prove that the step-50 mediator request includes the full epoch-0 assignment ledger and outcomes, and that assignments remain participant-only.
- Decision: implementation is ready for smoke validation; do not launch the longer remote-model run without reviewing the smoke and 100-step ledger validation.
- Next action: run the 30-step smoke and inspect serialized mediator/agent prompts, then request approval for the 100-step validation.
