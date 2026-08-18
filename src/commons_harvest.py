"""Commons Harvest environment construction and action decoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


COMMONS_HARVEST_SIGNATURES: Mapping[str, Mapping[str, int]] = {
    "NOOP": {"move": 0, "turn": 0, "fireZap": 0},
    "FORWARD": {"move": 1, "turn": 0, "fireZap": 0},
    "BACKWARD": {"move": 3, "turn": 0, "fireZap": 0},
    "STEP_LEFT": {"move": 4, "turn": 0, "fireZap": 0},
    "STEP_RIGHT": {"move": 2, "turn": 0, "fireZap": 0},
    "TURN_LEFT": {"move": 0, "turn": -1, "fireZap": 0},
    "TURN_RIGHT": {"move": 0, "turn": 1, "fireZap": 0},
    "FIRE_ZAP": {"move": 0, "turn": 0, "fireZap": 1},
}


@dataclass(frozen=True)
class ActionMapping:
    name_to_index: Mapping[str, int]

    def index(self, name: str) -> int:
        return self.name_to_index[name]


def action_mapping_from_action_set(action_set: Sequence[Mapping[str, int]]) -> ActionMapping:
    """Resolve named policy actions from the substrate's live action table."""
    actual = [{key: int(value) for key, value in item.items()} for item in action_set]
    resolved: dict[str, int] = {}
    for name, signature in COMMONS_HARVEST_SIGNATURES.items():
        matches = [index for index, item in enumerate(actual) if item == dict(signature)]
        if len(matches) != 1:
            raise RuntimeError(f"Commons Harvest action {name} did not resolve uniquely: {matches}")
        resolved[name] = matches[0]
    if len(resolved) != len(actual) or len(set(resolved.values())) != len(resolved):
        raise RuntimeError(f"unexpected Commons Harvest action set: resolved={resolved}, actual={actual}")
    return ActionMapping(resolved)


def build_commons_harvest(players: int, *, env_seed: int | None = None):
    """Build the standard open Commons Harvest substrate with RGB local views."""
    if not 1 <= players <= 7:
        raise ValueError("players must be in [1, 7]")
    from meltingpot import substrate
    from meltingpot.configs.substrates import commons_harvest__open
    from meltingpot.utils.substrates import builder
    from meltingpot.utils.substrates import substrate as substrate_lib
    from meltingpot.utils.substrates.wrappers import collective_reward_wrapper
    from meltingpot.utils.substrates.wrappers import discrete_action_wrapper
    from meltingpot.utils.substrates.wrappers import multiplayer_wrapper
    from meltingpot.utils.substrates.wrappers import observables_wrapper

    config = commons_harvest__open.get_config()

    def settings_builder(*, roles, config):
        return commons_harvest__open.build(roles=roles, config=config)

    with config.unlocked():
        config.lab2d_settings_builder = settings_builder
    factory = substrate.get_factory_from_config(config.lock())
    settings = factory._lab2d_settings_builder(("default",) * players)
    env = builder.builder(settings, env_seed=env_seed)
    env = observables_wrapper.ObservablesWrapper(env)
    env = multiplayer_wrapper.Wrapper(
        env,
        individual_observation_names=config.individual_observation_names,
        global_observation_names=config.global_observation_names,
    )
    env = discrete_action_wrapper.Wrapper(env, action_table=config.action_set)
    env = collective_reward_wrapper.CollectiveRewardWrapper(env)
    return substrate_lib.Substrate(env), action_mapping_from_action_set(config.action_set)
