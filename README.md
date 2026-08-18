# Cleanup grid-context LLM

## Objective

Test a **text/grid-context policy** for Melting Pot's `clean_up` environment while vision work is paused.

At each decision step, the LLM receives a structured, egocentric grid message describing its permitted local environment rather than an RGB image. It returns a validated action (and optionally a short public message). The grid is derived from simulator state for this controlled experiment.

This is an **oracle-context policy experiment**, not a vision result and not a fair image-only comparison. It isolates whether the language/action component can use a correct symbolic environment representation before we spend more time diagnosing visual grounding.

## Initial contract

**Policy input:**
- current egocentric local grid;
- agent position/orientation in local coordinates;
- visible objects and other agents;
- permitted local state such as own reward/history and ready-to-shoot status;
- valid action set.

**Policy must not receive:**
- global/world RGB;
- global coordinates or world-state outside the egocentric observation window;
- future state, future rewards, or simulator-derived optimal actions;
- other agents' hidden state.

## Grid message sketch

```json
{
  "frame": 120,
  "self": {"row": 5, "col": 5, "orientation": "NORTH"},
  "view": {"height": 11, "width": 11},
  "cells": [
    {"row": 4, "col": 5, "terrain": "river", "objects": ["dirt_live"]},
    {"row": 5, "col": 5, "terrain": "grass", "objects": ["self"]}
  ],
  "valid_actions": ["NOOP", "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT", "TURN_LEFT", "TURN_RIGHT", "FIRE_CLEAN", "FIRE_ZAP"]
}
```

The exact vocabulary must be frozen in a versioned schema before running comparisons.

## Layout

```text
src/       Grid serialisation, prompts, providers, runner adapters, metrics
configs/   Reproducible experiment configurations
data/      Local manifests only; do not copy raw generated datasets unnecessarily
outputs/   Run artifacts, trajectories, summaries and plots
tests/     Schema, visibility, leakage and action-validation tests
docs/      Protocol and design notes
```

## First milestone

Create and test a deterministic adapter from the existing Cleanup semantic exporter to an egocentric JSON grid message. Verify that the grid contains no information outside the policy-visible local window, then run a short mock-policy episode before attaching an LLM.
