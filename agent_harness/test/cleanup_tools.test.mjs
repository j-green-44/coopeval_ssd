import assert from "node:assert/strict";
import test from "node:test";

import { createBoundedCleanupTools } from "../src/cleanup_tools.mjs";


test("bounded Cleanup tools expose observation, history, and action only", async () => {
  const calls = [];
  const bridge = {
    observe: async () => ({ frame: 3, cells: [] }),
    history: async () => [{ action: "TURN_LEFT", reward: 0 }],
    act: async (action) => {
      calls.push(action);
      return { applied_action: action, reward: 0 };
    },
  };

  const tools = createBoundedCleanupTools(bridge, new Set(["NOOP", "FORWARD", "TURN_LEFT"]));
  assert.deepEqual(tools.map((tool) => tool.name), ["observe_grid", "recent_history", "act"]);

  const result = await tools[2].execute("call-1", { action: "TURN_LEFT" });
  assert.deepEqual(calls, ["TURN_LEFT"]);
  assert.match(result.content[0].text, /TURN_LEFT/);

  await assert.rejects(tools[2].execute("call-2", { action: "FIRE_ZAP" }), /invalid Cleanup action/);
});
