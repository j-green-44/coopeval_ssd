import { Type } from "@earendil-works/pi-ai";

const textResult = (value, details = {}) => ({
  content: [{ type: "text", text: JSON.stringify(value) }],
  details,
});

/**
 * Create the complete Pi tool surface for Cleanup. There are intentionally no
 * filesystem, shell, network, or arbitrary-code tools in this list.
 */
export function createBoundedCleanupTools(bridge, validActions) {
  return [
    {
      name: "observe_grid",
      label: "Observe current local grid",
      description: "Return the current egocentric symbolic Cleanup grid. This is the only environment observation.",
      parameters: Type.Object({}),
      executionMode: "sequential",
      execute: async () => textResult(await bridge.observe()),
    },
    {
      name: "recent_history",
      label: "Read recent action history",
      description: "Return recent attempted actions and their observed outcomes.",
      parameters: Type.Object({}),
      executionMode: "sequential",
      execute: async () => textResult(await bridge.history()),
    },
    {
      name: "act",
      label: "Apply one Cleanup action",
      description: "Apply exactly one permitted Cleanup action, then return its immediate observed outcome.",
      parameters: Type.Object({ action: Type.String({ description: "One allowed Cleanup action." }) }),
      executionMode: "sequential",
      execute: async (_toolCallId, { action }) => {
        if (!validActions.has(action)) {
          throw new Error(`invalid Cleanup action: ${action}`);
        }
        return textResult(await bridge.act(action), { action });
      },
    },
  ];
}
