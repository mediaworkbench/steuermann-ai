import { render, screen } from "@testing-library/react";
import { ToolsMenu } from "@/components/chat/ToolsMenu";
import type { SystemConfig } from "@/lib/api";

const systemConfig = {
  available_tools: [
    { id: "web_search_mcp", label: "Web Search", group: "text" },
    { id: "datetime_tool", label: "Datetime", group: "auxiliary" },
  ],
  rag_defaults: { collection_name: "framework", top_k: 5 },
  default_model: "openai/test-model",
  framework_version: "0.0.0",
  supported_languages: ["en"],
  model_roles: [],
  profile: {
    id: "starter",
    display_name: "Starter",
    role_label: "Assistant",
    theme: { colors: {}, fonts: {}, radius: {}, custom_css_vars: {} },
  },
} as unknown as SystemConfig;

const noop = () => {};

describe("ToolsMenu", () => {
  test("hides tools not in allowedTools", () => {
    render(
      <ToolsMenu
        open
        onToggle={noop}
        onClose={noop}
        systemConfig={systemConfig}
        toolToggles={{}}
        onToolToggle={noop}
        allowedTools={["datetime_tool"]}
      />
    );
    expect(screen.getByText("Datetime")).toBeInTheDocument();
    expect(screen.queryByText("Web Search")).not.toBeInTheDocument();
  });

  test("shows all tools when allowedTools is null", () => {
    render(
      <ToolsMenu
        open
        onToggle={noop}
        onClose={noop}
        systemConfig={systemConfig}
        toolToggles={{}}
        onToolToggle={noop}
        allowedTools={null}
      />
    );
    expect(screen.getByText("Datetime")).toBeInTheDocument();
    expect(screen.getByText("Web Search")).toBeInTheDocument();
  });
});
