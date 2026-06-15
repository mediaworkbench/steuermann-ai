import { render, screen } from "@testing-library/react";
import { ToolsMenu } from "@/components/chat/ToolsMenu";
import { useI18n } from "@/hooks/useI18n";
import type { SystemConfig } from "@/lib/api";

jest.mock("@/hooks/useI18n");
const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;

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

function renderMenu(props: Partial<React.ComponentProps<typeof ToolsMenu>> = {}) {
  return render(
    <ToolsMenu
      open
      onToggle={noop}
      onClose={noop}
      systemConfig={systemConfig}
      toolToggles={{}}
      onToolToggle={noop}
      disabledTools={[]}
      allowedTools={null}
      {...props}
    />
  );
}

describe("ToolsMenu", () => {
  beforeEach(() => {
    mockUseI18n.mockReturnValue({
      locale: "en",
      setLocale: jest.fn(),
      t: (key: string) => key,
      formatDate: () => "",
      formatTime: () => "",
      formatDateTime: () => "",
      formatNumber: (value: number) => String(value),
      formatRelativeTime: () => "",
    });
  });

  test("hides tools not in allowedTools (role gate)", () => {
    renderMenu({ allowedTools: ["datetime_tool"] });
    expect(screen.getByText("Datetime")).toBeInTheDocument();
    expect(screen.queryByText("Web Search")).not.toBeInTheDocument();
  });

  test("hides tools disabled in Settings (membership)", () => {
    renderMenu({ toolToggles: { web_search_mcp: false } });
    expect(screen.getByText("Datetime")).toBeInTheDocument();
    expect(screen.queryByText("Web Search")).not.toBeInTheDocument();
  });

  test("per-chat disabled tools stay listed but show OFF", () => {
    renderMenu({ disabledTools: ["web_search_mcp"] });
    // Both remain visible — the per-chat toggle never hides a tool.
    expect(screen.getByText("Web Search")).toBeInTheDocument();
    expect(screen.getByText("Datetime")).toBeInTheDocument();
    // Web Search is OFF for this chat; Datetime is ON.
    expect(screen.getByText("OFF")).toBeInTheDocument();
    expect(screen.getByText("ON")).toBeInTheDocument();
  });

  test("shows an empty-state hint when every tool is disabled in Settings", () => {
    renderMenu({ toolToggles: { web_search_mcp: false, datetime_tool: false } });
    expect(screen.queryByText("Datetime")).not.toBeInTheDocument();
    expect(screen.queryByText("Web Search")).not.toBeInTheDocument();
    expect(screen.getByText("chat.noToolsEnabled")).toBeInTheDocument();
  });
});
