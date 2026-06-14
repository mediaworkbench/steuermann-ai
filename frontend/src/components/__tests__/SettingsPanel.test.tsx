import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { axe } from "jest-axe";
import { SettingsPanel } from "@/components/SettingsPanel";
import { useI18n } from "@/hooks/useI18n";
import { useTheme } from "@/hooks/useTheme";
import { fetchSystemConfig } from "@/lib/api";

jest.mock("@/hooks/useI18n");
jest.mock("@/hooks/useTheme");
jest.mock("@/lib/api", () => ({
  fetchSystemConfig: jest.fn(),
}));

const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;
const mockUseTheme = useTheme as jest.MockedFunction<typeof useTheme>;
const mockFetchSystemConfig = fetchSystemConfig as jest.MockedFunction<typeof fetchSystemConfig>;

const BASE_SETTINGS = {
  user_id: "test-user",
  tool_toggles: { web_search_mcp: true },
  rag_config: { collection: "framework", top_k: 5, enabled: true },
  analytics_preferences: { sound_enabled: true },
  preferred_model: null,
  preferred_models: { chat: "" },
  language: "en",
  updated_at: null,
};

describe("SettingsPanel (user controls)", () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockUseTheme.mockReturnValue({
      theme: "auto",
      setTheme: jest.fn(),
      effectiveTheme: "light",
    });

    mockUseI18n.mockReturnValue({
      locale: "en",
      setLocale: jest.fn(),
      t: (key: string, params?: Record<string, string | number>) =>
        params ? `${key}|${Object.values(params).join("|")}` : key,
      formatDate: () => "",
      formatTime: () => "",
      formatDateTime: (value: string | number | Date) => String(value),
      formatNumber: (value: number) => String(value),
      formatRelativeTime: () => "",
    });

    mockFetchSystemConfig.mockResolvedValue({
      available_tools: [
        { id: "web_search_mcp", label: "Web Search" },
        { id: "calculator_tool", label: "Calculator" },
      ],
      rag_defaults: { collection_name: "framework", top_k: 5 },
      default_model: "openai/test-model",
      framework_version: "0.3.0",
      supported_languages: ["en", "de"],
      model_roles: [
        {
          role: "chat",
          provider_id: "lmstudio",
          default_model: "openai/test-model",
          available_models: ["openai/test-model", "openai/other-model"],
          model_load_error: null,
        },
        {
          role: "vision",
          provider_id: "lmstudio",
          default_model: "openai/vision-model",
          available_models: ["openai/vision-model"],
          model_load_error: null,
        },
      ],
      profile: {
        id: "starter",
        display_name: "Starter",
        role_label: "Assistant",
        description: null,
        app_name: "Steuermann",
        theme: { colors: {}, fonts: {}, radius: {}, custom_css_vars: {} },
      },
    });
  });

  test("renders language, sound, tool toggles, RAG and chat model sections — no a11y violations", async () => {
    const { container } = render(<SettingsPanel settings={BASE_SETTINGS} loading={false} onSave={jest.fn()} />);

    expect(await axe(container)).toHaveNoViolations();
    expect(screen.getByText("settingsPanel.interfaceSection")).toBeInTheDocument();
    expect(screen.getByText("settingsPanel.chatSection")).toBeInTheDocument();
    expect(screen.getByText("settingsPanel.toolSettings")).toBeInTheDocument();
    expect(screen.getByText("settingsPanel.ragConfiguration")).toBeInTheDocument();
    // Chat model section loads after system config
    await screen.findByText("settingsPanel.modelSelection");
  });

  test("shows only the chat model role, not vision or auxiliary", async () => {
    render(<SettingsPanel settings={BASE_SETTINGS} loading={false} onSave={jest.fn()} />);

    // The chat-role model selector appears once system config loads.
    await screen.findByText("settingsPanel.roleModelLabel|chat");
    // Vision/auxiliary roles are admin-only — never shown in user settings.
    expect(screen.queryByText("settingsPanel.roleModelLabel|vision")).not.toBeInTheDocument();
    expect(screen.queryByText("settingsPanel.roleModelLabel|auxiliary")).not.toBeInTheDocument();
  });

  test("does not render LLM capabilities table (admin feature)", async () => {
    render(<SettingsPanel settings={BASE_SETTINGS} loading={false} onSave={jest.fn()} />);

    await screen.findByText("settingsPanel.language");
    expect(screen.queryByText("settingsPanel.capabilitiesTitle")).not.toBeInTheDocument();
    expect(screen.queryByText("settingsPanel.copyDiagnostics")).not.toBeInTheDocument();
  });

  test("does not render reingest or reset buttons (admin feature)", async () => {
    render(<SettingsPanel settings={BASE_SETTINGS} loading={false} onSave={jest.fn()} />);

    await screen.findByText("settingsPanel.language");
    expect(screen.queryByText("settingsPanel.reingestAllDocuments")).not.toBeInTheDocument();
    expect(screen.queryByText("settingsPanel.resetAllDatabases")).not.toBeInTheDocument();
  });

  test("toggles tool checkbox and calls onSave with updated toggles", async () => {
    const onSave = jest.fn().mockResolvedValue(true);
    render(<SettingsPanel settings={BASE_SETTINGS} loading={false} onSave={onSave} />);

    await screen.findByText("Web Search");
    const checkbox = screen.getByRole("checkbox", { name: /web search/i });
    fireEvent.click(checkbox);
    fireEvent.click(screen.getByText("settingsPanel.saveSettings"));

    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith(
        expect.objectContaining({
          tool_toggles: expect.objectContaining({ web_search_mcp: false }),
        })
      );
    });
  });

  test("save payload includes all settings fields (read-modify-write)", async () => {
    const onSave = jest.fn().mockResolvedValue(true);
    render(<SettingsPanel settings={BASE_SETTINGS} loading={false} onSave={onSave} />);

    await screen.findByText("settingsPanel.saveSettings");
    fireEvent.click(screen.getByText("settingsPanel.saveSettings"));

    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith(
        expect.objectContaining({
          tool_toggles: expect.any(Object),
          rag_config: expect.any(Object),
          preferred_models: expect.any(Object),
          theme: expect.any(String),
          language: expect.any(String),
          analytics_preferences: expect.any(Object),
        })
      );
    });
  });
});
