import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { SettingsPanel } from "@/components/SettingsPanel";
import { useI18n } from "@/hooks/useI18n";
import {
  fetchAvailableModels,
  fetchLLMCapabilities,
  fetchSystemConfig,
} from "@/lib/api";

jest.mock("@/hooks/useI18n");
jest.mock("@/lib/api", () => ({
  fetchAvailableModels: jest.fn(),
  fetchLLMCapabilities: jest.fn(),
  fetchSystemConfig: jest.fn(),
  triggerReingestAllDocuments: jest.fn(),
}));

const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;
const mockFetchSystemConfig = fetchSystemConfig as jest.MockedFunction<typeof fetchSystemConfig>;
const mockFetchAvailableModels = fetchAvailableModels as jest.MockedFunction<typeof fetchAvailableModels>;
const mockFetchLLMCapabilities = fetchLLMCapabilities as jest.MockedFunction<typeof fetchLLMCapabilities>;

describe("SettingsPanel", () => {
  const writeTextMock = jest.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    jest.clearAllMocks();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: writeTextMock },
    });

    mockUseI18n.mockReturnValue({
      locale: "en",
      setLocale: jest.fn(),
      t: (key: string) => key,
      formatDate: () => "",
      formatTime: () => "",
      formatDateTime: (value: string | number | Date) => String(value),
      formatNumber: (value: number) => String(value),
      formatRelativeTime: () => "",
    });

    mockFetchSystemConfig.mockResolvedValue({
      available_tools: [{ id: "web_search_mcp", label: "Web Search" }],
      rag_defaults: { collection_name: "framework", top_k: 5 },
      default_model: "openai/test-model",
      framework_version: "0.2.3",
      supported_languages: ["en"],
      profile: {
        id: "starter",
        display_name: "Starter",
        role_label: "Assistant",
        description: null,
        app_name: "Steuermann",
        theme: {
          colors: {},
          fonts: {},
          radius: {},
          custom_css_vars: {},
        },
      },
    });

    mockFetchAvailableModels.mockResolvedValue(["openai/test-model"]);
  });

  test("shows model capability rows from backend", async () => {
    mockFetchLLMCapabilities.mockResolvedValue({
      status: "ok",
      profile_id: "starter",
      probe_ttl_seconds: 3600,
      items: [
        {
          provider_id: "primary",
          model_name: "openai/test-model",
          desired_mode: "native",
          configured_tool_calling_mode: "native",
          effective_mode: "structured",
          effective_mode_reason: "probe_stale_forced_structured",
          probe_status: "success",
          capability_mismatch: false,
          supports_bind_tools: true,
          supports_tool_schema: true,
          api_base: "http://host.docker.internal:1234/v1",
          error_message: null,
          metadata: { probe_kind: "native_bind_tools" },
          probed_at: "2026-05-11T10:20:30Z",
          capabilities: {},
        },
      ],
    });

    render(
      <SettingsPanel
        settings={{
          user_id: "test-user",
          tool_toggles: {},
          rag_config: { collection: "framework", top_k: 5 },
          analytics_preferences: {},
          preferred_model: null,
          language: "en",
          updated_at: null,
        }}
        loading={false}
        onSave={jest.fn().mockResolvedValue(true)}
      />
    );

    expect((await screen.findAllByText("openai/test-model")).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("probe_stale_forced_structured")).toBeInTheDocument();
    expect(screen.getByText("settingsPanel.capabilitiesTtl")).toBeInTheDocument();
    expect(screen.getByText("settingsPanel.legendNative")).toBeInTheDocument();
    expect(screen.getByText("settingsPanel.legendStructured")).toBeInTheDocument();
    expect(screen.getByText("settingsPanel.legendReact")).toBeInTheDocument();
    expect(screen.queryByText("settingsPanel.capabilitiesLoading")).not.toBeInTheDocument();
  });

  test("copies diagnostics payload to clipboard", async () => {
    mockFetchLLMCapabilities.mockResolvedValue({
      status: "ok",
      profile_id: "starter",
      probe_ttl_seconds: 3600,
      items: [
        {
          provider_id: "primary",
          model_name: "openai/test-model",
          desired_mode: "native",
          configured_tool_calling_mode: "native",
          effective_mode: "structured",
          effective_mode_reason: "probe_stale_forced_structured",
          probe_status: "success",
          capability_mismatch: false,
          supports_bind_tools: true,
          supports_tool_schema: true,
          api_base: "http://host.docker.internal:1234/v1",
          error_message: null,
          metadata: { probe_kind: "native_bind_tools" },
          probed_at: "2026-05-11T10:20:30Z",
          capabilities: {},
        },
      ],
    });

    render(
      <SettingsPanel
        settings={{
          user_id: "test-user",
          tool_toggles: {},
          rag_config: { collection: "framework", top_k: 5 },
          analytics_preferences: {},
          preferred_model: null,
          language: "en",
          updated_at: null,
        }}
        loading={false}
        onSave={jest.fn().mockResolvedValue(true)}
      />
    );

    await screen.findByText("probe_stale_forced_structured");
    fireEvent.click(screen.getByRole("button", { name: "settingsPanel.copyDiagnostics" }));

    await waitFor(() => {
      expect(writeTextMock).toHaveBeenCalledTimes(1);
    });
    expect(writeTextMock.mock.calls[0][0]).toContain("probe_ttl_seconds\t3600");
    expect(writeTextMock.mock.calls[0][0]).toContain("openai/test-model");
    expect(writeTextMock.mock.calls[0][0]).toContain("probe_stale_forced_structured");
    expect(writeTextMock.mock.calls[0][0]).toContain("http://host.docker.internal:1234/v1");
  });

  test("shows expandable details for each capability row", async () => {
    mockFetchLLMCapabilities.mockResolvedValue({
      status: "ok",
      profile_id: "starter",
      probe_ttl_seconds: 3600,
      items: [
        {
          provider_id: "primary",
          model_name: "openai/test-model",
          desired_mode: "native",
          configured_tool_calling_mode: "native",
          effective_mode: "structured",
          effective_mode_reason: "probe_stale_forced_structured",
          probe_status: "warning",
          capability_mismatch: true,
          supports_bind_tools: false,
          supports_tool_schema: false,
          api_base: "http://host.docker.internal:1234/v1",
          error_message: "bind_tools_failed: test",
          metadata: { probe_kind: "native_bind_tools" },
          probed_at: "2026-05-11T10:20:30Z",
          capabilities: {},
        },
      ],
    });

    render(
      <SettingsPanel
        settings={{
          user_id: "test-user",
          tool_toggles: {},
          rag_config: { collection: "framework", top_k: 5 },
          analytics_preferences: {},
          preferred_model: null,
          language: "en",
          updated_at: null,
        }}
        loading={false}
        onSave={jest.fn().mockResolvedValue(true)}
      />
    );

    await screen.findByText("probe_stale_forced_structured");
    fireEvent.click(screen.getByRole("button", { name: "settingsPanel.showDetails" }));

    expect(screen.getByText((content) => content.includes("settingsPanel.detailConfiguredMode"))).toBeInTheDocument();
    expect(screen.getByText((content) => content.includes("settingsPanel.detailApiBase"))).toBeInTheDocument();
    expect(screen.getByText("http://host.docker.internal:1234/v1")).toBeInTheDocument();
    expect(screen.getByText("bind_tools_failed: test")).toBeInTheDocument();
    expect(screen.getByText("settingsPanel.detailMetadata")).toBeInTheDocument();
  });

  test("shows capabilities load error if endpoint fails", async () => {
    mockFetchLLMCapabilities.mockResolvedValue(null);

    render(
      <SettingsPanel
        settings={{
          user_id: "test-user",
          tool_toggles: {},
          rag_config: { collection: "framework", top_k: 5 },
          analytics_preferences: {},
          preferred_model: null,
          language: "en",
          updated_at: null,
        }}
        loading={false}
        onSave={jest.fn().mockResolvedValue(true)}
      />
    );

    await waitFor(() => {
      expect(screen.getByText("settingsPanel.capabilitiesLoadFailed")).toBeInTheDocument();
    });
  });
});
