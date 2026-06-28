import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { axe } from "jest-axe";
import { AdminPanel } from "@/components/AdminPanel";
import { useI18n } from "@/hooks/useI18n";
import {
  fetchLLMCapabilities,
  fetchSystemConfig,
  fetchRoleTools,
  resetAllDatabases,
} from "@/lib/api";

jest.mock("@/hooks/useI18n");
jest.mock("@/lib/api", () => ({
  fetchLLMCapabilities: jest.fn(),
  fetchSystemConfig: jest.fn(),
  fetchRoleTools: jest.fn(),
  updateRoleTools: jest.fn(),
  fetchHeartbeatRate: jest.fn(),
  updateHeartbeatRate: jest.fn(),
  fetchHeartbeatTasks: jest.fn(),
  fetchHeartbeatRuns: jest.fn(),
  triggerReingestAllDocuments: jest.fn(),
  resetAllDatabases: jest.fn(),
}));

const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;
const mockFetchSystemConfig = fetchSystemConfig as jest.MockedFunction<typeof fetchSystemConfig>;
const mockFetchLLMCapabilities = fetchLLMCapabilities as jest.MockedFunction<typeof fetchLLMCapabilities>;
const mockFetchRoleTools = fetchRoleTools as jest.MockedFunction<typeof fetchRoleTools>;
const mockResetAllDatabases = resetAllDatabases as jest.MockedFunction<typeof resetAllDatabases>;

const BASE_SETTINGS = {
  user_id: "test-user",
  tool_toggles: {},
  rag_config: { collection: "framework", top_k: 5, enabled: true },
  analytics_preferences: {},
  preferred_model: null,
  preferred_models: { chat: "", vision: "", auxiliary: "" },
  language: "en",
  updated_at: null,
};

const CAPABILITY_ITEM = {
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
  supports_vision: false,
  supports_reasoning: false,
  capabilities: {},
};

describe("AdminPanel", () => {
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
      t: (key: string, params?: Record<string, string | number>) =>
        params ? `${key}|${Object.values(params).join("|")}` : key,
      formatDate: () => "",
      formatTime: () => "",
      formatDateTime: (value: string | number | Date) => String(value),
      formatNumber: (value: number) => String(value),
      formatRelativeTime: () => "",
    });

    mockFetchRoleTools.mockResolvedValue({
      tools: [
        { id: "web_search_mcp", label: "Web Search", group: "text" },
        { id: "datetime_tool", label: "Datetime", group: "auxiliary" },
      ],
      roles: { user: ["web_search_mcp"], researcher: [] },
    });

    mockFetchSystemConfig.mockResolvedValue({
      available_tools: [{ id: "web_search_mcp", label: "Web Search", group: "text" }],
      rag_defaults: { collection_name: "framework", top_k: 5 },
      default_model: "openai/test-model",
      framework_version: "0.3.0",
      supported_languages: ["en"],
      model_roles: [
        {
          role: "chat",
          provider_id: "lmstudio",
          default_model: "openai/test-model",
          available_models: ["openai/test-model"],
          model_load_error: null,
        },
        {
          role: "vision",
          provider_id: "lmstudio",
          default_model: "openai/vision-model",
          available_models: ["openai/vision-model"],
          model_load_error: null,
        },
        {
          role: "auxiliary",
          provider_id: "lmstudio",
          default_model: "openai/aux-model",
          available_models: ["openai/aux-model"],
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

  test("shows model capability rows from backend — no a11y violations", async () => {
    mockFetchLLMCapabilities.mockResolvedValue({
      status: "ok",
      profile_id: "starter",
      probe_ttl_seconds: 3600,
      items: [CAPABILITY_ITEM],
    });

    const { container } = render(<AdminPanel settings={BASE_SETTINGS} loading={false} onSave={jest.fn()} />);

    await screen.findAllByText("openai/test-model");
    expect(await axe(container)).toHaveNoViolations();
    expect(screen.getByText("probe_stale_forced_structured")).toBeInTheDocument();
    // The TTL label is rendered with an interpolated value (key|seconds in the test t()).
    expect(screen.getByText((c) => c.includes("settingsPanel.capabilitiesTtl"))).toBeInTheDocument();
    expect(screen.queryByText("settingsPanel.capabilitiesLoading")).not.toBeInTheDocument();
  });

  test("copies diagnostics payload to clipboard", async () => {
    mockFetchLLMCapabilities.mockResolvedValue({
      status: "ok",
      profile_id: "starter",
      probe_ttl_seconds: 3600,
      items: [CAPABILITY_ITEM],
    });

    render(<AdminPanel settings={BASE_SETTINGS} loading={false} onSave={jest.fn()} />);

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
          ...CAPABILITY_ITEM,
          probe_status: "warning",
          capability_mismatch: true,
          supports_bind_tools: false,
          error_message: "bind_tools_failed: test",
        },
      ],
    });

    render(<AdminPanel settings={BASE_SETTINGS} loading={false} onSave={jest.fn()} />);

    await screen.findByText("probe_stale_forced_structured");
    fireEvent.click(screen.getByRole("button", { name: "settingsPanel.showDetails" }));

    expect(screen.getByText((content) => content.includes("settingsPanel.detailConfiguredMode"))).toBeInTheDocument();
    expect(screen.getByText((content) => content.includes("settingsPanel.detailApiBase"))).toBeInTheDocument();
    expect(screen.getByText("http://host.docker.internal:1234/v1")).toBeInTheDocument();
    expect(screen.getByText("bind_tools_failed: test")).toBeInTheDocument();
  });

  test("shows capabilities load error if endpoint fails", async () => {
    mockFetchLLMCapabilities.mockResolvedValue(null);

    render(<AdminPanel settings={BASE_SETTINGS} loading={false} onSave={jest.fn()} />);

    await waitFor(() => {
      expect(screen.getByText("settingsPanel.capabilitiesLoadFailed")).toBeInTheDocument();
    });
  });

  test("danger zone renders all five category checkboxes checked by default", async () => {
    mockFetchLLMCapabilities.mockResolvedValue({ status: "ok", profile_id: "starter", probe_ttl_seconds: 3600, items: [] });

    render(<AdminPanel settings={BASE_SETTINGS} loading={false} onSave={jest.fn()} />);

    await screen.findByText("adminPage.dangerZoneSection");

    const checkboxes = screen.getAllByRole("checkbox");
    // 5 danger-zone checkboxes — all checked by default
    const dangerCheckboxes = checkboxes.filter((cb) => {
      const parent = cb.closest("label");
      return parent?.textContent?.includes("adminPage.reset");
    });
    expect(dangerCheckboxes.length).toBe(5);
    dangerCheckboxes.forEach((cb) => expect(cb).toBeChecked());
  });

  test("unchecking a category and clicking reset passes only selected options", async () => {
    mockFetchLLMCapabilities.mockResolvedValue({ status: "ok", profile_id: "starter", probe_ttl_seconds: 3600, items: [] });
    mockResetAllDatabases.mockResolvedValue({ status: "ok", errors: [] });

    render(<AdminPanel settings={BASE_SETTINGS} loading={false} onSave={jest.fn()} />);

    await screen.findByText("adminPage.dangerZoneSection");

    // Uncheck analytics
    const allCheckboxes = screen.getAllByRole("checkbox");
    const analyticsCheckbox = allCheckboxes.find(
      (cb) => cb.closest("label")?.textContent?.includes("adminPage.resetAnalyticsLabel")
    )!;
    fireEvent.click(analyticsCheckbox);
    expect(analyticsCheckbox).not.toBeChecked();

    // Uncheck llm_probes
    const probesCheckbox = allCheckboxes.find(
      (cb) => cb.closest("label")?.textContent?.includes("adminPage.resetLlmProbesLabel")
    )!;
    fireEvent.click(probesCheckbox);
    expect(probesCheckbox).not.toBeChecked();

    // Open confirm dialog — click the single trigger button (dialog not yet open)
    fireEvent.click(screen.getByRole("button", { name: "adminPage.resetSelectedButton" }));

    // ConfirmDialog is now open. Check the "I understand" checkbox inside it.
    // It's the checkbox associated with confirmDialog.resetCheckboxLabel text.
    const dialogCheckbox = screen.getByLabelText("confirmDialog.resetCheckboxLabel");
    fireEvent.click(dialogCheckbox);

    // Now the confirm button inside the dialog is enabled — it's the second button
    // with this name (trigger button + dialog confirm button)
    const confirmButtons = screen.getAllByRole("button", { name: "adminPage.resetSelectedButton" });
    const dialogConfirm = confirmButtons[confirmButtons.length - 1];
    fireEvent.click(dialogConfirm);

    await waitFor(() => {
      expect(mockResetAllDatabases).toHaveBeenCalledWith({
        conversations: true,
        workspace: true,
        memories: true,
        analytics: false,
        llm_probes: false,
      });
    });
  });

  test("shows only vision and auxiliary model roles, not chat", async () => {
    mockFetchLLMCapabilities.mockResolvedValue({ status: "ok", profile_id: "starter", probe_ttl_seconds: 3600, items: [] });

    render(<AdminPanel settings={BASE_SETTINGS} loading={false} onSave={jest.fn()} />);

    await screen.findByText("adminPage.modelSection");
    // The admin system-model section exposes vision + auxiliary roles…
    await screen.findByText("settingsPanel.roleModelLabel|vision");
    expect(screen.getByText("settingsPanel.roleModelLabel|auxiliary")).toBeInTheDocument();
    // …but never the user-facing chat role.
    expect(screen.queryByText("settingsPanel.roleModelLabel|chat")).not.toBeInTheDocument();
  });
});
