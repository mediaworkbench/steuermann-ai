import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { HeartbeatSettingsSection } from "@/components/product/HeartbeatSettingsSection";
import { useI18n } from "@/hooks/useI18n";
import {
  fetchHeartbeatRate,
  fetchHeartbeatTasks,
  fetchHeartbeatRuns,
  updateHeartbeatCooldown,
} from "@/lib/api";

jest.mock("@/hooks/useI18n");
jest.mock("@/lib/api", () => ({
  fetchHeartbeatRate: jest.fn(),
  fetchHeartbeatTasks: jest.fn(),
  fetchHeartbeatRuns: jest.fn(),
  updateHeartbeatRate: jest.fn(),
  updateHeartbeatCooldown: jest.fn(),
}));

const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;
const mockFetchRate = fetchHeartbeatRate as jest.MockedFunction<typeof fetchHeartbeatRate>;
const mockFetchTasks = fetchHeartbeatTasks as jest.MockedFunction<typeof fetchHeartbeatTasks>;
const mockFetchRuns = fetchHeartbeatRuns as jest.MockedFunction<typeof fetchHeartbeatRuns>;
const mockUpdateCooldown = updateHeartbeatCooldown as jest.MockedFunction<typeof updateHeartbeatCooldown>;

const TASKS = [
  { name: "health", type: "pkg:Health", scope: "global" as const, cooldown_seconds: 0, cooldown_default: 0, cooldown_source: "default" as const, enabled: true, last_run: null },
  { name: "user_pulse", type: "pkg:Health", scope: "per_user" as const, cooldown_seconds: 300, cooldown_default: 300, cooldown_source: "default" as const, enabled: true, last_run: null },
];

const RUNS = [
  { task_name: "health", user_id: null, status: "ok", duration_ms: 12, fired_at: "2026-06-28T10:00:00Z", detail: {} },
  { task_name: "user_pulse", user_id: "u1", status: "ok", duration_ms: 11, fired_at: "2026-06-28T10:00:00Z", detail: {} },
  { task_name: "user_pulse", user_id: "u2", status: "error", duration_ms: 30, fired_at: "2026-06-28T10:00:00Z", detail: { phase: "observe" } },
];

// Body rows (excludes the header row).
function bodyRowCount(): number {
  const table = screen.getByRole("table");
  return within(table).getAllByRole("row").length - 1;
}

beforeEach(() => {
  jest.clearAllMocks();
  mockUseI18n.mockReturnValue({
    locale: "en",
    setLocale: jest.fn(),
    t: (key: string) => key,
    formatDate: () => "",
    formatTime: () => "",
    formatDateTime: () => "",
    formatNumber: (v: number) => String(v),
    formatRelativeTime: () => "1m ago",
  } as unknown as ReturnType<typeof useI18n>);
  mockFetchRate.mockResolvedValue({
    heartbeat_rate_minutes: 5,
    default_rate_minutes: 5,
    enabled: true,
    source: "default",
    last_run: null,
  });
  mockFetchTasks.mockResolvedValue(TASKS);
  mockFetchRuns.mockResolvedValue(RUNS);
});

test("renders the configured tasks (global + per-user) and the run log", async () => {
  render(<HeartbeatSettingsSection />);
  // Scope badges only appear in the configured-tasks list.
  expect(await screen.findByText("adminPage.heartbeatScopePerUser")).toBeInTheDocument();
  expect(screen.getByText("adminPage.heartbeatScopeGlobal")).toBeInTheDocument();
  await waitFor(() => expect(screen.getByRole("table")).toBeInTheDocument());
  expect(bodyRowCount()).toBe(3);
  // Loads the last 24h once (server-windowed); filtering/paging is client-side.
  expect(mockFetchRuns).toHaveBeenCalledWith({ hours: 24 });
});

test("status filter narrows the log client-side without refetching", async () => {
  render(<HeartbeatSettingsSection />);
  await waitFor(() => expect(screen.getByRole("table")).toBeInTheDocument());
  expect(bodyRowCount()).toBe(3);

  fireEvent.change(screen.getByLabelText("adminPage.heartbeatColStatus"), {
    target: { value: "error" },
  });

  expect(bodyRowCount()).toBe(1);
  // No extra fetch — filtering is purely client-side over the loaded rows.
  expect(mockFetchRuns).toHaveBeenCalledTimes(1);
});

test("user filter narrows the log to a single user", async () => {
  render(<HeartbeatSettingsSection />);
  await waitFor(() => expect(screen.getByRole("table")).toBeInTheDocument());

  fireEvent.change(screen.getByLabelText("adminPage.heartbeatColUser"), {
    target: { value: "u1" },
  });

  expect(bodyRowCount()).toBe(1);
});

test("editing a task cooldown saves the override", async () => {
  mockUpdateCooldown.mockResolvedValue([
    { ...TASKS[1], cooldown_seconds: 600, cooldown_source: "override" },
  ]);
  render(<HeartbeatSettingsSection />);
  // Wait for the per-task cooldown inputs (one per configured task).
  await waitFor(() =>
    expect(screen.getAllByLabelText("adminPage.heartbeatCooldownEditLabel")).toHaveLength(2),
  );

  // Two cooldown inputs (one per task); the second is user_pulse (value 300).
  const inputs = screen.getAllByLabelText("adminPage.heartbeatCooldownEditLabel");
  fireEvent.change(inputs[1], { target: { value: "600" } });

  // The Save next to the cooldown row (Save buttons: rate + per-task). Last one is user_pulse.
  const saveButtons = screen.getAllByText("adminPage.heartbeatSave");
  fireEvent.click(saveButtons[saveButtons.length - 1]);

  await waitFor(() => expect(mockUpdateCooldown).toHaveBeenCalledWith("user_pulse", 600));
});

test("run log paginates 25 rows per page", async () => {
  const many = Array.from({ length: 30 }, () => ({
    task_name: "health",
    user_id: null,
    status: "ok",
    duration_ms: 1,
    fired_at: "2026-06-30T10:00:00Z",
    detail: {},
  }));
  mockFetchRuns.mockResolvedValue(many);
  render(<HeartbeatSettingsSection />);
  await waitFor(() => expect(screen.getByRole("table")).toBeInTheDocument());

  // Page 1: first 25 of 30.
  expect(bodyRowCount()).toBe(25);
  // Next → page 2 shows the remaining 5.
  fireEvent.click(screen.getByText("adminPage.heartbeatLogNext"));
  expect(bodyRowCount()).toBe(5);
  // Previous → back to 25.
  fireEvent.click(screen.getByText("adminPage.heartbeatLogPrev"));
  expect(bodyRowCount()).toBe(25);
});
