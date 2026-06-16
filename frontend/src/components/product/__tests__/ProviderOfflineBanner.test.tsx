import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { ProviderOfflineBanner } from "@/components/product/ProviderOfflineBanner";
import { useProviderHealth } from "@/context/ProviderHealthContext";
import { useI18n } from "@/hooks/useI18n";

jest.mock("@/context/ProviderHealthContext", () => ({ useProviderHealth: jest.fn() }));
jest.mock("@/hooks/useI18n");

const mockUseProviderHealth = useProviderHealth as jest.MockedFunction<typeof useProviderHealth>;
const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;

function mockHealth(overrides: Partial<ReturnType<typeof useProviderHealth>>) {
  mockUseProviderHealth.mockReturnValue({
    status: "offline",
    providers: [],
    loading: false,
    lastCheckedAt: null,
    refresh: jest.fn(),
    ...overrides,
  });
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
    formatNumber: (value: number) => String(value),
    formatRelativeTime: () => "",
  } as unknown as ReturnType<typeof useI18n>);
});

test("renders nothing when the provider is online", () => {
  mockHealth({ status: "online" });
  const { container } = render(<ProviderOfflineBanner />);
  expect(container).toBeEmptyDOMElement();
});

test("shows the offline message when the provider is offline", () => {
  mockHealth({ status: "offline" });
  render(<ProviderOfflineBanner />);
  expect(screen.getByText("providerHealth.offlineTitle")).toBeInTheDocument();
});

test("shows the degraded message when degraded", () => {
  mockHealth({ status: "degraded" });
  render(<ProviderOfflineBanner />);
  expect(screen.getByText("providerHealth.degradedTitle")).toBeInTheDocument();
});

test("Retry triggers refresh", () => {
  const refresh = jest.fn();
  mockHealth({ status: "offline", refresh });
  render(<ProviderOfflineBanner />);
  screen.getByRole("button").click();
  expect(refresh).toHaveBeenCalledTimes(1);
});

test("Retry is disabled while a check is in flight", () => {
  mockHealth({ status: "offline", loading: true });
  render(<ProviderOfflineBanner />);
  expect(screen.getByRole("button")).toBeDisabled();
});

test("has no accessibility violations", async () => {
  mockHealth({ status: "offline" });
  const { container } = render(<ProviderOfflineBanner />);
  expect(await axe(container)).toHaveNoViolations();
});
