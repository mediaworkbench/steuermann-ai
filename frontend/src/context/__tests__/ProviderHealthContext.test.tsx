import { render, screen, waitFor, act } from "@testing-library/react";
import React from "react";
import { ProviderHealthProvider, useProviderHealth } from "@/context/ProviderHealthContext";
import { fetchProviderHealth, triggerLLMReprobe } from "@/lib/api";

jest.mock("@/lib/api", () => ({
  fetchProviderHealth: jest.fn(),
  triggerLLMReprobe: jest.fn(),
}));

const mockFetch = fetchProviderHealth as jest.MockedFunction<typeof fetchProviderHealth>;
const mockReprobe = triggerLLMReprobe as jest.MockedFunction<typeof triggerLLMReprobe>;

beforeEach(() => {
  jest.clearAllMocks();
  mockReprobe.mockResolvedValue(undefined);
});

function Probe() {
  const { status, refresh } = useProviderHealth();
  return (
    <div>
      <span data-testid="status">{status}</span>
      <button onClick={refresh}>retry</button>
    </div>
  );
}

function renderProvider() {
  return render(
    <ProviderHealthProvider>
      <Probe />
    </ProviderHealthProvider>,
  );
}

test("reflects an offline provider on initial poll", async () => {
  mockFetch.mockResolvedValue({ status: "offline", providers: [] });
  renderProvider();
  await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("offline"));
});

test("a null response (network/proxy failure) is treated as offline", async () => {
  mockFetch.mockResolvedValue(null);
  renderProvider();
  await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("offline"));
});

test("retry re-checks, recovers, and fires a capability reprobe", async () => {
  mockFetch.mockResolvedValueOnce({ status: "offline", providers: [] });
  renderProvider();
  await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("offline"));

  mockFetch.mockResolvedValue({ status: "online", providers: [] });
  await act(async () => {
    screen.getByText("retry").click();
  });

  await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("online"));
  expect(mockReprobe).toHaveBeenCalled();
});

test("surfaces a degraded provider state", async () => {
  mockFetch.mockResolvedValue({
    status: "degraded",
    providers: [{ roles: ["chat"], api_base: "http://x/v1", reachable: true, detail: "HTTP 200" }],
  });
  renderProvider();
  await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("degraded"));
});
