import { render, screen, waitFor, act } from "@testing-library/react";
import React from "react";
import { SessionProvider, useSession } from "@/context/SessionContext";
import { hardNavigate } from "@/lib/navigation";

jest.mock("@/lib/navigation", () => ({ hardNavigate: jest.fn() }));
const mockNavigate = hardNavigate as jest.Mock;

let fetchMock: jest.Mock;

beforeEach(() => {
  jest.clearAllMocks();
  fetchMock = jest.fn();
  global.fetch = fetchMock;
});

afterEach(() => {
  delete (global as Record<string, unknown>).fetch;
});

function sessionResponse(body: Record<string, unknown>) {
  return { json: async () => body };
}

function Probe() {
  const { role, userId } = useSession();
  return <div data-testid="probe">{`${userId}:${role}`}</div>;
}

function renderWithProvider() {
  return render(
    <SessionProvider>
      <Probe />
    </SessionProvider>
  );
}

test("resolves identity + role from the session endpoint", async () => {
  fetchMock.mockResolvedValueOnce(
    sessionResponse({ enabled: true, user: { userId: "u1", username: "u1", role: "researcher" } })
  );
  renderWithProvider();
  await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("u1:researcher"));
});

test("refetches on tab focus and reflects a role change without reload", async () => {
  fetchMock
    .mockResolvedValueOnce(
      sessionResponse({ enabled: true, user: { userId: "u1", username: "u1", role: "administrator" } })
    )
    .mockResolvedValueOnce(
      sessionResponse({ enabled: true, user: { userId: "u1", username: "u1", role: "user" } })
    );
  renderWithProvider();
  await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("u1:administrator"));

  await act(async () => {
    document.dispatchEvent(new Event("visibilitychange"));
  });

  await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("u1:user"));
  expect(mockNavigate).not.toHaveBeenCalled();
});

test("logs out (redirects to /login) when a refetch shows the session was revoked", async () => {
  fetchMock
    .mockResolvedValueOnce(
      sessionResponse({ enabled: true, user: { userId: "u1", username: "u1", role: "administrator" } })
    )
    .mockResolvedValueOnce(sessionResponse({ enabled: true, user: null }));
  renderWithProvider();
  await waitFor(() => expect(screen.getByTestId("probe")).toHaveTextContent("u1:administrator"));

  await act(async () => {
    document.dispatchEvent(new Event("visibilitychange"));
  });

  await waitFor(() => expect(mockNavigate).toHaveBeenCalledWith("/login"));
});
