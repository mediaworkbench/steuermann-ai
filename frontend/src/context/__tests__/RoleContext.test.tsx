import { renderHook, waitFor } from "@testing-library/react";
import React from "react";
import { SessionProvider } from "@/context/SessionContext";
import { useRole } from "@/context/RoleContext";

// useRole is now a thin view over SessionContext, which resolves identity + role from
// GET /api/auth/session. We stub that fetch and assert the derived role flags.

let fetchMock: jest.Mock;

beforeEach(() => {
  fetchMock = jest.fn();
  global.fetch = fetchMock;
});

afterEach(() => {
  delete (global as Record<string, unknown>).fetch;
});

function wrapper({ children }: { children: React.ReactNode }) {
  return React.createElement(SessionProvider, null, children);
}

function mockSession(user: Record<string, unknown> | null) {
  fetchMock.mockResolvedValueOnce({ json: async () => ({ user }) });
}

describe("useRole via SessionProvider", () => {
  test("administrator gets admin + rag access", async () => {
    mockSession({ userId: "a", username: "a", role: "administrator", mustChangePassword: false });
    const { result } = renderHook(() => useRole(), { wrapper });
    await waitFor(() => expect(result.current?.role).toBe("administrator"));
    expect(result.current.isAdmin).toBe(true);
    expect(result.current.canAccessRag).toBe(true);
    expect(result.current.roleLoading).toBe(false);
  });

  test("researcher gets rag access but is not admin", async () => {
    mockSession({ userId: "r", username: "r", role: "researcher" });
    const { result } = renderHook(() => useRole(), { wrapper });
    await waitFor(() => expect(result.current?.role).toBe("researcher"));
    expect(result.current.isAdmin).toBe(false);
    expect(result.current.canAccessRag).toBe(true);
  });

  test("basic user has no admin or rag access", async () => {
    mockSession({ userId: "u", username: "u", role: "user" });
    const { result } = renderHook(() => useRole(), { wrapper });
    await waitFor(() => expect(result.current?.role).toBe("user"));
    expect(result.current.isAdmin).toBe(false);
    expect(result.current.canAccessRag).toBe(false);
  });

  test("unknown role strings fall back to 'user'", async () => {
    mockSession({ userId: "x", username: "x", role: "superuser" });
    const { result } = renderHook(() => useRole(), { wrapper });
    await waitFor(() => expect(result.current?.role).toBe("user"));
  });

  test("calls the session endpoint", async () => {
    mockSession({ userId: "u", username: "u", role: "user" });
    renderHook(() => useRole(), { wrapper });
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("/api/auth/session", expect.anything())
    );
  });
});

describe("default context (no provider)", () => {
  test("returns user / not admin / loading when used outside SessionProvider", () => {
    const { result } = renderHook(() => useRole());
    expect(result.current.role).toBe("user");
    expect(result.current.isAdmin).toBe(false);
    expect(result.current.canAccessRag).toBe(false);
    expect(result.current.roleLoading).toBe(true);
  });
});
