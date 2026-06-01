import { renderHook, waitFor } from "@testing-library/react";
import React from "react";
import { RoleProvider, useRole } from "@/context/RoleContext";

// Mutable runtime mock — getters are re-evaluated on each property access,
// so we can change values between tests without resetting modules.
const mockRuntime = {
  AUTH_ENABLED: false,
  DEV_ROLE: "user" as "user" | "administrator",
};

jest.mock("@/lib/runtime", () => ({
  get AUTH_ENABLED() { return mockRuntime.AUTH_ENABLED; },
  get DEV_ROLE() { return mockRuntime.DEV_ROLE; },
}));

function makeWrapper() {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(RoleProvider, null, children);
  };
}

// Stub fetch before each test; restore after.
let fetchMock: jest.Mock;

beforeEach(() => {
  fetchMock = jest.fn();
  global.fetch = fetchMock;
  // Safe defaults
  mockRuntime.AUTH_ENABLED = false;
  mockRuntime.DEV_ROLE = "user";
});

afterEach(() => {
  // Remove the stub so other tests aren't affected
  delete (global as Record<string, unknown>).fetch;
});

describe("auth disabled", () => {
  test("resolves immediately as 'user' when DEV_ROLE is user", () => {
    mockRuntime.AUTH_ENABLED = false;
    mockRuntime.DEV_ROLE = "user";

    const { result } = renderHook(() => useRole(), { wrapper: makeWrapper() });

    expect(result.current.roleLoading).toBe(false);
    expect(result.current.role).toBe("user");
    expect(result.current.isAdmin).toBe(false);
  });

  test("resolves immediately as 'administrator' when DEV_ROLE is administrator", () => {
    mockRuntime.AUTH_ENABLED = false;
    mockRuntime.DEV_ROLE = "administrator";

    const { result } = renderHook(() => useRole(), { wrapper: makeWrapper() });

    expect(result.current.roleLoading).toBe(false);
    expect(result.current.role).toBe("administrator");
    expect(result.current.isAdmin).toBe(true);
  });

  test("does not call fetch when auth is disabled", () => {
    mockRuntime.AUTH_ENABLED = false;

    renderHook(() => useRole(), { wrapper: makeWrapper() });

    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("auth enabled", () => {
  beforeEach(() => {
    mockRuntime.AUTH_ENABLED = true;
    mockRuntime.DEV_ROLE = "user";
  });

  test("starts loading, then resolves to 'administrator' from session", async () => {
    fetchMock.mockResolvedValueOnce({
      json: async () => ({ role: "administrator" }),
    });

    const { result } = renderHook(() => useRole(), { wrapper: makeWrapper() });

    expect(result.current.roleLoading).toBe(true);

    await waitFor(() => expect(result.current.roleLoading).toBe(false));

    expect(result.current.role).toBe("administrator");
    expect(result.current.isAdmin).toBe(true);
  });

  test("resolves to 'user' when session returns user role", async () => {
    fetchMock.mockResolvedValueOnce({
      json: async () => ({ role: "user" }),
    });

    const { result } = renderHook(() => useRole(), { wrapper: makeWrapper() });

    await waitFor(() => expect(result.current.roleLoading).toBe(false));

    expect(result.current.role).toBe("user");
    expect(result.current.isAdmin).toBe(false);
  });

  test("falls back to 'user' for unknown role strings", async () => {
    fetchMock.mockResolvedValueOnce({
      json: async () => ({ role: "superuser" }),
    });

    const { result } = renderHook(() => useRole(), { wrapper: makeWrapper() });

    await waitFor(() => expect(result.current.roleLoading).toBe(false));

    expect(result.current.role).toBe("user");
  });

  test("falls back to 'user' and clears loading when fetch fails", async () => {
    fetchMock.mockRejectedValueOnce(new Error("network error"));

    const { result } = renderHook(() => useRole(), { wrapper: makeWrapper() });

    await waitFor(() => expect(result.current.roleLoading).toBe(false));

    expect(result.current.role).toBe("user");
    expect(result.current.isAdmin).toBe(false);
  });

  test("calls the session endpoint", async () => {
    fetchMock.mockResolvedValueOnce({
      json: async () => ({ role: "user" }),
    });

    renderHook(() => useRole(), { wrapper: makeWrapper() });

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/auth/session"));
  });
});

describe("default context (no provider)", () => {
  test("returns user / not admin / loading when used outside RoleProvider", () => {
    // No wrapper — uses the default context value
    const { result } = renderHook(() => useRole());

    expect(result.current.role).toBe("user");
    expect(result.current.isAdmin).toBe(false);
    expect(result.current.roleLoading).toBe(true);
  });
});
