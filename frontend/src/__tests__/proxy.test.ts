/** @jest-environment node */
import { NextRequest } from "next/server";
import { proxy } from "@/proxy";
import { getSessionFromCookieValue, isAuthEnabled } from "@/lib/auth/session";

jest.mock("@/lib/auth/session", () => ({
  getSessionCookieName: () => "uaf_session",
  isAuthEnabled: jest.fn(),
  getSessionFromCookieValue: jest.fn(),
}));

const mockAuthEnabled = isAuthEnabled as jest.Mock;
const mockSession = getSessionFromCookieValue as jest.Mock;

function req(path: string) {
  return new NextRequest(`http://localhost:3000${path}`);
}

function session(role: string, mustChangePassword = false) {
  return { userId: "u", username: "u", displayName: "u", email: "", role, mustChangePassword };
}

function locationOf(res: Response): string | null {
  return res.headers.get("location");
}

function pathOf(res: Response): string | null {
  const loc = locationOf(res);
  return loc ? new URL(loc).pathname : null;
}

beforeEach(() => {
  jest.clearAllMocks();
  mockAuthEnabled.mockReturnValue(true);
});

describe("auth disabled", () => {
  test("passes everything through", async () => {
    mockAuthEnabled.mockReturnValue(false);
    expect(locationOf(await proxy(req("/admin")))).toBeNull();
  });
});

describe("authentication", () => {
  test("unauthenticated page redirects to /login", async () => {
    mockSession.mockResolvedValue(null);
    expect(pathOf(await proxy(req("/chats")))).toBe("/login");
  });

  test("authenticated user reaches a normal page", async () => {
    mockSession.mockResolvedValue(session("user"));
    expect(locationOf(await proxy(req("/chats")))).toBeNull();
  });

  test("logged-in user visiting /login is sent home", async () => {
    mockSession.mockResolvedValue(session("administrator"));
    expect(pathOf(await proxy(req("/login")))).toBe("/");
  });
});

describe("forced password change", () => {
  test("redirects every page to /change-password while flagged", async () => {
    mockSession.mockResolvedValue(session("administrator", true));
    expect(pathOf(await proxy(req("/chats")))).toBe("/change-password");
  });

  test("allows the /change-password page itself", async () => {
    mockSession.mockResolvedValue(session("administrator", true));
    expect(locationOf(await proxy(req("/change-password")))).toBeNull();
  });

  test("sends flagged user from /login to /change-password", async () => {
    mockSession.mockResolvedValue(session("user", true));
    expect(pathOf(await proxy(req("/login")))).toBe("/change-password");
  });
});

describe("role gating", () => {
  test("basic user is blocked from /admin and /admin/rag and /metrics", async () => {
    mockSession.mockResolvedValue(session("user"));
    expect(pathOf(await proxy(req("/admin")))).toBe("/");
    expect(pathOf(await proxy(req("/admin/rag")))).toBe("/");
    expect(pathOf(await proxy(req("/metrics")))).toBe("/");
  });

  test("researcher can reach /admin/rag but not /admin or /metrics", async () => {
    mockSession.mockResolvedValue(session("researcher"));
    expect(locationOf(await proxy(req("/admin/rag")))).toBeNull();
    expect(pathOf(await proxy(req("/admin")))).toBe("/");
    expect(pathOf(await proxy(req("/admin/users")))).toBe("/");
    expect(pathOf(await proxy(req("/metrics")))).toBe("/");
  });

  test("administrator can reach everything", async () => {
    mockSession.mockResolvedValue(session("administrator"));
    expect(locationOf(await proxy(req("/admin")))).toBeNull();
    expect(locationOf(await proxy(req("/admin/rag")))).toBeNull();
    expect(locationOf(await proxy(req("/admin/users")))).toBeNull();
    expect(locationOf(await proxy(req("/metrics")))).toBeNull();
  });
});
