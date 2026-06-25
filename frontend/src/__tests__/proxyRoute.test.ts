/** @jest-environment node */
import { NextRequest } from "next/server";

// The backend trusts x-authenticated-* + x-chat-token headers, so the proxy MUST set them
// itself and never forward client-supplied copies. These tests assert that boundary.

jest.mock("@/lib/auth/session", () => ({
  getSessionCookieName: () => "uaf_session",
  isAuthEnabled: jest.fn(() => true),
  getSessionFromCookieValue: jest.fn(async () => ({
    userId: "real-user",
    username: "real",
    displayName: "real",
    email: "",
    role: "user",
    mustChangePassword: false,
    tokenVersion: 7,
  })),
}));

function ctx(path: string[]) {
  return { params: Promise.resolve({ path }) };
}

async function loadRoute() {
  process.env.CHAT_ACCESS_TOKEN = "proxy-secret";
  process.env.FASTAPI_URL = "http://fastapi:8001";
  return import("@/app/api/proxy/[...path]/route");
}

function capturedHeaders(): Headers {
  const call = (global.fetch as jest.Mock).mock.calls[0];
  return (call[1] as RequestInit).headers as Headers;
}

beforeEach(() => {
  global.fetch = jest.fn(async () =>
    new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "content-type": "application/json" },
    }),
  ) as jest.Mock;
});

afterEach(() => {
  delete (global as Record<string, unknown>).fetch;
});

test("strips spoofed x-authenticated-* and sets identity from the session", async () => {
  const { GET } = await loadRoute();
  const request = new NextRequest("http://localhost:3000/api/proxy/api/conversations", {
    headers: {
      "x-authenticated-role": "administrator", // spoof attempt
      "x-authenticated-user-id": "attacker", // spoof attempt
      "x-authenticated-username": "attacker",
      "x-authenticated-token-version": "999", // spoof attempt
    },
  });

  await GET(request, ctx(["api", "conversations"]));

  const headers = capturedHeaders();
  expect(headers.get("x-authenticated-role")).toBe("user");
  expect(headers.get("x-authenticated-user-id")).toBe("real-user");
  expect(headers.get("x-authenticated-username")).toBe("real");
  // The spoofed version is dropped; the proxy sets it from the trusted session instead.
  expect(headers.get("x-authenticated-token-version")).toBe("7");
});

test("ignores a client-supplied x-chat-token, using the env value", async () => {
  const { GET } = await loadRoute();
  const request = new NextRequest("http://localhost:3000/api/proxy/api/conversations", {
    headers: { "x-chat-token": "forged-token" },
  });

  await GET(request, ctx(["api", "conversations"]));

  expect(capturedHeaders().get("x-chat-token")).toBe("proxy-secret");
});
