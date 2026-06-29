import { NextRequest, NextResponse } from "next/server";
import {
  createSessionToken,
  getSessionCookieName,
  getSessionCookieOptions,
  isAuthEnabled,
} from "@/lib/auth/session";

export const runtime = "nodejs";

const FASTAPI_URL = process.env.FASTAPI_URL || "http://fastapi:8001";
const CHAT_ACCESS_TOKEN = process.env.CHAT_ACCESS_TOKEN || "";

export async function POST(request: NextRequest) {
  if (!isAuthEnabled()) {
    return NextResponse.json({ detail: "Authentication is disabled" }, { status: 400 });
  }

  const body = (await request.json().catch(() => null)) as
    | { username?: string; password?: string }
    | null;

  const username = body?.username?.trim() || "";
  const password = body?.password || "";

  if (!username || !password) {
    return NextResponse.json({ detail: "Username and password are required" }, { status: 400 });
  }

  const sessionSecret = (process.env.AUTH_SESSION_SECRET || "").trim();
  if (!sessionSecret) {
    return NextResponse.json(
      { detail: "Authentication is enabled but not fully configured" },
      { status: 500 }
    );
  }

  // Verify credentials against the backend (argon2id, DB-backed users).
  let upstream: Response;
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (CHAT_ACCESS_TOKEN.trim()) {
      headers["x-chat-token"] = CHAT_ACCESS_TOKEN.trim();
    }
    upstream = await fetch(`${FASTAPI_URL}/api/auth/login`, {
      method: "POST",
      headers,
      body: JSON.stringify({ username, password }),
      cache: "no-store",
    });
  } catch {
    return NextResponse.json({ detail: "Authentication service unavailable" }, { status: 503 });
  }

  if (!upstream.ok) {
    // Surface 401/403 as-is; collapse anything else to 401 to avoid leaking detail.
    const status = upstream.status === 403 ? 403 : 401;
    const detail =
      status === 403 ? "Account is not active" : "Invalid username or password";
    return NextResponse.json({ detail }, { status });
  }

  const data = (await upstream.json()) as {
    user_id: string;
    username: string;
    email: string;
    role: string;
    must_change_password: boolean;
    token_version?: number;
  };

  const token = await createSessionToken({
    userId: data.user_id,
    username: data.username,
    displayName: data.username,
    email: data.email || "",
    role: data.role === "administrator" ? "administrator" : data.role === "researcher" ? "researcher" : "user",
    mustChangePassword: Boolean(data.must_change_password),
    tokenVersion: typeof data.token_version === "number" ? data.token_version : 0,
  });

  const response = NextResponse.json({
    authenticated: true,
    mustChangePassword: Boolean(data.must_change_password),
  });
  response.cookies.set(getSessionCookieName(), token, getSessionCookieOptions());
  return response;
}
