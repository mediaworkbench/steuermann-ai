import { NextRequest, NextResponse } from "next/server";
import {
  createSessionToken,
  getSessionCookieName,
  getSessionCookieOptions,
  getSessionFromCookieValue,
  isAuthEnabled,
} from "@/lib/auth/session";

export const runtime = "nodejs";

const FASTAPI_URL = process.env.FASTAPI_URL || "http://fastapi:8001";
const CHAT_ACCESS_TOKEN = process.env.CHAT_ACCESS_TOKEN || "";

export async function POST(request: NextRequest) {
  if (!isAuthEnabled()) {
    return NextResponse.json({ detail: "Authentication is disabled" }, { status: 400 });
  }

  const session = await getSessionFromCookieValue(
    request.cookies.get(getSessionCookieName())?.value
  );
  if (!session) {
    return NextResponse.json({ detail: "Authentication required" }, { status: 401 });
  }

  const body = (await request.json().catch(() => null)) as
    | { current_password?: string; new_password?: string }
    | null;
  if (!body?.current_password || !body?.new_password) {
    return NextResponse.json(
      { detail: "Current and new password are required" },
      { status: 400 }
    );
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "x-authenticated-user-id": session.userId,
    "x-authenticated-username": session.username,
    "x-authenticated-role": session.role,
  };
  if (CHAT_ACCESS_TOKEN.trim()) {
    headers["x-chat-token"] = CHAT_ACCESS_TOKEN.trim();
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${FASTAPI_URL}/api/auth/change-password`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        current_password: body.current_password,
        new_password: body.new_password,
      }),
      cache: "no-store",
    });
  } catch {
    return NextResponse.json({ detail: "Authentication service unavailable" }, { status: 503 });
  }

  if (!upstream.ok) {
    const detail = await upstream.json().catch(() => ({ detail: "Password change failed" }));
    return NextResponse.json(detail, { status: upstream.status });
  }

  // Re-mint the session cookie so the must-change flag is cleared immediately.
  const token = await createSessionToken({ ...session, mustChangePassword: false });
  const response = NextResponse.json({ ok: true });
  response.cookies.set(getSessionCookieName(), token, getSessionCookieOptions());
  return response;
}
