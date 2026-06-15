import { NextRequest, NextResponse } from "next/server";
import {
  getSessionCookieName,
  getSessionCookieOptions,
  getSessionFromCookieValue,
  isAuthEnabled,
  type UserRole,
} from "@/lib/auth/session";

export const runtime = "nodejs";

const FASTAPI_URL = process.env.FASTAPI_URL || "http://fastapi:8001";
const CHAT_ACCESS_TOKEN = process.env.CHAT_ACCESS_TOKEN || "";

function devRole(): UserRole {
  const raw = (process.env.NEXT_PUBLIC_AUTH_USER_ROLE || "administrator").trim().toLowerCase();
  return raw === "researcher" ? "researcher" : raw === "user" ? "user" : "administrator";
}

function normalizeRole(value: unknown): UserRole {
  return value === "administrator" ? "administrator" : value === "researcher" ? "researcher" : "user";
}

export async function GET(request: NextRequest) {
  // Dev bypass: identity comes from server env so it matches the backend's dev-bypass user
  // (resolve_current_user → AUTH_USERNAME). This keeps one identity source across both modes.
  if (!isAuthEnabled()) {
    const userId = (process.env.AUTH_USERNAME || "anonymous").trim() || "anonymous";
    const role = devRole();
    const user = {
      userId,
      username: userId,
      displayName: process.env.NEXT_PUBLIC_SINGLE_USER_DISPLAY_NAME || userId,
      email: process.env.NEXT_PUBLIC_SINGLE_USER_EMAIL || "",
      role,
      mustChangePassword: false,
    };
    return NextResponse.json({ enabled: false, authenticated: true, user, role });
  }

  const session = await getSessionFromCookieValue(request.cookies.get(getSessionCookieName())?.value);
  if (!session) {
    return NextResponse.json({ enabled: true, authenticated: false, user: null, role: "user" });
  }

  // Re-validate against the backend so role/status/must-change are DB-fresh rather than the
  // (up to 7-day) JWT claims: a demotion hides admin UI promptly, and a suspended/deleted
  // account is logged out (cookie cleared).
  try {
    const headers: Record<string, string> = {
      "x-authenticated-user-id": session.userId,
      "x-authenticated-username": session.username,
      "x-authenticated-role": session.role,
    };
    if (CHAT_ACCESS_TOKEN.trim()) {
      headers["x-chat-token"] = CHAT_ACCESS_TOKEN.trim();
    }
    const meRes = await fetch(`${FASTAPI_URL}/api/auth/me`, { headers, cache: "no-store" });

    if (meRes.ok) {
      const me = (await meRes.json()) as {
        user_id: string;
        username: string;
        email: string;
        role: string;
        must_change_password: boolean;
      };
      const user = {
        userId: me.user_id,
        username: me.username,
        displayName: session.displayName,
        email: me.email || "",
        role: normalizeRole(me.role),
        mustChangePassword: Boolean(me.must_change_password),
      };
      return NextResponse.json({ enabled: true, authenticated: true, user, role: user.role });
    }

    if (meRes.status === 401 || meRes.status === 403) {
      // Account no longer valid — clear the cookie so the user is logged out.
      const res = NextResponse.json({ enabled: true, authenticated: false, user: null, role: "user" });
      res.cookies.set(getSessionCookieName(), "", getSessionCookieOptions(0));
      return res;
    }
  } catch {
    // Backend unreachable — fall back to the JWT so a transient outage doesn't log everyone out.
  }

  return NextResponse.json({ enabled: true, authenticated: true, user: session, role: session.role });
}
