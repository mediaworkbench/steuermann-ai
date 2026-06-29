import { NextRequest, NextResponse } from "next/server";
import {
  getSessionCookieName,
  getSessionCookieOptions,
  getSessionFromCookieValue,
  isAuthEnabled,
} from "@/lib/auth/session";

export const runtime = "nodejs";

const FASTAPI_URL = process.env.FASTAPI_URL || "http://fastapi:8001";
const CHAT_ACCESS_TOKEN = process.env.CHAT_ACCESS_TOKEN || "";

export async function POST(request: NextRequest) {
  // Best-effort server-side revocation: bump the user's token_version so the cookie we're
  // about to clear can't be replayed. The cookie is cleared regardless of the outcome.
  if (isAuthEnabled()) {
    const session = await getSessionFromCookieValue(
      request.cookies.get(getSessionCookieName())?.value
    );
    if (session) {
      try {
        const headers: Record<string, string> = {
          "x-authenticated-user-id": session.userId,
          "x-authenticated-username": session.username,
          "x-authenticated-role": session.role,
          "x-authenticated-token-version": String(session.tokenVersion ?? 0),
        };
        if (CHAT_ACCESS_TOKEN.trim()) {
          headers["x-chat-token"] = CHAT_ACCESS_TOKEN.trim();
        }
        await fetch(`${FASTAPI_URL}/api/auth/logout`, {
          method: "POST",
          headers,
          cache: "no-store",
        });
      } catch {
        // Backend unreachable — clearing the cookie below still logs the user out locally.
      }
    }
  }

  const response = NextResponse.json({ authenticated: false });
  response.cookies.set(getSessionCookieName(), "", getSessionCookieOptions(0));
  return response;
}
