import { NextRequest, NextResponse } from "next/server";
import { verifyPassword } from "@/lib/auth/password";
import {
  createSessionToken,
  getSessionCookieName,
  getSessionCookieOptions,
  isAuthEnabled,
} from "@/lib/auth/session";

export const runtime = "nodejs";

function getConfiguredUsername(): string {
  return (process.env.AUTH_USERNAME || "admin").trim();
}

function getConfiguredPasswordHash(): string {
  return (process.env.AUTH_PASSWORD_HASH || "").trim();
}

export async function POST(request: NextRequest) {
  if (!isAuthEnabled()) {
    return NextResponse.json({ detail: "Authentication is disabled" }, { status: 400 });
  }

  const body = await request.json().catch(() => null) as
    | { username?: string; password?: string }
    | null;

  const username = body?.username?.trim() || "";
  const password = body?.password || "";

  if (!username || !password) {
    return NextResponse.json({ detail: "Username and password are required" }, { status: 400 });
  }

  const passwordHash = getConfiguredPasswordHash();
  const sessionSecret = (process.env.AUTH_SESSION_SECRET || "").trim();
  if (!passwordHash || !sessionSecret) {
    return NextResponse.json(
      { detail: "Authentication is enabled but not fully configured" },
      { status: 500 }
    );
  }

  const validUsername = username === getConfiguredUsername();
  const validPassword = verifyPassword(password, passwordHash);

  if (!validUsername || !validPassword) {
    return NextResponse.json({ detail: "Invalid username or password" }, { status: 401 });
  }

  const token = await createSessionToken({
    userId: (process.env.AUTH_USERNAME || "anonymous").trim() || "anonymous",
    username,
    displayName: process.env.NEXT_PUBLIC_SINGLE_USER_DISPLAY_NAME || username,
    email: process.env.NEXT_PUBLIC_SINGLE_USER_EMAIL || "",
  });

  const response = NextResponse.json({ authenticated: true });
  response.cookies.set(getSessionCookieName(), token, getSessionCookieOptions());
  return response;
}