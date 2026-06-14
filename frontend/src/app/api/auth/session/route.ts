import { NextRequest, NextResponse } from "next/server";
import {
  getSessionCookieName,
  getSessionFromCookieValue,
  isAuthEnabled,
  type UserRole,
} from "@/lib/auth/session";

function devRole(): UserRole {
  const raw = (process.env.NEXT_PUBLIC_AUTH_USER_ROLE || "administrator").trim().toLowerCase();
  return raw === "researcher" ? "researcher" : raw === "user" ? "user" : "administrator";
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

  const token = request.cookies.get(getSessionCookieName())?.value;
  const session = await getSessionFromCookieValue(token);

  return NextResponse.json({
    enabled: true,
    authenticated: session != null,
    user: session,
    role: session?.role ?? "user",
  });
}
