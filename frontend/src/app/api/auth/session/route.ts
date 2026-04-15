import { NextRequest, NextResponse } from "next/server";
import {
  getSessionCookieName,
  getSessionFromCookieValue,
  isAuthEnabled,
} from "@/lib/auth/session";

export async function GET(request: NextRequest) {
  if (!isAuthEnabled()) {
    return NextResponse.json({ enabled: false, authenticated: false, user: null });
  }

  const token = request.cookies.get(getSessionCookieName())?.value;
  const session = await getSessionFromCookieValue(token);

  return NextResponse.json({
    enabled: true,
    authenticated: session != null,
    user: session,
  });
}