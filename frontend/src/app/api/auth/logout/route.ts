import { NextResponse } from "next/server";
import { getSessionCookieName, getSessionCookieOptions } from "@/lib/auth/session";

export async function POST() {
  const response = NextResponse.json({ authenticated: false });
  response.cookies.set(getSessionCookieName(), "", getSessionCookieOptions(0));
  return response;
}