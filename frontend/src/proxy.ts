import { NextRequest, NextResponse } from "next/server";
import {
  getSessionCookieName,
  getSessionFromCookieValue,
  isAuthEnabled,
} from "@/lib/auth/session";

const ADMIN_PREFIXES = ["/admin", "/metrics"];

function isAdminRoute(pathname: string): boolean {
  return ADMIN_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(prefix + "/")
  );
}

function buildLoginRedirect(request: NextRequest): NextResponse {
  const loginUrl = new URL("/login", request.url);
  const nextPath = `${request.nextUrl.pathname}${request.nextUrl.search}`;
  if (nextPath !== "/") {
    loginUrl.searchParams.set("next", nextPath);
  }
  return NextResponse.redirect(loginUrl);
}

export async function proxy(request: NextRequest) {
  if (!isAuthEnabled()) {
    return NextResponse.next();
  }

  const pathname = request.nextUrl.pathname;
  const token = request.cookies.get(getSessionCookieName())?.value;
  const session = await getSessionFromCookieValue(token);

  if (pathname === "/login") {
    if (session) {
      return NextResponse.redirect(new URL("/", request.url));
    }
    return NextResponse.next();
  }

  if (!session) {
    return buildLoginRedirect(request);
  }

  if (isAdminRoute(pathname) && session.role !== "administrator") {
    return NextResponse.redirect(new URL("/", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|.*\\..*$).*)"],
};
