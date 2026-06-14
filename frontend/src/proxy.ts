import { NextRequest, NextResponse } from "next/server";
import {
  getSessionCookieName,
  getSessionFromCookieValue,
  isAuthEnabled,
} from "@/lib/auth/session";

const CHANGE_PASSWORD_PATH = "/change-password";

function isPath(pathname: string, prefix: string): boolean {
  return pathname === prefix || pathname.startsWith(prefix + "/");
}

// RAG explorer is open to researchers + administrators; everything else under /admin
// and all of /metrics is administrator-only.
function isRagRoute(pathname: string): boolean {
  return isPath(pathname, "/admin/rag");
}

function isAdminOnlyRoute(pathname: string): boolean {
  return isPath(pathname, "/admin") || isPath(pathname, "/metrics");
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
      const dest = session.mustChangePassword ? CHANGE_PASSWORD_PATH : "/";
      return NextResponse.redirect(new URL(dest, request.url));
    }
    return NextResponse.next();
  }

  if (!session) {
    return buildLoginRedirect(request);
  }

  // Force a password change before any other page is reachable.
  if (session.mustChangePassword && pathname !== CHANGE_PASSWORD_PATH) {
    return NextResponse.redirect(new URL(CHANGE_PASSWORD_PATH, request.url));
  }

  const canAccessRag = session.role === "researcher" || session.role === "administrator";
  if (isRagRoute(pathname)) {
    if (!canAccessRag) {
      return NextResponse.redirect(new URL("/", request.url));
    }
  } else if (isAdminOnlyRoute(pathname) && session.role !== "administrator") {
    return NextResponse.redirect(new URL("/", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|.*\\..*$).*)"],
};
