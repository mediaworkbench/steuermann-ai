import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";
import {
  getSessionCookieName,
  getSessionFromCookieValue,
  isAuthEnabled,
} from "@/lib/auth/session";

const FASTAPI_URL = process.env.FASTAPI_URL || "http://fastapi:8001";
const CHAT_ACCESS_TOKEN = process.env.CHAT_ACCESS_TOKEN || "";

// Trust headers the backend honors — always set by the proxy, never trusted from the client.
const SPOOFABLE_TRUST_HEADERS = new Set([
  "x-chat-token",
  "x-authenticated-user-id",
  "x-authenticated-username",
  "x-authenticated-role",
  "x-authenticated-token-version",
]);

const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "host",
  "content-length",
]);

async function forward(request: NextRequest, path: string[]) {
  const authEnabled = isAuthEnabled();
  const session = authEnabled
    ? await getSessionFromCookieValue(request.cookies.get(getSessionCookieName())?.value)
    : null;

  if (authEnabled && !session) {
    return NextResponse.json({ detail: "Authentication required" }, { status: 401 });
  }

  const upstreamPath = path.join("/");
  const upstreamUrl = new URL(`${FASTAPI_URL}/${upstreamPath}`);
  request.nextUrl.searchParams.forEach((value, key) => {
    upstreamUrl.searchParams.append(key, value);
  });

  const headers = new Headers();
  request.headers.forEach((value, key) => {
    const lower = key.toLowerCase();
    // Strip hop-by-hop headers AND any client-supplied trust headers — these are set
    // exclusively by the proxy below so a client can never spoof identity/role.
    if (HOP_BY_HOP_HEADERS.has(lower) || SPOOFABLE_TRUST_HEADERS.has(lower)) {
      return;
    }
    headers.set(key, value);
  });

  if (CHAT_ACCESS_TOKEN.trim()) {
    headers.set("x-chat-token", CHAT_ACCESS_TOKEN.trim());
  }
  if (session) {
    headers.set("x-authenticated-user-id", session.userId);
    headers.set("x-authenticated-username", session.username);
    headers.set("x-authenticated-role", session.role);
    headers.set("x-authenticated-token-version", String(session.tokenVersion ?? 0));
  }

  let body: BodyInit | undefined;
  if (request.method !== "GET" && request.method !== "HEAD") {
    const contentType = request.headers.get("content-type") || "";
    if (contentType.includes("multipart/form-data")) {
      body = await request.formData();
      headers.delete("content-type");
    } else {
      body = await request.text();
    }
  }

  const upstream = await fetch(upstreamUrl, {
    method: request.method,
    headers,
    body,
    cache: "no-store",
    redirect: "manual",
  });

  const responseHeaders = new Headers();
  upstream.headers.forEach((value, key) => {
    if (!HOP_BY_HOP_HEADERS.has(key.toLowerCase())) {
      responseHeaders.set(key, value);
    }
  });

  // 204/304 responses must not include a body.
  if (upstream.status === 204 || upstream.status === 304 || request.method === "HEAD") {
    return new NextResponse(null, {
      status: upstream.status,
      headers: responseHeaders,
    });
  }

  const contentType = upstream.headers.get("content-type") ?? "";
  if (contentType.includes("text/event-stream")) {
    return new NextResponse(upstream.body, {
      status: upstream.status,
      headers: responseHeaders,
    });
  }

  const responseBody = await upstream.arrayBuffer();
  return new NextResponse(responseBody, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

type RouteContext = { params: Promise<{ path: string[] }> };

export async function GET(request: NextRequest, context: RouteContext) {
  const { path } = await context.params;
  return forward(request, path);
}

export async function POST(request: NextRequest, context: RouteContext) {
  const { path } = await context.params;
  return forward(request, path);
}

export async function PUT(request: NextRequest, context: RouteContext) {
  const { path } = await context.params;
  return forward(request, path);
}

export async function PATCH(request: NextRequest, context: RouteContext) {
  const { path } = await context.params;
  return forward(request, path);
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  const { path } = await context.params;
  return forward(request, path);
}
