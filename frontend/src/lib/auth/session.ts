import { SignJWT } from "jose/jwt/sign";
import { jwtVerify } from "jose/jwt/verify";

export interface SessionUser {
  userId: string;
  username: string;
  displayName: string;
  email: string;
}

const SESSION_COOKIE_NAME = "uaf_session";
const SESSION_ISSUER = "steuermann-ai";
const SESSION_AUDIENCE = "steuermann-ai-ui";
const SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7;

function getSessionSecret(): Uint8Array {
  const secret = process.env.AUTH_SESSION_SECRET?.trim();
  if (!secret) {
    throw new Error("AUTH_SESSION_SECRET is required when authentication is enabled");
  }
  return new TextEncoder().encode(secret);
}

export function isAuthEnabled(): boolean {
  const value = (process.env.AUTH_ENABLED || "false").trim().toLowerCase();
  return value === "true" || value === "1" || value === "yes";
}

export function getSessionCookieName(): string {
  return SESSION_COOKIE_NAME;
}

export function getSessionCookieOptions(maxAge: number = SESSION_MAX_AGE_SECONDS) {
  return {
    httpOnly: true,
    sameSite: "lax" as const,
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge,
  };
}

export async function createSessionToken(user: SessionUser): Promise<string> {
  return new SignJWT({
    username: user.username,
    displayName: user.displayName,
    email: user.email,
  })
    .setProtectedHeader({ alg: "HS256" })
    .setSubject(user.userId)
    .setIssuedAt()
    .setIssuer(SESSION_ISSUER)
    .setAudience(SESSION_AUDIENCE)
    .setExpirationTime(`${SESSION_MAX_AGE_SECONDS}s`)
    .sign(getSessionSecret());
}

export async function getSessionFromCookieValue(token?: string): Promise<SessionUser | null> {
  if (!token) {
    return null;
  }

  try {
    const { payload } = await jwtVerify(token, getSessionSecret(), {
      issuer: SESSION_ISSUER,
      audience: SESSION_AUDIENCE,
    });

    const userId = typeof payload.sub === "string" ? payload.sub : "";
    const username = typeof payload.username === "string" ? payload.username : "";
    const displayName = typeof payload.displayName === "string" ? payload.displayName : username;
    const email = typeof payload.email === "string" ? payload.email : "";

    if (!userId || !username) {
      return null;
    }

    return {
      userId,
      username,
      displayName,
      email,
    };
  } catch {
    return null;
  }
}