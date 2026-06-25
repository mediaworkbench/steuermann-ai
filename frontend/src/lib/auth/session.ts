import { SignJWT } from "jose/jwt/sign";
import { jwtVerify } from "jose/jwt/verify";
import { MIN_SESSION_SECRET_LENGTH, isWeakSessionSecret } from "@/lib/auth/sessionSecret";

export type UserRole = "user" | "researcher" | "administrator";

const VALID_ROLES: readonly UserRole[] = ["user", "researcher", "administrator"];

function normalizeRole(value: unknown): UserRole {
  return VALID_ROLES.includes(value as UserRole) ? (value as UserRole) : "user";
}

export interface SessionUser {
  userId: string;
  username: string;
  displayName: string;
  email: string;
  role: UserRole;
  mustChangePassword: boolean;
  /** Per-user revocation counter; the JWT is rejected once the DB value moves past it. */
  tokenVersion: number;
}

const SESSION_COOKIE_NAME = "uaf_session";
const SESSION_ISSUER = "steuermann-ai";
const SESSION_AUDIENCE = "steuermann-ai-ui";
const SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7;

let _warnedWeakSecret = false;

function getSessionSecret(): Uint8Array {
  const secret = process.env.AUTH_SESSION_SECRET?.trim();
  if (!secret) {
    throw new Error("AUTH_SESSION_SECRET is required when authentication is enabled");
  }
  if (isWeakSessionSecret(secret)) {
    if (process.env.NODE_ENV === "production") {
      // Fail closed: a default/short secret means JWTs are trivially forgeable.
      throw new Error(
        "AUTH_SESSION_SECRET is weak or a known default — set a strong value " +
          `(≥ ${MIN_SESSION_SECRET_LENGTH} chars, e.g. \`python -c "import secrets; print(secrets.token_hex(32))"\`) in production.`
      );
    } else if (!_warnedWeakSecret) {
      _warnedWeakSecret = true;
      console.warn(
        `[auth] AUTH_SESSION_SECRET is weak/default — acceptable for local dev, but set a strong value (≥ ${MIN_SESSION_SECRET_LENGTH} chars) before deploying.`
      );
    }
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
  const epoch = (process.env.SESSION_EPOCH || "").trim();
  return new SignJWT({
    username: user.username,
    displayName: user.displayName,
    email: user.email,
    role: user.role,
    mustChangePassword: user.mustChangePassword,
    tv: user.tokenVersion,
    ...(epoch ? { se: epoch } : {}),
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

    // Deployment-scoped logout lever: when SESSION_EPOCH is set, a token minted under a
    // different (or no) epoch is rejected so a redeploy can force re-login. Unset → no-op.
    const epoch = (process.env.SESSION_EPOCH || "").trim();
    if (epoch && payload.se !== epoch) {
      return null;
    }

    const userId = typeof payload.sub === "string" ? payload.sub : "";
    const username = typeof payload.username === "string" ? payload.username : "";
    const displayName = typeof payload.displayName === "string" ? payload.displayName : username;
    const email = typeof payload.email === "string" ? payload.email : "";
    const role = normalizeRole(payload.role);
    const mustChangePassword = payload.mustChangePassword === true;
    // Default 0 so legacy cookies minted before this claim existed stay valid.
    const tokenVersion = typeof payload.tv === "number" ? payload.tv : 0;

    if (!userId || !username) {
      return null;
    }

    return {
      userId,
      username,
      displayName,
      email,
      role,
      mustChangePassword,
      tokenVersion,
    };
  } catch {
    return null;
  }
}