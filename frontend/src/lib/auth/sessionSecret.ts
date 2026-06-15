/** Policy for the JWT signing secret. Kept jose-free so it is unit-testable. */
export const MIN_SESSION_SECRET_LENGTH = 32;

const WEAK_SESSION_SECRETS = new Set(["change-me", "changeme", "secret", "your-secret"]);

/** A secret is weak if it's a known default or shorter than the minimum length. */
export function isWeakSessionSecret(secret: string): boolean {
  return WEAK_SESSION_SECRETS.has(secret.toLowerCase()) || secret.length < MIN_SESSION_SECRET_LENGTH;
}
