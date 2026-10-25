/**
 * Post-login redirect target: only same-origin relative paths are allowed.
 * Browsers normalise `\` to `/` and strip tabs/newlines while parsing URLs, so
 * `/\evil.com` or `/\t/evil.com` would otherwise become protocol-relative and
 * leave the site (open redirect).
 */
export function sanitizeNextPath(path?: string): string {
  if (!path || !path.startsWith("/") || path.startsWith("//")) {
    return "/";
  }
  if (/[\\\u0000-\u001f\u007f\s]/.test(path)) {
    return "/";
  }
  try {
    const base = "http://localhost";
    const resolved = new URL(path, base);
    if (resolved.origin !== base) return "/";
    return `${resolved.pathname}${resolved.search}${resolved.hash}`;
  } catch {
    return "/";
  }
}
