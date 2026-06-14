/**
 * Full-document navigation. Unlike `router.push/replace` (client-side soft nav), this
 * triggers a real document request so the Next.js middleware re-runs server-side with the
 * current session cookie — used after login / password change so auth gates are enforced.
 */
export function hardNavigate(url: string): void {
  window.location.assign(url);
}
