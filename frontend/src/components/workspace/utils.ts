export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
}

/** Best-effort MIME type from a workspace document filename extension. */
export function mimeTypeForFilename(name: string): string {
  if (name.endsWith(".md") || name.endsWith(".markdown")) return "text/markdown";
  if (name.endsWith(".json")) return "application/json";
  if (name.endsWith(".yaml") || name.endsWith(".yml")) return "application/yaml";
  if (name.endsWith(".csv")) return "text/csv";
  if (name.endsWith(".html")) return "text/html";
  if (name.endsWith(".xml")) return "text/xml";
  return "text/plain";
}

/**
 * Headers for the proxied workspace API, merged with any extras. Auth is added
 * server-side by the proxy (session cookie → identity headers + shared token);
 * the client must never carry the shared token itself.
 */
export function workspaceAuthHeaders(extra?: Record<string, string>): Record<string, string> {
  return { ...(extra ?? {}) };
}
