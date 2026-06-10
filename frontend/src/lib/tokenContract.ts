/** Profile theme token contract.
 *
 * Single source of truth for valid token keys per ui.yaml section.
 * Used by applyThemeTokens to warn on unknown keys.
 */

export const COLOR_TOKENS: ReadonlySet<string> = new Set([
  "background",
  "foreground",
  "surface",
  "surface-elevated",
  "surface-muted",
  "border",
  "border-strong",
  "primary",
  "primary-foreground",
  "secondary",
  "secondary-foreground",
  "accent",
  "accent-foreground",
  "muted",
  "muted-foreground",
  "destructive",
  "destructive-foreground",
  "success",
  "success-foreground",
  "warning",
  "warning-foreground",
  "info",
  "info-foreground",
  "focus-ring",
  "primary-dark",
  "sidebar-background",
  "sidebar-foreground",
  "sidebar-muted",
  "sidebar-border",
  "sidebar-hover",
  "sidebar-active",
  "sidebar",
  "sidebar-primary",
  "sidebar-primary-foreground",
  "sidebar-accent",
  "sidebar-accent-foreground",
  "sidebar-ring",
  "chart-grid",
  "chart-axis",
  "chart-blue",
  "chart-blue-strong",
  "chart-blue-soft",
  "chart-green",
  "chart-red",
  "chart-amber",
  "chart-amber-soft",
  "chart-indigo",
  "chart-violet",
  "chart-teal",
  "chart-teal-soft",
  "chart-orange",
  "chart-1",
  "chart-2",
  "chart-3",
  "chart-4",
  "chart-5",
  "login-dev-bg",
  "login-main-bg",
  "login-card-shadow",
  "login-panel-shadow",
  "card",
  "card-foreground",
  "popover",
  "popover-foreground",
  "input",
  "ring",
  "sidebar-bg",
  "sidebar-text",
  "sidebar-text-muted",
  "bg-primary",
  "bg-secondary",
  "bg-card",
  "text-primary",
  "text-secondary",
  "border-color",
]);

export const FONT_TOKENS: ReadonlySet<string> = new Set([
  "font-sans",
  "font-display",
  "font-mono",
]);

export const RADIUS_TOKENS: ReadonlySet<string> = new Set(["radius"]);

export const KNOWN_TOKENS: ReadonlySet<string> = new Set([
  ...COLOR_TOKENS,
  ...FONT_TOKENS,
  ...RADIUS_TOKENS,
]);

export function findUnknownTokens(
  section: "colors" | "fonts" | "radius",
  keys: string[],
): string[] {
  const allowed = section === "colors"
    ? COLOR_TOKENS
    : section === "fonts"
    ? FONT_TOKENS
    : RADIUS_TOKENS;
  return keys.filter((k) => !allowed.has(k));
}
