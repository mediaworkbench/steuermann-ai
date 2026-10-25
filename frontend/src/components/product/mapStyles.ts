export const OPENFREEMAP_STYLE = "https://tiles.openfreemap.org/styles/positron";
export const SHOW_PIN_THRESHOLD = 5; // skip marker for zoom <= this (continent/world view)

export const DISTANCE_LINE_SOURCE_ID = "distance-line";
export const DISTANCE_LINE_LAYER_ID = "distance-line";

export const MAP_COLOR_PRIMARY = "#2563eb";
export const MAP_COLOR_SECONDARY = "#dc2626";
export const MAP_COLOR_DISTANCE_LINE = "#6366f1";
export const MAP_COLOR_SUCCESS = "#16a34a";
export const MAP_COLOR_WARNING = "#d97706";
export const MAP_COLOR_ACCENT = "#7c3aed";

export const MAP_MULTI_MARKER_COLORS = [
  MAP_COLOR_PRIMARY,
  MAP_COLOR_SECONDARY,
  MAP_COLOR_SUCCESS,
  MAP_COLOR_WARNING,
  MAP_COLOR_ACCENT,
] as const;

const MAP_COLOR_VAR_KEYS = {
  primary: "--map-primary",
  secondary: "--map-secondary",
  distanceLine: "--map-distance-line",
  success: "--map-success",
  warning: "--map-warning",
  accent: "--map-accent",
} as const;

const MAP_FALLBACK_VAR_KEYS = {
  primary: "--chart-blue-strong",
  secondary: "--chart-red",
  distanceLine: "--chart-indigo",
  success: "--chart-green",
  warning: "--chart-amber",
  accent: "--chart-violet",
} as const;

// MapLibre's style parser only understands hex / rgb() / hsl() / named colors, but the
// theme tokens are authored in lab()/oklch(). Anything else is painted onto a 1×1 canvas
// so the browser converts it to sRGB; a value the canvas rejects keeps the fallback.
const MAPLIBRE_PARSEABLE_COLOR = /^(#|rgba?\(|hsla?\()/i;

export function toMapLibreColor(value: string, fallback: string): string {
  if (MAPLIBRE_PARSEABLE_COLOR.test(value)) return value;
  if (typeof document === "undefined") return fallback;
  const ctx = document.createElement("canvas").getContext("2d", { willReadFrequently: true });
  if (!ctx) return fallback;
  ctx.fillStyle = fallback;
  ctx.fillStyle = value;
  ctx.fillRect(0, 0, 1, 1);
  const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data;
  return `rgba(${r}, ${g}, ${b}, ${Number((a / 255).toFixed(3))})`;
}

function readCssColor(
  style: CSSStyleDeclaration,
  varName: string,
  fallbackVarName: string,
  fallback: string,
): string {
  const value =
    style.getPropertyValue(varName).trim() || style.getPropertyValue(fallbackVarName).trim();
  return value ? toMapLibreColor(value, fallback) : fallback;
}

export interface MapThemeColors {
  primary: string;
  secondary: string;
  distanceLine: string;
  success: string;
  warning: string;
  accent: string;
  multiMarkerColors: readonly string[];
}

export function resolveMapThemeColors(root: HTMLElement = document.documentElement): MapThemeColors {
  const style = getComputedStyle(root);
  const primary = readCssColor(style, MAP_COLOR_VAR_KEYS.primary, MAP_FALLBACK_VAR_KEYS.primary, MAP_COLOR_PRIMARY);
  const secondary = readCssColor(style, MAP_COLOR_VAR_KEYS.secondary, MAP_FALLBACK_VAR_KEYS.secondary, MAP_COLOR_SECONDARY);
  const distanceLine = readCssColor(style, MAP_COLOR_VAR_KEYS.distanceLine, MAP_FALLBACK_VAR_KEYS.distanceLine, MAP_COLOR_DISTANCE_LINE);
  const success = readCssColor(style, MAP_COLOR_VAR_KEYS.success, MAP_FALLBACK_VAR_KEYS.success, MAP_COLOR_SUCCESS);
  const warning = readCssColor(style, MAP_COLOR_VAR_KEYS.warning, MAP_FALLBACK_VAR_KEYS.warning, MAP_COLOR_WARNING);
  const accent = readCssColor(style, MAP_COLOR_VAR_KEYS.accent, MAP_FALLBACK_VAR_KEYS.accent, MAP_COLOR_ACCENT);

  return {
    primary,
    secondary,
    distanceLine,
    success,
    warning,
    accent,
    multiMarkerColors: [primary, secondary, success, warning, accent],
  };
}
