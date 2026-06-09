"""Profile theme token contract.

Single source of truth for valid token keys per ui.yaml section.
Used by Pydantic validation and the CLI validate command.
"""

# Valid keys for the `theme.colors` section of ui.yaml.
# Profiles may override any subset; omitted keys inherit CSS defaults.
COLOR_TOKENS: set[str] = {
    # Core surfaces
    "background",
    "foreground",
    "surface",
    "surface-elevated",
    "surface-muted",
    "border",
    "border-strong",
    # Semantic roles
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
    # Primary variants
    "primary-dark",
    "primary-light",
    # Sidebar
    "sidebar-background",
    "sidebar-foreground",
    "sidebar-muted",
    "sidebar-border",
    "sidebar-hover",
    "sidebar-active",
    # Sidebar shadcn compat
    "sidebar",
    "sidebar-primary",
    "sidebar-primary-foreground",
    "sidebar-accent",
    "sidebar-accent-foreground",
    "sidebar-ring",
    # Chart grid/axis
    "chart-grid",
    "chart-axis",
    # Chart data colors
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
    # Chart shadcn compat
    "chart-1",
    "chart-2",
    "chart-3",
    "chart-4",
    "chart-5",
    # Map tokens
    "map-primary",
    "map-secondary",
    "map-distance-line",
    "map-success",
    "map-warning",
    "map-accent",
    # Login screen
    "login-dev-bg",
    "login-main-bg",
    "login-card-shadow",
    "login-panel-shadow",
    # Shadow tokens
    "shadow-xs",
    "shadow-sm",
    "shadow-md",
    "shadow-lg",
    # shadcn compat aliases
    "card",
    "card-foreground",
    "popover",
    "popover-foreground",
    "input",
    "ring",
    # Legacy compat aliases
    "sidebar-bg",
    "sidebar-text",
    "sidebar-text-muted",
    "bg-primary",
    "bg-secondary",
    "bg-card",
    "text-primary",
    "text-secondary",
    "text-muted",
    "border-color",
}

# Valid keys for the `theme.fonts` section of ui.yaml.
FONT_TOKENS: set[str] = {
    "font-sans",
    "font-display",
    "font-mono",
}

# Valid keys for the `theme.radius` section of ui.yaml.
# The base `radius` drives all computed variants (sm, md, lg, xl, 2xl, 3xl, 4xl).
RADIUS_TOKENS: set[str] = {
    "radius",
}

# All known token keys across all sections.
KNOWN_TOKENS: set[str] = COLOR_TOKENS | FONT_TOKENS | RADIUS_TOKENS

# Human-readable labels for each section.
SECTION_LABELS: dict[str, str] = {
    "colors": "theme.colors",
    "fonts": "theme.fonts",
    "radius": "theme.radius",
    "custom_css_vars": "theme.custom_css_vars",
}

# Valid key sets per section for validation.
SECTION_TOKENS: dict[str, set[str]] = {
    "colors": COLOR_TOKENS,
    "fonts": FONT_TOKENS,
    "radius": RADIUS_TOKENS,
    # custom_css_vars has no validation — it is the escape hatch
}


def check_unknown_tokens(
    section: str,
    keys: set[str],
) -> list[str]:
    """Return a list of unknown token keys for a given section.

    custom_css_vars is always allowed (escape hatch).
    Returns empty list if all keys are valid.
    """
    if section == "custom_css_vars":
        return []
    allowed = SECTION_TOKENS.get(section)
    if allowed is None:
        return [f"Unknown section '{section}'"]
    return [k for k in keys if k not in allowed]
