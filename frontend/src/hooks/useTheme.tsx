"use client";

import React, { createContext, useContext, useEffect, useState } from "react";

type Theme = "light" | "dark" | "auto";

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  effectiveTheme: "light" | "dark"; // Computed: theme if not "auto", else system preference
}

export const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

/**
 * Get the system's preferred color scheme
 */
function getSystemTheme(): "light" | "dark" {
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

/**
 * Compute the effective theme (resolving "auto" to system preference)
 */
function getEffectiveTheme(theme: Theme): "light" | "dark" {
  if (theme === "auto") {
    return getSystemTheme();
  }
  return theme;
}

/**
 * Apply theme to the document
 */
function applyTheme(effectiveTheme: "light" | "dark") {
  const html = document.documentElement;
  html.classList.remove("light", "dark");
  html.classList.add(effectiveTheme);
  html.setAttribute("data-theme", effectiveTheme);
}

interface ThemeProviderProps {
  children: React.ReactNode;
  initialTheme?: Theme;
}

/**
 * ThemeProvider wraps the app and provides theme context
 * Supports light/dark/auto (system detection)
 * Persists theme preference to localStorage
 */
export function ThemeProvider({
  children,
  initialTheme = "auto",
}: ThemeProviderProps) {
  const [theme, setThemeState] = useState<Theme>(initialTheme);
  const [effectiveTheme, setEffectiveTheme] = useState<"light" | "dark">(
    getEffectiveTheme(initialTheme)
  );
  const [mounted, setMounted] = useState(false);

  // On mount, load saved theme from localStorage
  useEffect(() => {
    const savedTheme = localStorage.getItem("theme") as Theme | null;
    if (savedTheme && ["light", "dark", "auto"].includes(savedTheme)) {
      setThemeState(savedTheme);
      applyTheme(getEffectiveTheme(savedTheme));
    } else {
      applyTheme(getEffectiveTheme(initialTheme));
    }
    setMounted(true);
  }, [initialTheme]);

  // When theme changes, update effective theme and persist
  useEffect(() => {
    if (!mounted) return;
    const effective = getEffectiveTheme(theme);
    setEffectiveTheme(effective);
    applyTheme(effective);
    localStorage.setItem("theme", theme);
  }, [theme, mounted]);

  // Listen for system theme changes when using "auto"
  useEffect(() => {
    if (!mounted || theme !== "auto") return;

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = (e: MediaQueryListEvent) => {
      const newTheme = e.matches ? "dark" : "light";
      setEffectiveTheme(newTheme);
      applyTheme(newTheme);
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, [theme, mounted]);

  const setTheme = (newTheme: Theme) => {
    setThemeState(newTheme);
  };

  return (
    <ThemeContext.Provider value={{ theme, setTheme, effectiveTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

/**
 * Hook to use the theme context
 */
export function useTheme(): ThemeContextType {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
}
