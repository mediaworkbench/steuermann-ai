"use client";

import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { fetchSystemConfig } from "@/lib/api";
import {
  SINGLE_USER_APP_NAME,
  SINGLE_USER_DISPLAY_NAME,
  SINGLE_USER_ROLE_LABEL,
} from "@/lib/runtime";

interface ProfileTheme {
  colors: Record<string, string>;
  fonts: Record<string, string>;
  radius: Record<string, string>;
  custom_css_vars: Record<string, string>;
}

interface ProfileContextValue {
  id: string;
  displayName: string;
  roleLabel: string;
  appName: string;
  description: string;
  theme: ProfileTheme;
  loading: boolean;
}

const EMPTY_THEME: ProfileTheme = {
  colors: {},
  fonts: {},
  radius: {},
  custom_css_vars: {},
};

const ProfileContext = createContext<ProfileContextValue | undefined>(undefined);

function toCssVarName(key: string): string {
  const normalized = key.trim().replace(/_/g, "-").replace(/\s+/g, "-").toLowerCase();
  return normalized.startsWith("--") ? normalized : `--${normalized}`;
}

function applyThemeTokens(profileId: string, theme: ProfileTheme): void {
  const html = document.documentElement;
  html.setAttribute("data-profile", profileId);

  const tokenEntries: Array<[string, string]> = [
    ...Object.entries(theme.colors),
    ...Object.entries(theme.fonts),
    ...Object.entries(theme.radius),
    ...Object.entries(theme.custom_css_vars),
  ];

  for (const [key, value] of tokenEntries) {
    if (!value) {
      continue;
    }
    html.style.setProperty(toCssVarName(key), value);
  }
}

export function ProfileProvider({ children }: { children: React.ReactNode }) {
  const [value, setValue] = useState<ProfileContextValue>({
    id: "base",
    displayName: SINGLE_USER_DISPLAY_NAME,
    roleLabel: SINGLE_USER_ROLE_LABEL,
    appName: SINGLE_USER_APP_NAME,
    description: "",
    theme: EMPTY_THEME,
    loading: true,
  });

  useEffect(() => {
    let cancelled = false;

    async function loadProfile() {
      const config = await fetchSystemConfig();
      if (cancelled) {
        return;
      }

      const profile = config?.profile;
      if (!profile) {
        setValue((prev) => ({ ...prev, loading: false }));
        applyThemeTokens("base", EMPTY_THEME);
        return;
      }

      const next: ProfileContextValue = {
        id: profile.id || "base",
        displayName: profile.display_name || profile.app_name || SINGLE_USER_APP_NAME,
        roleLabel: profile.role_label || SINGLE_USER_ROLE_LABEL,
        appName: profile.app_name || profile.display_name || SINGLE_USER_APP_NAME,
        description: profile.description || "",
        theme: profile.theme || EMPTY_THEME,
        loading: false,
      };

      setValue(next);
      applyThemeTokens(next.id, next.theme);
    }

    void loadProfile();

    return () => {
      cancelled = true;
    };
  }, []);

  const contextValue = useMemo(() => value, [value]);

  return <ProfileContext.Provider value={contextValue}>{children}</ProfileContext.Provider>;
}

export function useProfile(): ProfileContextValue {
  const context = useContext(ProfileContext);
  if (!context) {
    throw new Error("useProfile must be used within a ProfileProvider");
  }
  return context;
}
