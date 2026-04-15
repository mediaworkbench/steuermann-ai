"use client";

import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { messages, type Locale, type Messages } from "@/i18n/messages";
import { useSettings } from "@/hooks/useSettings";
import { CURRENT_USER_ID } from "@/lib/runtime";

type MessageKey = string;

type I18nContextValue = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: MessageKey, vars?: Record<string, string | number>) => string;
  formatDate: (value: Date | string | number, options?: Intl.DateTimeFormatOptions) => string;
  formatTime: (value: Date | string | number, options?: Intl.DateTimeFormatOptions) => string;
  formatDateTime: (value: Date | string | number, options?: Intl.DateTimeFormatOptions) => string;
  formatNumber: (value: number, options?: Intl.NumberFormatOptions) => string;
  formatRelativeTime: (value: Date | string | number) => string;
};

const I18nContext = createContext<I18nContextValue | undefined>(undefined);

function isLocale(value: string): value is Locale {
  return value === "en" || value === "de";
}

function interpolate(template: string, vars?: Record<string, string | number>): string {
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (_, token: string) => {
    const replacement = vars[token];
    return replacement == null ? `{${token}}` : String(replacement);
  });
}

function getValueByPath(dict: Messages, path: string): unknown {
  return path.split(".").reduce<unknown>((acc, key) => {
    if (acc && typeof acc === "object" && key in acc) {
      return (acc as Record<string, unknown>)[key];
    }
    return undefined;
  }, dict);
}

function toDate(value: Date | string | number): Date {
  return value instanceof Date ? value : new Date(value);
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocale] = useState<Locale>("en");
  const { settings } = useSettings(CURRENT_USER_ID);

  useEffect(() => {
    const configuredLanguage = settings?.language?.toLowerCase();
    if (configuredLanguage && isLocale(configuredLanguage)) {
      setLocale(configuredLanguage);
      return;
    }
    const browser = typeof navigator !== "undefined" ? navigator.language.slice(0, 2).toLowerCase() : "en";
    if (isLocale(browser)) {
      setLocale(browser);
    }
  }, [settings?.language]);

  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.lang = locale;
    }
  }, [locale]);

  const value = useMemo<I18nContextValue>(() => {
    const dict = messages[locale];

    const t = (key: MessageKey, vars?: Record<string, string | number>): string => {
      const candidate = getValueByPath(dict, key);
      if (typeof candidate === "string") {
        return interpolate(candidate, vars);
      }
      return key;
    };

    const formatDate = (valueToFormat: Date | string | number, options?: Intl.DateTimeFormatOptions) => {
      return new Intl.DateTimeFormat(locale, options).format(toDate(valueToFormat));
    };

    const formatTime = (valueToFormat: Date | string | number, options?: Intl.DateTimeFormatOptions) => {
      return new Intl.DateTimeFormat(locale, {
        hour: "2-digit",
        minute: "2-digit",
        ...options,
      }).format(toDate(valueToFormat));
    };

    const formatDateTime = (valueToFormat: Date | string | number, options?: Intl.DateTimeFormatOptions) => {
      return new Intl.DateTimeFormat(locale, {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        ...options,
      }).format(toDate(valueToFormat));
    };

    const formatNumber = (valueToFormat: number, options?: Intl.NumberFormatOptions) => {
      return new Intl.NumberFormat(locale, options).format(valueToFormat);
    };

    const formatRelativeTime = (valueToFormat: Date | string | number) => {
      const target = toDate(valueToFormat).getTime();
      const now = Date.now();
      const diffSec = Math.round((target - now) / 1000);
      const absSec = Math.abs(diffSec);
      const rtf = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });

      if (absSec < 60) return rtf.format(diffSec, "second");
      const diffMin = Math.round(diffSec / 60);
      if (Math.abs(diffMin) < 60) return rtf.format(diffMin, "minute");
      const diffHr = Math.round(diffMin / 60);
      if (Math.abs(diffHr) < 24) return rtf.format(diffHr, "hour");
      const diffDay = Math.round(diffHr / 24);
      if (Math.abs(diffDay) < 7) return rtf.format(diffDay, "day");
      return formatDate(valueToFormat, { month: "short", day: "numeric" });
    };

    return {
      locale,
      setLocale,
      t,
      formatDate,
      formatTime,
      formatDateTime,
      formatNumber,
      formatRelativeTime,
    };
  }, [locale]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nContextValue {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error("useI18n must be used within an I18nProvider");
  }
  return context;
}
