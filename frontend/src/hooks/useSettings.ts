"use client";

import { useCallback, useEffect, useState } from "react";
import { UserSettings, fetchUserSettings, updateUserSettings } from "@/lib/api";

interface UseSettingsReturn {
  settings: UserSettings | null;
  loading: boolean;
  error: string | null;
  saveSettings: (updates: Partial<Omit<UserSettings, "user_id" | "updated_at">>) => Promise<boolean>;
  refetch: () => Promise<void>;
}

export function useSettings(userId: string): UseSettingsReturn {
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchUserSettings(userId);
      if (data) {
        setSettings(data);
      } else {
        setError("Failed to load settings");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [userId]);

  const saveSettings = useCallback(
    async (updates: Partial<Omit<UserSettings, "user_id" | "updated_at">>): Promise<boolean> => {
      try {
        const updated = await updateUserSettings(userId, updates);
        if (updated) {
          setSettings(updated);
          return true;
        }
        setError("Failed to save settings");
        return false;
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(message);
        return false;
      }
    },
    [userId, settings]
  );

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { settings, loading, error, saveSettings, refetch };
}

