"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  fetchProviderHealth,
  triggerLLMReprobe,
  type ProviderHealthEndpoint,
  type ProviderHealthStatus,
} from "@/lib/api";

interface ProviderHealthContextValue {
  status: ProviderHealthStatus;
  providers: ProviderHealthEndpoint[];
  loading: boolean;
  lastCheckedAt: string | null;
  /** Immediate live re-check + a background full capability reprobe. */
  refresh: () => void;
}

const ProviderHealthContext = createContext<ProviderHealthContextValue | null>(null);

// Poll cadence: relaxed while healthy, tighter while degraded/offline so recovery
// (or a fresh outage) surfaces quickly. The browser is also nudged to re-check on
// tab focus and on the `online` event, so steady-state polling can stay relaxed.
const POLL_ONLINE_MS = 30_000;
const POLL_OFFLINE_MS = 10_000;

export function ProviderHealthProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<ProviderHealthStatus>("online");
  const [providers, setProviders] = useState<ProviderHealthEndpoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [lastCheckedAt, setLastCheckedAt] = useState<string | null>(null);

  // Guards: `inFlight` collapses overlapping polls; `pollToken` invalidates a
  // scheduled tick after an unmount or a manual refresh reschedules.
  const inFlightRef = useRef(false);
  const pollTokenRef = useRef(0);
  const prevStatusRef = useRef<ProviderHealthStatus>("online");
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Gate state updates so a fetch in flight at unmount can't setState on a dead tree.
  const mountedRef = useRef(true);

  // Holds the latest scheduler so callbacks (visibility/online listeners, refresh)
  // can reschedule without depending on a not-yet-defined function.
  const scheduleRef = useRef<(delayMs: number) => void>(() => {});

  const runCheck = useCallback(async (): Promise<ProviderHealthStatus> => {
    if (inFlightRef.current) return prevStatusRef.current;
    inFlightRef.current = true;
    setLoading(true);
    try {
      const health = await fetchProviderHealth();
      // A null response (network/proxy failure) is treated as offline.
      const next: ProviderHealthStatus = health?.status ?? "offline";
      if (!mountedRef.current) return next;
      setStatus(next);
      setProviders(health?.providers ?? []);
      setLastCheckedAt(health?.checked_at ?? new Date().toISOString());

      // On recovery, refresh capability/tool-calling caches once in the background.
      if (prevStatusRef.current === "offline" && next !== "offline") {
        void triggerLLMReprobe();
      }
      prevStatusRef.current = next;
      return next;
    } finally {
      inFlightRef.current = false;
      if (mountedRef.current) setLoading(false);
    }
  }, []);

  const schedule = useCallback(
    (delayMs: number) => {
      if (timerRef.current) clearTimeout(timerRef.current);
      const token = ++pollTokenRef.current;
      timerRef.current = setTimeout(async () => {
        if (token !== pollTokenRef.current) return;
        const next = await runCheck();
        if (token !== pollTokenRef.current) return;
        schedule(next === "online" ? POLL_ONLINE_MS : POLL_OFFLINE_MS);
      }, delayMs);
    },
    [runCheck],
  );

  useEffect(() => {
    scheduleRef.current = schedule;
  }, [schedule]);

  const refresh = useCallback(() => {
    // Manual retry: re-ping now, fire a full reprobe, and restart the cadence.
    void triggerLLMReprobe();
    void runCheck().then((next) => {
      scheduleRef.current(next === "online" ? POLL_ONLINE_MS : POLL_OFFLINE_MS);
    });
  }, [runCheck]);

  useEffect(() => {
    // Re-arm on (re)mount — under React StrictMode the effect runs setup→cleanup→setup
    // and the ref persists, so without this the second setup would stay "unmounted" and
    // silently drop every state update.
    mountedRef.current = true;

    // Kick an immediate check, then let it self-schedule.
    void runCheck().then((next) => {
      scheduleRef.current(next === "online" ? POLL_ONLINE_MS : POLL_OFFLINE_MS);
    });

    const onVisible = () => {
      if (document.visibilityState === "visible") scheduleRef.current(0);
    };
    const onOnline = () => scheduleRef.current(0);
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("online", onOnline);

    return () => {
      mountedRef.current = false; // stop any in-flight check from updating state
      if (timerRef.current) clearTimeout(timerRef.current); // cancel the pending tick
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("online", onOnline);
    };
    // Mount-only: runCheck/schedule are reached via refs to avoid re-running this effect.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const value: ProviderHealthContextValue = {
    status,
    providers,
    loading,
    lastCheckedAt,
    refresh,
  };

  return (
    <ProviderHealthContext.Provider value={value}>{children}</ProviderHealthContext.Provider>
  );
}

export function useProviderHealth(): ProviderHealthContextValue {
  const ctx = useContext(ProviderHealthContext);
  if (!ctx) {
    throw new Error("useProviderHealth must be used within a ProviderHealthProvider");
  }
  return ctx;
}
