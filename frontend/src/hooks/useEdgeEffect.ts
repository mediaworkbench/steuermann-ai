"use client";

import { useEffect, useLayoutEffect, useRef } from "react";

interface EdgeCallbacks {
  onRising?: () => void;
  onFalling?: () => void;
}

/**
 * Fires onRising on the leading edge (false→true) and onFalling on the
 * trailing edge (true→false) of a boolean flag.
 *
 * Callbacks are stored in a ref updated via useLayoutEffect so the effect
 * dependency array stays as [flag] — callers do not need useCallback, and
 * there is no risk of stale closures or infinite re-render loops from
 * unstable callback references.
 */
export function useEdgeEffect(flag: boolean, callbacks: EdgeCallbacks): void {
  const cbRef = useRef<EdgeCallbacks>(callbacks);
  useLayoutEffect(() => {
    cbRef.current = callbacks;
  });

  const prevRef = useRef(flag);
  useEffect(() => {
    const prev = prevRef.current;
    prevRef.current = flag;
    if (!prev && flag) cbRef.current.onRising?.();
    if (prev && !flag) cbRef.current.onFalling?.();
  }, [flag]);
}
