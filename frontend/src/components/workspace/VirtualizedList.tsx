"use client";

import { useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

interface VirtualizedListProps<T> {
  items: T[];
  getKey: (item: T) => string;
  renderItem: (item: T) => React.ReactNode;
  /** First-paint size estimate per row before measurement (px). */
  estimateSize?: number;
  /** Vertical gap between rows (px), folded into each row's measured box. */
  gap?: number;
  /** Tailwind max-height class for the internal scroll container. */
  maxHeightClass?: string;
  /** Optional test id on the scroll container (used to assert the windowed path). */
  testId?: string;
}

/**
 * Windowed list for the workspace Documents tab. Rows have variable height
 * (cards expand to reveal an action bar), so we rely on dynamic measurement via
 * `measureElement` rather than a fixed `estimateSize`. The list gets its OWN
 * bounded scroll container (the panel column otherwise scrolls as one unit), so
 * only the visible rows are mounted regardless of document count.
 *
 * Used only past a threshold; small lists render plainly (see DocumentsTab) so
 * the common case keeps its exact prior DOM/behavior.
 */
export function VirtualizedList<T>({
  items,
  getKey,
  renderItem,
  estimateSize = 64,
  gap = 6,
  maxHeightClass = "max-h-[60vh]",
  testId,
}: VirtualizedListProps<T>) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => estimateSize + gap,
    overscan: 6,
  });

  return (
    <div ref={parentRef} data-testid={testId} className={`${maxHeightClass} overflow-y-auto`}>
      <div style={{ height: virtualizer.getTotalSize(), position: "relative", width: "100%" }}>
        {virtualizer.getVirtualItems().map((virtualRow) => (
          <div
            key={getKey(items[virtualRow.index])}
            data-index={virtualRow.index}
            ref={virtualizer.measureElement}
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              transform: `translateY(${virtualRow.start}px)`,
              paddingBottom: gap,
            }}
          >
            {renderItem(items[virtualRow.index])}
          </div>
        ))}
      </div>
    </div>
  );
}
