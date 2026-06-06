import * as React from "react";
import { cn } from "@/lib/utils";

interface SegmentedTabItem<T extends string> {
  value: T;
  label: React.ReactNode;
}

interface SegmentedTabsProps<T extends string> {
  items: Array<SegmentedTabItem<T>>;
  value: T;
  onValueChange: (next: T) => void;
  ariaLabel: string;
  className?: string;
}

export function SegmentedTabs<T extends string>({
  items,
  value,
  onValueChange,
  ariaLabel,
  className,
}: SegmentedTabsProps<T>) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface-elevated p-1 shadow-xs",
        className,
      )}
    >
      {items.map((item) => {
        const active = item.value === value;
        return (
          <button
            key={item.value}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onValueChange(item.value)}
            className={cn(
              "cursor-pointer rounded-lg px-4 py-2 text-sm font-bold tracking-wide transition-colors",
              active
                ? "bg-primary/20 text-foreground"
                : "text-muted-foreground hover:bg-surface hover:text-foreground",
            )}
          >
            {item.label}
          </button>
        );
      })}
    </div>
  );
}
