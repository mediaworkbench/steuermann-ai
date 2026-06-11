"use client";

import { Minimize2 } from "lucide-react";
import { ContextRingIndicator } from "@/components/ContextRingIndicator";
import { Button } from "@/components/ui/button";

function CtxRow({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="tabular-nums text-foreground">{value}</span>
    </div>
  );
}

interface ContextWindowMenuProps {
  open: boolean;
  onToggle: () => void;
  onClose: () => void;
  contextTokens: number;
  maxContextTokens: number | null;
  userMessageCount: number;
  assistantMessageCount: number;
  isStreaming: boolean;
  isCompacting: boolean;
  activeId: string | null;
  onCompact: () => void;
}

export function ContextWindowMenu({
  open,
  onToggle,
  onClose,
  contextTokens,
  maxContextTokens,
  userMessageCount,
  assistantMessageCount,
  isStreaming,
  isCompacting,
  activeId,
  onCompact,
}: ContextWindowMenuProps) {
  const ctxPct = maxContextTokens
    ? Math.min(100, Math.round((contextTokens / maxContextTokens) * 100))
    : 0;
  const ctxBarColor =
    ctxPct >= 85 ? "bg-destructive"
    : ctxPct >= 60 ? "bg-warning"
    : "bg-primary/70";

  return (
    <div className="relative">
      <Button
        type="button"
        onClick={onToggle}
        variant="ghost"
        size="sm"
        className="rounded-lg p-1.5 transition-colors hover:bg-surface-muted"
        aria-label="Context window details"
      >
        <ContextRingIndicator
          contextTokens={contextTokens}
          maxContextTokens={maxContextTokens}
        />
      </Button>

      {open && (
        <>
          <div aria-hidden="true" className="fixed inset-0 z-10" onClick={onClose} />
          <div className="absolute bottom-full right-0 z-20 mb-2 min-w-60 rounded-xl border border-border bg-surface py-2 shadow-lg">
            <p className="px-3 pb-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Context Window
            </p>

            {/* Usage bar + numbers */}
            <div className="px-3 pb-2">
              <div className="mb-1 flex items-center justify-between text-xs text-foreground">
                <span>{contextTokens.toLocaleString()} tokens</span>
                {maxContextTokens && <span className="text-muted-foreground">{ctxPct}%</span>}
              </div>
              {maxContextTokens ? (
                <>
                  <div className="h-1 overflow-hidden rounded-full bg-surface-muted">
                    <div
                      className={`h-full rounded-full transition-all ${ctxBarColor}`}
                      style={{ width: `${ctxPct}%` }}
                    />
                  </div>
                  <p className="mt-1 text-[10px] text-muted-foreground">of {maxContextTokens.toLocaleString()} max</p>
                </>
              ) : (
                <p className="mt-1 text-[10px] text-muted-foreground">context window size unknown</p>
              )}
            </div>

            <div className="my-1 border-t border-border" />

            {/* Message counts */}
            <div className="px-3 py-1 space-y-0.5">
              <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Messages</p>
              <CtxRow label="User" value={userMessageCount} />
              <CtxRow label="Assistant" value={assistantMessageCount} />
            </div>

            <div className="my-1 border-t border-border" />

            {/* Compact button */}
            <div className="px-2 pt-1">
              <Button
                type="button"
                disabled={isStreaming || isCompacting || !activeId || contextTokens === 0}
                onClick={onCompact}
                variant="ghost"
                size="sm"
                className="w-full items-center gap-2 rounded-lg px-3 py-2 text-sm text-foreground transition-colors hover:bg-surface-muted disabled:cursor-not-allowed disabled:opacity-40"
              >
                <Minimize2 size={16} className="text-muted-foreground" />
                {isCompacting ? "Compacting…" : "Compact Context"}
              </Button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
