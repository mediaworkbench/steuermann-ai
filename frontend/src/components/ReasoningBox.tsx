"use client";
import { useState } from "react";
import { Brain, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ReasoningBoxProps {
  content: string;
  isStreaming?: boolean;
}

export function ReasoningBox({ content, isStreaming = false }: ReasoningBoxProps) {
  const [expanded, setExpanded] = useState(false);

  if (!content && !isStreaming) return null;

  return (
    <div className="mb-3 w-full overflow-hidden rounded-xl border border-border bg-surface-muted">
      <Button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        variant="ghost"
        size="sm"
        className="w-full justify-start gap-2 px-3 py-2 text-xs text-muted-foreground hover:text-foreground"
        aria-expanded={expanded}
      >
        {isStreaming ? (
          <span className="h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-2 border-primary/40
                           border-t-primary" />
        ) : (
          <ChevronDown
            size={14}
            className={`reasoning-chevron ${expanded ? "open" : ""}`}
          />
        )}
        <Brain size={13} className="shrink-0 text-primary/80" />
        <span className="font-medium">
          {isStreaming ? "Reasoning…" : "Reasoning"}
        </span>
      </Button>
      <div className={`reasoning-body ${expanded ? "open" : ""}`}>
        <div>
          <div className="px-3 pb-3 text-xs text-muted-foreground font-mono leading-relaxed
                          whitespace-pre-wrap overflow-x-auto max-h-80 overflow-y-auto">
            {content}
          </div>
        </div>
      </div>
    </div>
  );
}
