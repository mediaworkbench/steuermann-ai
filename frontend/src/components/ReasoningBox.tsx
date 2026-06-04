"use client";
import { useState } from "react";
import { Icon } from "./Icon";

interface ReasoningBoxProps {
  content: string;
  isStreaming?: boolean;
}

export function ReasoningBox({ content, isStreaming = false }: ReasoningBoxProps) {
  const [expanded, setExpanded] = useState(false);

  if (!content && !isStreaming) return null;

  return (
    <div className="w-full mb-3 rounded-xl border border-pacific-blue/20 bg-light-cyan/10 overflow-hidden">
      <button
        onClick={() => setExpanded((e) => !e)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs
                   text-evergreen/55 hover:text-evergreen/80 transition-colors"
        aria-expanded={expanded}
      >
        {isStreaming ? (
          <span className="w-3.5 h-3.5 rounded-full border-2 border-pacific-blue/40
                           border-t-pacific-blue animate-spin shrink-0" />
        ) : (
          <Icon
            name="expand_more"
            size={14}
            className={`reasoning-chevron ${expanded ? "open" : ""}`}
          />
        )}
        <Icon name="psychology" size={13} className="text-pacific-blue/70 shrink-0" />
        <span className="font-medium">
          {isStreaming ? "Reasoning…" : "Reasoning"}
        </span>
      </button>
      <div className={`reasoning-body ${expanded ? "open" : ""}`}>
        <div>
          <div className="px-3 pb-3 text-xs text-evergreen/65 font-mono leading-relaxed
                          whitespace-pre-wrap overflow-x-auto max-h-80 overflow-y-auto">
            {content}
          </div>
        </div>
      </div>
    </div>
  );
}
