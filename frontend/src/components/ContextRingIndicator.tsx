"use client";

interface ContextRingIndicatorProps {
  contextTokens: number;
  maxContextTokens: number | null | undefined;
}

const SIZE = 22;
const RADIUS = 9;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

function formatCompact(n: number): string {
  if (n >= 1000) {
    const k = n / 1000;
    return `${k >= 100 ? Math.round(k) : k.toFixed(1).replace(/\.0$/, "")}k`;
  }
  return String(n);
}

export function ContextRingIndicator({ contextTokens, maxContextTokens }: ContextRingIndicatorProps) {
  // No known context window → show a raw token count instead of a misleading percentage.
  if (!maxContextTokens) {
    const tooltip = `Context: ${contextTokens.toLocaleString()} tokens (window size unknown)`;
    return (
      <div className="flex items-center gap-1 text-muted-foreground" title={tooltip} aria-label={tooltip}>
        <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} aria-hidden="true">
          <circle
            cx={SIZE / 2} cy={SIZE / 2} r={RADIUS}
            fill="none" stroke="currentColor" strokeWidth="2.5" strokeOpacity="0.25"
          />
        </svg>
        <span className="text-[10px] font-medium tabular-nums leading-none select-none">
          {formatCompact(contextTokens)}
        </span>
      </div>
    );
  }

  const pct = Math.min(1, contextTokens / maxContextTokens);
  const displayPct = Math.round(pct * 100);
  const filled = pct * CIRCUMFERENCE;

  const colorClass =
    contextTokens === 0 ? "text-muted-foreground"
    : pct < 0.6        ? "text-foreground"
    : pct < 0.85       ? "text-warning"
    :                    "text-destructive";

  const tooltip =
    contextTokens === 0
      ? "Context window: no tokens used yet"
      : `Context window: ${displayPct}% used (${contextTokens.toLocaleString()} / ${maxContextTokens.toLocaleString()} tokens)`;

  return (
    <div className={`flex items-center gap-1 ${colorClass}`} title={tooltip} aria-label={tooltip}>
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} aria-hidden="true">
        <circle
          cx={SIZE / 2} cy={SIZE / 2} r={RADIUS}
          fill="none" stroke="currentColor" strokeWidth="2.5" strokeOpacity="0.15"
        />
        <circle
          cx={SIZE / 2} cy={SIZE / 2} r={RADIUS}
          fill="none" stroke="currentColor" strokeWidth="2.5"
          strokeDasharray={`${filled} ${CIRCUMFERENCE - filled}`}
          strokeLinecap="round"
          transform={`rotate(-90 ${SIZE / 2} ${SIZE / 2})`}
        />
      </svg>
      <span className="text-[10px] font-medium tabular-nums leading-none select-none">
        {displayPct}%
      </span>
    </div>
  );
}
