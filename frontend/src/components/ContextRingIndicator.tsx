"use client";

interface ContextRingIndicatorProps {
  contextTokens: number;
  maxContextTokens: number | null | undefined;
}

const SIZE = 22;
const RADIUS = 9;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export function ContextRingIndicator({ contextTokens, maxContextTokens }: ContextRingIndicatorProps) {
  if (!maxContextTokens) return null;

  const pct = Math.min(1, contextTokens / maxContextTokens);
  const displayPct = Math.round(pct * 100);
  const filled = pct * CIRCUMFERENCE;

  const colorClass =
    contextTokens === 0 ? "text-evergreen/30"
    : pct < 0.6        ? "text-evergreen/60"
    : pct < 0.85       ? "text-amber-500"
    :                    "text-red-500";

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
