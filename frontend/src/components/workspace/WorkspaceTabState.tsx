import { Button } from "@/components/ui/button";
import { iconMap } from "@/lib/iconMap";

type Tone = "idle" | "loading" | "error";

interface WorkspaceTabStateProps {
  icon: string;
  title: string;
  hint?: string;
  tone?: Tone;
  action?: { label: string; onClick: () => void };
}

/**
 * Shared centered state for workspace tabs: idle/empty, loading (spinner), and
 * error (with an optional retry action). Used by DocumentsTab and the read-only
 * evidence tabs so all states look consistent.
 */
export function WorkspaceTabState({ icon, title, hint, tone = "idle", action }: WorkspaceTabStateProps) {
  const isError = tone === "error";
  const isLoading = tone === "loading";
  const StateIcon = iconMap[icon];

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
      <span
        className={`grid place-items-center w-12 h-12 rounded-2xl mb-3 ${
          isError ? "bg-destructive/10 text-destructive" : "bg-surface-muted text-muted-foreground"
        }`}
      >
        {isLoading ? (
          <span className="w-6 h-6 rounded-full border-2 border-primary/25 border-t-primary animate-spin" />
        ) : (
          <StateIcon size={24} />
        )}
      </span>
      <p className={`text-xs font-semibold mb-1 ${isError ? "text-destructive" : "text-muted-foreground"}`}>
        {title}
      </p>
      {hint && <p className="text-xs text-muted-foreground max-w-52 wrap-break-word">{hint}</p>}
      {action && (
        <Button
          type="button"
          onClick={action.onClick}
          variant="secondary"
          size="sm"
          className="mt-3"
        >
          {action.label}
        </Button>
      )}
    </div>
  );
}
