import { Button } from "@/components/ui/Button";
import { Icon } from "../Icon";

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

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
      <span
        className={`grid place-items-center w-12 h-12 rounded-2xl mb-3 ${
          isError ? "bg-red-50 text-red-400" : "bg-gray-100 text-evergreen/30"
        }`}
      >
        {isLoading ? (
          <span className="w-6 h-6 rounded-full border-2 border-pacific-blue/25 border-t-pacific-blue animate-spin" />
        ) : (
          <Icon name={icon} size={24} />
        )}
      </span>
      <p className={`text-xs font-semibold mb-1 ${isError ? "text-red-500" : "text-evergreen/60"}`}>
        {title}
      </p>
      {hint && <p className="text-xs text-evergreen/40 max-w-[13rem] break-words">{hint}</p>}
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
