export type MetricsSaveStatus = "idle" | "saving" | "saved" | "error";

interface MetricsSaveStatusIndicatorProps {
  status: MetricsSaveStatus;
  savingLabel: string;
  savedLabel: string;
  errorLabel: string;
}

export function MetricsSaveStatusIndicator({
  status,
  savingLabel,
  savedLabel,
  errorLabel,
}: MetricsSaveStatusIndicatorProps) {
  if (status === "idle") {
    return null;
  }

  return (
    <div className="inline-flex items-center gap-2 text-sm text-muted-foreground">
      {status === "saving" && (
        <>
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-border border-t-primary" />
          <span>{savingLabel}</span>
        </>
      )}
      {status === "saved" && (
        <>
          <span className="font-bold text-success">✓</span>
          <span>{savedLabel}</span>
        </>
      )}
      {status === "error" && (
        <>
          <span className="font-bold text-destructive">✕</span>
          <span>{errorLabel}</span>
        </>
      )}
    </div>
  );
}
