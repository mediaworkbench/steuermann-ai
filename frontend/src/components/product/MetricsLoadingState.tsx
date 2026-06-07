import { Icon } from "@/components/Icon";

interface MetricsLoadingStateProps {
  label: string;
}

export function MetricsLoadingState({ label }: MetricsLoadingStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
      <Icon name="refresh" size={32} className="animate-spin" />
      <p className="mt-4 mb-0">{label}</p>
    </div>
  );
}
