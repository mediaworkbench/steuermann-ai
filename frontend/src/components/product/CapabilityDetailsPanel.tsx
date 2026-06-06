import { LabeledValue } from "@/components/product/LabeledValue";
import type { LLMCapabilityItem } from "@/lib/api";

interface CapabilityDetailsPanelProps {
  item: LLMCapabilityItem;
  labels: {
    configuredMode: string;
    apiBase: string;
    error: string;
    bindTools: string;
    vision: string;
    reasoning: string;
    mismatch: string;
    metadata: string;
    na: string;
  };
}

export function CapabilityDetailsPanel({ item, labels }: CapabilityDetailsPanelProps) {
  return (
    <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
      <LabeledValue label={labels.configuredMode} value={item.configured_tool_calling_mode || labels.na} />
      <LabeledValue label={labels.apiBase} value={item.api_base || labels.na} />
      <LabeledValue label={labels.error} value={item.error_message || labels.na} />
      <LabeledValue
        label={labels.bindTools}
        value={item.supports_bind_tools === null ? labels.na : String(item.supports_bind_tools)}
      />
      <LabeledValue
        label={labels.vision}
        value={
          item.supports_vision === null || item.supports_vision === undefined
            ? labels.na
            : String(item.supports_vision)
        }
      />
      <LabeledValue label={labels.reasoning} value={String(item.supports_reasoning ?? false)} />
      <LabeledValue label={labels.mismatch} value={String(item.capability_mismatch)} />
      <div className="md:col-span-2">
        <div className="mb-1 font-semibold">{labels.metadata}</div>
        <pre className="overflow-x-auto rounded border border-border bg-surface p-2">
          {JSON.stringify(item.metadata || {}, null, 2)}
        </pre>
      </div>
    </div>
  );
}
