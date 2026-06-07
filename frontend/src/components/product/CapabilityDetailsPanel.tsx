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
      <div>
        <span className="font-semibold">{labels.configuredMode}: </span>
        <span>{item.configured_tool_calling_mode || labels.na}</span>
      </div>
      <div>
        <span className="font-semibold">{labels.apiBase}: </span>
        <span>{item.api_base || labels.na}</span>
      </div>
      <div>
        <span className="font-semibold">{labels.error}: </span>
        <span>{item.error_message || labels.na}</span>
      </div>
      <div>
        <span className="font-semibold">{labels.bindTools}: </span>
        <span>{item.supports_bind_tools === null ? labels.na : String(item.supports_bind_tools)}</span>
      </div>
      <div>
        <span className="font-semibold">{labels.vision}: </span>
        <span>
          {item.supports_vision === null || item.supports_vision === undefined
            ? labels.na
            : String(item.supports_vision)}
        </span>
      </div>
      <div>
        <span className="font-semibold">{labels.reasoning}: </span>
        <span>{String(item.supports_reasoning ?? false)}</span>
      </div>
      <div>
        <span className="font-semibold">{labels.mismatch}: </span>
        <span>{String(item.capability_mismatch)}</span>
      </div>
      <div className="md:col-span-2">
        <div className="mb-1 font-semibold">{labels.metadata}</div>
        <pre className="overflow-x-auto rounded border border-border bg-surface p-2">
          {JSON.stringify(item.metadata || {}, null, 2)}
        </pre>
      </div>
    </div>
  );
}
