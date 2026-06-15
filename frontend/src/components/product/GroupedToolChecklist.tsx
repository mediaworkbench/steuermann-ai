"use client";

import { OptionChecklist } from "@/components/product/OptionChecklist";
import { groupTools } from "@/lib/toolGroups";
import { useI18n } from "@/hooks/useI18n";
import type { ToolCatalogItem } from "@/lib/api";

interface GroupedToolChecklistProps {
  tools: ToolCatalogItem[];
  isChecked: (toolId: string) => boolean;
  onToggle: (toolId: string) => void;
}

/**
 * Renders a tool catalog as three responsive columns (Text · Vision · Auxiliary),
 * each an OptionChecklist. Shared by the settings page and the admin role editor.
 */
export function GroupedToolChecklist({ tools, isChecked, onToggle }: GroupedToolChecklistProps) {
  const { t } = useI18n();
  const buckets = groupTools(tools);

  if (buckets.length === 0) {
    return <p className="text-sm text-muted-foreground">{t("settingsPanel.noToolsAvailable")}</p>;
  }

  return (
    <div className="grid gap-6 md:grid-cols-3">
      {buckets.map((bucket) => (
        <div key={bucket.group} className="space-y-3">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {t(`toolGroups.${bucket.group}`)}
          </h4>
          <OptionChecklist
            items={bucket.items.map((tool) => ({
              key: tool.id,
              checked: isChecked(tool.id),
              onToggle: () => onToggle(tool.id),
              label: tool.label,
              alignment: "center" as const,
              checkboxClassName: "w-5 h-5",
            }))}
          />
        </div>
      ))}
    </div>
  );
}
