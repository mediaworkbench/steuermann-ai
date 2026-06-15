import type { ToolCatalogItem, ToolGroup } from "@/lib/api";

// Fixed column order on the admin + settings pages.
export const TOOL_GROUP_ORDER: ToolGroup[] = ["text", "vision", "auxiliary"];

export interface ToolGroupBucket {
  group: ToolGroup;
  items: ToolCatalogItem[];
}

/**
 * Bucket tools into the three UI groups in fixed order, dropping empty groups.
 * An item with a missing/unknown group falls back to "auxiliary" so it always
 * lands in a column.
 */
export function groupTools(items: ToolCatalogItem[]): ToolGroupBucket[] {
  const buckets: Record<ToolGroup, ToolCatalogItem[]> = { text: [], vision: [], auxiliary: [] };
  for (const item of items) {
    const group: ToolGroup = item.group && item.group in buckets ? item.group : "auxiliary";
    buckets[group].push(item);
  }
  return TOOL_GROUP_ORDER.map((group) => ({ group, items: buckets[group] })).filter(
    (bucket) => bucket.items.length > 0
  );
}
