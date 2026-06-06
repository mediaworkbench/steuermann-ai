"use client";

import { WorkspacePanel } from "./workspace/WorkspacePanel";
import type { WorkspacePanelProps } from "./workspace/types";

// Re-exported for backwards compatibility — existing callers import the type
// from this module. The implementation now lives in ./workspace/.
export type { WorkspaceDocument } from "./workspace/types";
export type WorkspaceSidebarProps = WorkspacePanelProps;

/**
 * Compatibility wrapper. The workspace UI was extracted into the modular
 * `./workspace/` panel (tabbed sections); this preserves the original
 * export/props shape so mount points keep working unchanged.
 */
export function WorkspaceSidebar(props: WorkspaceSidebarProps) {
  return <WorkspacePanel {...props} />;
}
