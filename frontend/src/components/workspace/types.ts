import type { ConversationAttachment, MessageMetrics, NodeTraceEntry } from "@/lib/types";

export type VersionSource = "user" | "assistant" | "restore";

export interface VersionEntry {
  id: string;
  document_id: string;
  version: number;
  size_bytes: number;
  sha256: string;
  /** Who created this version's content: a user edit, the assistant, or a restore. */
  source?: VersionSource | null;
  created_at?: string;
}

export interface WorkspaceDocument {
  id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  version: number;
  created_at?: string;
  updated_at?: string;
}

export type WorkspaceTabId = "documents" | "knowledge" | "memory" | "outputs" | "inspector";

/**
 * Shared props for the workspace panel. `WorkspaceSidebar` re-exports this as
 * `WorkspaceSidebarProps` so existing callers keep compiling unchanged.
 */
export interface WorkspacePanelProps {
  isOpen: boolean;
  onToggle: () => void;
  conversationId?: string | null;
  documents: WorkspaceDocument[];
  isLoading?: boolean;
  onDocumentsRefresh?: () => void;
  onEnsureConversation?: () => Promise<string | null>;
  onAttachmentUploaded?: (attachment: ConversationAttachment) => void;
  /** True while the document list is being (re)fetched by the parent. */
  documentsLoading?: boolean;
  /** Non-null when the last document-list fetch failed (message for display). */
  documentsError?: string | null;
  /** Retry handler for a failed document-list fetch. */
  onRetryDocuments?: () => void;
  /**
   * Metrics of the latest assistant answer in the active conversation. Drives
   * the read-only evidence tabs (Knowledge/Memory/Outputs). Null when there is
   * no answer yet. Already active-conversation-scoped by the caller.
   */
  answerMetrics?: MessageMetrics | null;
  /**
   * Ordered per-node execution trace of the latest/live answer (Inspector tab).
   * Already active-conversation-gated by the caller. Session-local: not persisted.
   */
  nodeTrace?: NodeTraceEntry[];
  /** True while the active conversation's answer is streaming (drives Inspector live state). */
  isStreaming?: boolean;
  /** True when the panel is pinned to an earlier (non-latest) answer — shows a banner. */
  historicalAnswer?: boolean;
  /** Returns the panel to following the latest answer (banner "Jump to latest"). */
  onJumpToLatest?: () => void;
}
