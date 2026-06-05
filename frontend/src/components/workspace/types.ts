import type { ConversationAttachment } from "@/lib/types";

export interface VersionEntry {
  id: string;
  document_id: string;
  version: number;
  size_bytes: number;
  sha256: string;
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

export type WorkspaceTabId = "documents" | "knowledge" | "memory" | "outputs";

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
  writebackSavedDocId?: string | null;
  onActiveDocumentChange?: (docId: string | null) => void;
}
