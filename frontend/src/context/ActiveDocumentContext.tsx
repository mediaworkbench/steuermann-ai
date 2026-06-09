"use client";

import { createContext, useContext, useState, useEffect, useRef } from "react";
import type { ConversationAttachment } from "@/lib/types";
import type { WorkspaceDocument } from "@/components/workspace/types";
import { useDocumentEditor } from "@/components/workspace/useDocumentEditor";
import { useVersionHistory } from "@/components/workspace/useVersionHistory";

type DocumentEditor = ReturnType<typeof useDocumentEditor>;
type VersionHistory = ReturnType<typeof useVersionHistory>;

type ActiveDocumentContextValue = DocumentEditor &
  VersionHistory & {
    /** Shared cross-action disable token (was DocumentsTab-local). */
    processingAction: string | null;
    setProcessingAction: (value: string | null) => void;
  };

interface ActiveDocumentProviderProps {
  documents: WorkspaceDocument[];
  conversationId?: string | null;
  writebackSavedDocId?: string | null;
  onActiveDocumentChange?: (docId: string | null) => void;
  onAttachmentUploaded?: (attachment: ConversationAttachment) => void;
  onDocumentsRefresh?: () => void;
  children: React.ReactNode;
}

const ActiveDocumentContext = createContext<ActiveDocumentContextValue | null>(null);

export function useActiveDocument(): ActiveDocumentContextValue {
  const ctx = useContext(ActiveDocumentContext);
  if (!ctx) throw new Error("useActiveDocument must be used within ActiveDocumentProvider");
  return ctx;
}

/** localStorage map: { [conversationId]: openDocId } */
const ACTIVE_DOC_KEY = "workspace.activeDoc";

function readActiveDocMap(): Record<string, string> {
  try {
    const raw = localStorage.getItem(ACTIVE_DOC_KEY);
    return raw ? (JSON.parse(raw) as Record<string, string>) : {};
  } catch {
    return {};
  }
}

/**
 * Single source of truth for the active workspace document: one
 * `useDocumentEditor` + one `useVersionHistory`, shared by the Documents tab's
 * inline editor and the split-view `ActiveDocumentPane` so they never run two
 * competing editors. Fed by `ChatInterface` (which owns documents/conversation).
 *
 * Also remembers the open document **per conversation** (localStorage), so
 * switching conversations and back restores the editor (today the active doc was
 * local to ChatInterface and lost on unmount).
 */
export function ActiveDocumentProvider({
  documents,
  conversationId,
  writebackSavedDocId,
  onActiveDocumentChange,
  onAttachmentUploaded,
  onDocumentsRefresh,
  children,
}: ActiveDocumentProviderProps) {
  const [processingAction, setProcessingAction] = useState<string | null>(null);

  const editor = useDocumentEditor({
    documents,
    conversationId,
    writebackSavedDocId,
    onActiveDocumentChange,
    onAttachmentUploaded,
    onDocumentsRefresh,
    setProcessingAction,
  });

  const history = useVersionHistory({
    onDocumentsRefresh,
    setProcessingAction,
    onAfterRestore: (docId) => {
      if (editor.editorDocId === docId) editor.openEditor(docId);
    },
  });

  const { editorDocId } = editor;
  const editorRef = useRef(editor);
  editorRef.current = editor;
  const conversationIdRef = useRef(conversationId);
  const restoredConvRef = useRef<string | null | undefined>(undefined);

  // ── Restore the per-conversation open doc on conversation switch ──
  // Runs again when `documents` arrive so a doc saved for this conversation can
  // be resolved even if the list loaded after the switch. The ref guard stops it
  // re-restoring (or fighting a manual close) within the same conversation.
  useEffect(() => {
    conversationIdRef.current = conversationId;
    if (restoredConvRef.current === conversationId) return;

    const storedId = conversationId ? (readActiveDocMap()[conversationId] ?? null) : null;
    if (storedId) {
      if (documents.some((d) => d.id === storedId)) {
        restoredConvRef.current = conversationId;
        if (editorRef.current.editorDocId !== storedId) editorRef.current.openEditor(storedId);
      }
      // else: doc list not ready yet — retry when `documents` changes.
      return;
    }
    // No saved doc for this conversation → clear any editor carried over.
    restoredConvRef.current = conversationId;
    if (editorRef.current.editorDocId !== null) editorRef.current.closeEditor();
  }, [conversationId, documents]);

  // ── Persist the open doc for the current conversation ──
  // Depends only on `editorDocId` (not `conversationId`) so a conversation switch
  // can't write the old doc under the new conversation before restore runs.
  useEffect(() => {
    const convId = conversationIdRef.current;
    if (!convId) return;
    try {
      const map = readActiveDocMap();
      if (editorDocId) map[convId] = editorDocId;
      else delete map[convId];
      localStorage.setItem(ACTIVE_DOC_KEY, JSON.stringify(map));
    } catch {
      /* ignore persistence failures */
    }
  }, [editorDocId]);

  const value: ActiveDocumentContextValue = {
    ...editor,
    ...history,
    processingAction,
    setProcessingAction,
  };

  return <ActiveDocumentContext.Provider value={value}>{children}</ActiveDocumentContext.Provider>;
}
