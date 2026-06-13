"use client";

import { createContext, useContext, useState, useEffect, useLayoutEffect, useRef, useCallback } from "react";
import type { ConversationAttachment } from "@/lib/types";
import type { WorkspaceDocument } from "@/components/workspace/types";
import { useDocumentEditor } from "@/components/workspace/useDocumentEditor";
import { useVersionHistory } from "@/components/workspace/useVersionHistory";

type DocumentEditor = ReturnType<typeof useDocumentEditor>;
type VersionHistory = ReturnType<typeof useVersionHistory>;

/** Minimal editor surface exposed to the chat composer for auto-save-before-send. */
export interface ActiveDocumentEditorApi {
  isDirty: boolean;
  flushSave: () => Promise<boolean>;
}

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
  /**
   * Filled by the provider with the live editor API so a component rendered
   * *above* this provider (the chat composer in `ChatInterface`) can flush
   * unsaved edits before sending a message.
   */
  editorApiRef?: React.MutableRefObject<ActiveDocumentEditorApi | null>;
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
  editorApiRef,
  children,
}: ActiveDocumentProviderProps) {
  const [processingAction, setProcessingAction] = useState<string | null>(null);
  // Forward ref to the history hook so the editor's onAfterSave can refresh an
  // open version-history panel (history is created after the editor below).
  const historyRef = useRef<VersionHistory | null>(null);

  const editor = useDocumentEditor({
    documents,
    conversationId,
    writebackSavedDocId,
    onActiveDocumentChange,
    onAttachmentUploaded,
    onDocumentsRefresh,
    onAfterSave: (docId) => {
      const h = historyRef.current;
      if (h && h.historyDocId === docId) h.loadHistory(docId);
    },
    setProcessingAction,
  });

  const history = useVersionHistory({
    onDocumentsRefresh,
    setProcessingAction,
    onAfterRestore: (docId) => {
      if (editor.editorDocId === docId) editor.openEditor(docId);
    },
  });

  const { editorDocId, isDirty, flushSave } = editor;
  const editorRef = useRef(editor);
  useLayoutEffect(() => { editorRef.current = editor; });
  // Mirror the latest history hook into the ref so the editor's onAfterSave (and
  // the writeback effect) can reach loadHistory without a render-time ref write.
  useLayoutEffect(() => { historyRef.current = history; });

  // Expose the live editor API to the chat composer (rendered above this provider).
  useLayoutEffect(() => {
    if (!editorApiRef) return;
    editorApiRef.current = { isDirty, flushSave };
    return () => {
      editorApiRef.current = null;
    };
  }, [editorApiRef, isDirty, flushSave]);

  // Keep an open history panel in sync after the model writes back a new version.
  // Only `writebackSavedDocId` matters; `historyRef` is a ref (stable).
  useEffect(() => {
    if (writebackSavedDocId && historyRef.current?.historyDocId === writebackSavedDocId) {
      historyRef.current.loadHistory(writebackSavedDocId);
    }
  }, [writebackSavedDocId]);

  const conversationIdRef = useRef(conversationId);
  const restoredConvRef = useRef<string | null | undefined>(undefined);

  // ── Restore the per-conversation open doc on conversation switch ──
  // Runs again when `documents` arrive so a doc saved for this conversation can
  // be resolved even if the list loaded after the switch. The ref guard stops it
  // re-restoring (or fighting a manual close) within the same conversation.
  // A dirty editor is auto-saved before being replaced so edits aren't lost.
  useEffect(() => {
    conversationIdRef.current = conversationId;
    if (restoredConvRef.current === conversationId) return;

    // The version-history panel is transient per-conversation view state: close
    // it on a real switch so the pane (which can mount on history alone) doesn't
    // linger showing the previous conversation's document.
    historyRef.current?.closeHistory();

    const swap = (fn: () => void) => {
      const ed = editorRef.current;
      if (ed.isDirty && ed.editorDocId) {
        void ed.flushSave().finally(fn);
      } else {
        fn();
      }
    };

    const storedId = conversationId ? (readActiveDocMap()[conversationId] ?? null) : null;
    if (storedId) {
      if (documents.some((d) => d.id === storedId)) {
        restoredConvRef.current = conversationId;
        if (editorRef.current.editorDocId !== storedId) {
          swap(() => editorRef.current.openEditor(storedId));
        }
      }
      // else: doc list not ready yet — retry when `documents` changes.
      return;
    }
    // No saved doc for this conversation → clear any editor carried over.
    restoredConvRef.current = conversationId;
    if (editorRef.current.editorDocId !== null) swap(() => editorRef.current.closeEditor({ force: true }));
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

  // Restoring while the editor holds unsaved edits: flush them first so they
  // become their own version before the restore creates the next one (no silent loss).
  const restoreVersion = useCallback(
    async (docId: string, version: number) => {
      const ed = editorRef.current;
      if (ed.isDirty && ed.editorDocId === docId) {
        const ok = await ed.flushSave();
        if (!ok) return; // save failed (e.g. conflict) — abort restore, editor reloaded
      }
      await historyRef.current?.restoreVersion(docId, version);
    },
    [],
  );

  const value: ActiveDocumentContextValue = {
    ...editor,
    ...history,
    restoreVersion,
    processingAction,
    setProcessingAction,
  };

  return <ActiveDocumentContext.Provider value={value}>{children}</ActiveDocumentContext.Provider>;
}
