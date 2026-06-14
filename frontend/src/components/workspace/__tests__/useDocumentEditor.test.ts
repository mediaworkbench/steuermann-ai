import { act, renderHook } from "@testing-library/react";
import { toast } from "sonner";
import { useDocumentEditor } from "@/components/workspace/useDocumentEditor";

// ── Module mocks ──────────────────────────────────────────────────────────────

jest.mock("sonner", () => ({ toast: { error: jest.fn(), success: jest.fn(), warning: jest.fn() } }));
jest.mock("@/hooks/useI18n", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}));
const mockMimeTypeForFilename = jest.fn(() => "text/plain");

jest.mock("@/components/workspace/utils", () => ({
  workspaceAuthHeaders: (extra?: Record<string, string>) => ({ ...(extra ?? {}) }),
  mimeTypeForFilename: (name: string) => mockMimeTypeForFilename(name),
}));

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeDocRow(id: string, content = "Original content.", version = 1) {
  return { content_text: content, version, filename: `${id}.md` };
}

function makeHookArgs(overrides = {}) {
  return {
    documents: [{ id: "doc-1", filename: "doc-1.md", mime_type: "text/plain", size_bytes: 100, version: 1 }],
    conversationId: "conv-1",
    writebackSavedDocId: null,
    onActiveDocumentChange: jest.fn(),
    onDocumentsRefresh: jest.fn(),
    onAfterSave: jest.fn(),
    setProcessingAction: jest.fn(),
    ...overrides,
  };
}

function mockGetDoc(content: string, version = 1, status = 200) {
  global.fetch = jest.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => makeDocRow("doc-1", content, version),
  } as unknown as Response);
}

function mockPutDoc(status: number, body?: Record<string, unknown>) {
  global.fetch = jest.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 409 ? "Conflict" : status === 200 ? "OK" : "Error",
    json: async () => body ?? {},
  } as unknown as Response);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("useDocumentEditor", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("isDirty tracking", () => {
    it("is false when editor is closed", () => {
      const { result } = renderHook(() => useDocumentEditor(makeHookArgs()));
      expect(result.current.isDirty).toBe(false);
    });

    it("is false immediately after openEditor (content matches server)", async () => {
      mockGetDoc("Server content.");
      const { result } = renderHook(() => useDocumentEditor(makeHookArgs()));

      await act(async () => {
        await result.current.openEditor("doc-1");
      });

      expect(result.current.editorContent).toBe("Server content.");
      expect(result.current.isDirty).toBe(false);
    });

    it("becomes true when editorContent diverges from savedContent", async () => {
      mockGetDoc("Original.");
      const { result } = renderHook(() => useDocumentEditor(makeHookArgs()));

      await act(async () => {
        await result.current.openEditor("doc-1");
      });

      act(() => {
        result.current.setEditorContent("Edited.");
      });

      expect(result.current.isDirty).toBe(true);
    });

    it("goes back to false after a successful flushSave", async () => {
      mockGetDoc("Original.");
      const onAfterSave = jest.fn();
      const { result } = renderHook(() =>
        useDocumentEditor(makeHookArgs({ onAfterSave }))
      );

      await act(async () => {
        await result.current.openEditor("doc-1");
      });

      act(() => {
        result.current.setEditorContent("Edited content.");
      });

      expect(result.current.isDirty).toBe(true);

      // PUT succeeds — server returns version 2
      mockPutDoc(200, { document: { version: 2 } });

      await act(async () => {
        await result.current.flushSave();
      });

      expect(result.current.isDirty).toBe(false);
      expect(onAfterSave).toHaveBeenCalledWith("doc-1");
    });
  });

  describe("flushSave", () => {
    it("returns false and toasts conflict on 409 response", async () => {
      mockGetDoc("Original.", 1);
      const { result } = renderHook(() => useDocumentEditor(makeHookArgs()));

      await act(async () => {
        await result.current.openEditor("doc-1");
      });

      act(() => {
        result.current.setEditorContent("My edits.");
      });

      // First fetch (for 409 reload): return conflict, second fetch (openEditor reload): return new content
      const fetchMock = jest.fn()
        .mockResolvedValueOnce({
          ok: false,
          status: 409,
          statusText: "Conflict",
          json: async () => ({}),
        } as unknown as Response)
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => makeDocRow("doc-1", "Server updated content.", 2),
        } as unknown as Response);
      global.fetch = fetchMock;

      let saveResult = false;
      await act(async () => {
        saveResult = await result.current.flushSave();
      });

      expect(saveResult).toBe(false);
      expect(toast.error).toHaveBeenCalled();
      // After 409, editor reloads from server
      expect(result.current.editorContent).toBe("Server updated content.");
      expect(result.current.isDirty).toBe(false);
    });

    it("returns true (no-op success) and does nothing when editor is closed", async () => {
      const { result } = renderHook(() => useDocumentEditor(makeHookArgs()));

      let saveResult = false;
      await act(async () => {
        saveResult = await result.current.flushSave();
      });

      expect(saveResult).toBe(true);
      expect(global.fetch).not.toHaveBeenCalled();
    });

    it("sends expected_version in FormData when a version is set", async () => {
      mockGetDoc("Original.", 5);
      const { result } = renderHook(() => useDocumentEditor(makeHookArgs()));

      await act(async () => {
        await result.current.openEditor("doc-1");
      });

      act(() => {
        result.current.setEditorContent("Edited.");
      });

      const capturedFormData: FormData[] = [];
      const fetchMock = jest.fn().mockImplementation((url: string, opts: RequestInit) => {
        if (opts.method === "PUT") {
          capturedFormData.push(opts.body as FormData);
          return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({ document: { version: 6 } }),
          } as unknown as Response);
        }
        return Promise.resolve({ ok: true, status: 200, json: async () => ({}) } as unknown as Response);
      });
      global.fetch = fetchMock;

      await act(async () => {
        await result.current.flushSave();
      });

      expect(capturedFormData).toHaveLength(1);
      const fd = capturedFormData[0];
      expect(fd.get("expected_version")).toBe("5");
    });

    it("updates editorDocVersion from server response after successful save", async () => {
      mockGetDoc("Original.", 1);
      const { result } = renderHook(() => useDocumentEditor(makeHookArgs()));

      await act(async () => {
        await result.current.openEditor("doc-1");
      });

      act(() => {
        result.current.setEditorContent("Updated.");
      });

      mockPutDoc(200, { document: { version: 2 } });

      await act(async () => {
        await result.current.flushSave();
      });

      // The next save attempt should send expected_version=2
      const capturedFormData: FormData[] = [];
      global.fetch = jest.fn().mockImplementation((url: string, opts: RequestInit) => {
        if (opts.method === "PUT") capturedFormData.push(opts.body as FormData);
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ document: { version: 3 } }),
        } as unknown as Response);
      });

      act(() => {
        result.current.setEditorContent("Updated again.");
      });

      await act(async () => {
        await result.current.flushSave();
      });

      expect(capturedFormData[0].get("expected_version")).toBe("2");
    });

    it("uses text/csv MIME type when saving a .csv document", async () => {
      mockMimeTypeForFilename.mockImplementation((name: string) =>
        name.endsWith(".csv") ? "text/csv" : "text/plain"
      );

      const csvDoc = { id: "doc-csv", filename: "data.csv", mime_type: "text/csv", size_bytes: 200, version: 1 };
      global.fetch = jest.fn()
        .mockResolvedValueOnce({
          ok: true, status: 200,
          json: async () => ({ content_text: "col1,col2\n1,2", version: 1, filename: "data.csv" }),
        } as unknown as Response)
        .mockResolvedValueOnce({
          ok: true, status: 200,
          json: async () => ({ document: { version: 2 } }),
        } as unknown as Response);

      const { result } = renderHook(() =>
        useDocumentEditor(makeHookArgs({ documents: [csvDoc] }))
      );

      await act(async () => { await result.current.openEditor("doc-csv"); });
      act(() => { result.current.setEditorContent("col1,col2\n1,99"); });

      const capturedFormData: FormData[] = [];
      const putFetch = jest.fn().mockImplementation((_url: string, opts: RequestInit) => {
        if (opts.method === "PUT") capturedFormData.push(opts.body as FormData);
        return Promise.resolve({
          ok: true, status: 200,
          json: async () => ({ document: { version: 2 } }),
        } as unknown as Response);
      });
      global.fetch = putFetch;

      await act(async () => { await result.current.flushSave(); });

      expect(capturedFormData).toHaveLength(1);
      const uploadedFile = capturedFormData[0].get("file") as File;
      expect(uploadedFile.type).toBe("text/csv");

      mockMimeTypeForFilename.mockReset();
      mockMimeTypeForFilename.mockImplementation(() => "text/plain");
    });
  });

  describe("writeback reload", () => {
    it("reloads the editor when writebackSavedDocId matches the open doc and editor is clean", async () => {
      mockGetDoc("Initial content.", 1);
      const { result, rerender } = renderHook(
        (args) => useDocumentEditor(args),
        { initialProps: makeHookArgs({ writebackSavedDocId: null }) }
      );

      await act(async () => {
        await result.current.openEditor("doc-1");
      });

      expect(result.current.editorContent).toBe("Initial content.");
      // Editor is clean — writeback reload should trigger openEditor again
      mockGetDoc("Model improved content.", 2);

      await act(async () => {
        rerender(makeHookArgs({ writebackSavedDocId: "doc-1" }));
        // Allow the effect + async openEditor to run
        await new Promise((r) => setTimeout(r, 0));
      });

      expect(result.current.editorContent).toBe("Model improved content.");
    });

    it("does NOT reload when editor has unsaved changes (isDirty=true)", async () => {
      mockGetDoc("Initial content.", 1);
      const { result, rerender } = renderHook(
        (args) => useDocumentEditor(args),
        { initialProps: makeHookArgs({ writebackSavedDocId: null }) }
      );

      await act(async () => {
        await result.current.openEditor("doc-1");
      });

      act(() => {
        result.current.setEditorContent("My unsaved edits.");
      });

      expect(result.current.isDirty).toBe(true);

      const fetchCallCount = (global.fetch as jest.Mock).mock.calls.length;

      await act(async () => {
        rerender(makeHookArgs({ writebackSavedDocId: "doc-1" }));
        await new Promise((r) => setTimeout(r, 0));
      });

      // fetch should NOT have been called again (no reload)
      expect((global.fetch as jest.Mock).mock.calls.length).toBe(fetchCallCount);
      // Content is preserved
      expect(result.current.editorContent).toBe("My unsaved edits.");
      expect(toast.warning).toHaveBeenCalled();
    });
  });
});
