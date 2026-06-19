import { useEffect, type ReactElement } from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { WorkspacePanel } from "../WorkspacePanel";
import { EvidenceChips } from "../EvidenceChips";
import { InspectorTab } from "../InspectorTab";
import { ActiveDocumentPane, ActiveDocumentPaneSlot } from "../ActiveDocumentPane";
import { DocumentEditorView } from "../DocumentEditorView";
import { useActiveDocument } from "@/context/ActiveDocumentContext";
import { WorkspaceSidebar } from "@/components/WorkspaceSidebar";
import { WorkspacePanelProvider } from "@/context/WorkspacePanelContext";
import { ActiveDocumentProvider } from "@/context/ActiveDocumentContext";
import { useI18n } from "@/hooks/useI18n";
import type { WorkspaceDocument } from "../types";
import type { MessageMetrics, NodeTraceEntry } from "@/lib/types";

jest.mock("@/hooks/useI18n");

const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;

const doc = (id: string, filename: string, mimeType = "text/plain"): WorkspaceDocument => ({
  id,
  filename,
  mime_type: mimeType,
  size_bytes: 100,
  version: 1,
});

const baseProps = {
  isOpen: true,
  onToggle: jest.fn(),
  documents: [],
};

// WorkspacePanel/DocumentsTab read internal view state from WorkspacePanelContext
// and the lifted editor state from ActiveDocumentContext (one shared editor).
const renderWithPanel = (ui: ReactElement, documents: WorkspaceDocument[] = []) =>
  render(
    <ActiveDocumentProvider documents={documents}>
      <WorkspacePanelProvider>{ui}</WorkspacePanelProvider>
    </ActiveDocumentProvider>,
  );

beforeEach(() => {
  jest.clearAllMocks();
  localStorage.clear();
  mockUseI18n.mockReturnValue({
    locale: "en",
    setLocale: jest.fn(),
    // Echo the key so assertions can target stable i18n keys.
    t: (key: string) => key,
    formatDate: () => "",
    formatTime: () => "",
    formatDateTime: () => "",
    formatNumber: (value: number) => String(value),
    formatRelativeTime: () => "",
  });
});

describe("WorkspacePanel", () => {
  test("renders the workspace tabs with Documents active by default", () => {
    renderWithPanel(<WorkspacePanel {...baseProps} />);
    expect(screen.getAllByRole("tab")).toHaveLength(5);
    // Documents tab is active → its empty state shows.
    expect(screen.getByText("workspace.noDocuments")).toBeInTheDocument();
  });

  test("switches to the Knowledge evidence tab", () => {
    renderWithPanel(<WorkspacePanel {...baseProps} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabKnowledge/ }));
    expect(screen.getByText("workspace.knowledgeEmpty")).toBeInTheDocument();
  });

  test("switches to the Memory and Outputs evidence tabs", () => {
    renderWithPanel(<WorkspacePanel {...baseProps} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabMemory/ }));
    expect(screen.getByText("workspace.memoryEmpty")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabOutputs/ }));
    expect(screen.getByText("workspace.outputsEmpty")).toBeInTheDocument();
  });

  test("compat wrapper WorkspaceSidebar renders the modular panel", () => {
    renderWithPanel(<WorkspaceSidebar {...baseProps} />);
    expect(screen.getAllByRole("tab")).toHaveLength(5);
    expect(screen.getByText("workspace.noDocuments")).toBeInTheDocument();
  });

  test("shows the documents count badge and filters by search", () => {
    renderWithPanel(<WorkspacePanel {...baseProps} documents={[doc("1", "alpha.md"), doc("2", "beta.txt")]} />);
    expect(screen.getByRole("tab", { name: /workspace\.tabDocuments/ })).toHaveTextContent("2");
    expect(screen.getByText("alpha.md")).toBeInTheDocument();
    expect(screen.getByText("beta.txt")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("workspace.searchDocuments"), {
      target: { value: "alpha" },
    });
    expect(screen.getByText("alpha.md")).toBeInTheDocument();
    expect(screen.queryByText("beta.txt")).not.toBeInTheDocument();
  });

  test("renders the CSV icon (Grid3x3) for a text/csv document", () => {
    const docs = [doc("1", "data.csv", "text/csv")];
    renderWithPanel(<WorkspacePanel {...baseProps} documents={docs} />, docs);
    expect(screen.getByText("data.csv")).toBeInTheDocument();
    expect(screen.getByTestId("csv-icon")).toBeInTheDocument();
  });

  test("renders the CSV icon when doc has no mime_type but .csv extension", () => {
    const csvDocNoMime: WorkspaceDocument = { id: "2", filename: "export.csv", mime_type: "", size_bytes: 50, version: 1 };
    renderWithPanel(<WorkspacePanel {...baseProps} documents={[csvDocNoMime]} />, [csvDocNoMime]);
    expect(screen.getByTestId("csv-icon")).toBeInTheDocument();
  });

  test("does not render the CSV icon for a non-CSV document", () => {
    const docs = [doc("1", "notes.md")];
    renderWithPanel(<WorkspacePanel {...baseProps} documents={docs} />, docs);
    expect(screen.queryByTestId("csv-icon")).not.toBeInTheDocument();
  });

  test("each document row exposes a kebab actions menu (no expandable view)", () => {
    const docs = [doc("1", "alpha.md")];
    renderWithPanel(<WorkspacePanel {...baseProps} documents={docs} />, docs);
    // Single actions trigger per row; no chevron/expand toggle.
    const trigger = screen.getByRole("button", { name: "workspace.documentActions" });
    fireEvent.click(trigger);
    // Consolidated actions appear in the menu.
    expect(screen.getByText("workspace.edit")).toBeInTheDocument();
    expect(screen.getByText("workspace.rename")).toBeInTheDocument();
    expect(screen.getByText("workspace.delete")).toBeInTheDocument();
  });

  test("the Rename menu action reveals the inline rename input", () => {
    const docs = [doc("1", "alpha.md")];
    renderWithPanel(<WorkspacePanel {...baseProps} documents={docs} />, docs);
    fireEvent.click(screen.getByRole("button", { name: "workspace.documentActions" }));
    fireEvent.click(screen.getByText("workspace.rename"));
    // Inline rename input is seeded with the current filename; Save/Cancel appear.
    expect(screen.getByDisplayValue("alpha.md")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "common.save" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "common.cancel" })).toBeInTheDocument();
  });

  test("shows the no-results state when search matches nothing", () => {
    renderWithPanel(<WorkspacePanel {...baseProps} documents={[doc("1", "alpha.md")]} />);
    fireEvent.change(screen.getByLabelText("workspace.searchDocuments"), {
      target: { value: "zzz" },
    });
    expect(screen.getByText("workspace.noResults")).toBeInTheDocument();
  });

  test("renders the plain list at the threshold but the windowed list above it", () => {
    // At the threshold (50): plain rendering → every row is mounted, no windowing.
    const fifty = Array.from({ length: 50 }, (_, i) => doc(String(i), `doc-${i}.md`));
    const { unmount } = renderWithPanel(<WorkspacePanel {...baseProps} documents={fifty} />, fifty);
    expect(screen.queryByTestId("virtualized-doc-list")).not.toBeInTheDocument();
    expect(screen.getByText("doc-49.md")).toBeInTheDocument();
    unmount();

    // Above the threshold (51): the list switches to the windowed container.
    const fiftyOne = Array.from({ length: 51 }, (_, i) => doc(String(i), `doc-${i}.md`));
    renderWithPanel(<WorkspacePanel {...baseProps} documents={fiftyOne} />, fiftyOne);
    expect(screen.getByRole("tab", { name: /workspace\.tabDocuments/ })).toHaveTextContent("51");
    expect(screen.getByTestId("virtualized-doc-list")).toBeInTheDocument();
  });

  test("renders the loading state while documents are fetching", () => {
    renderWithPanel(<WorkspacePanel {...baseProps} documents={[]} documentsLoading />);
    expect(screen.getByText("workspace.loadingDocuments")).toBeInTheDocument();
  });

  test("renders the error state with a working retry action", () => {
    const onRetryDocuments = jest.fn();
    renderWithPanel(
      <WorkspacePanel
        {...baseProps}
        documents={[]}
        documentsError="boom"
        onRetryDocuments={onRetryDocuments}
      />,
    );
    expect(screen.getByText("workspace.documentsLoadError")).toBeInTheDocument();
    fireEvent.click(screen.getByText("workspace.retry"));
    expect(onRetryDocuments).toHaveBeenCalledTimes(1);
  });

  test("Memory tab shows recalled memories with a count badge", () => {
    const answerMetrics: MessageMetrics = { memories_used: [{ memory_id: "m1", text: "likes tea" }] };
    renderWithPanel(<WorkspacePanel {...baseProps} answerMetrics={answerMetrics} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabMemory/ }));
    // Count shows on the active tab.
    expect(screen.getByRole("tab", { name: /workspace\.tabMemory/ })).toHaveTextContent("1");
    expect(screen.getByText("likes tea")).toBeInTheDocument();
  });

  test("Outputs tab lists executed tools, excluding the knowledge_base pseudo-tool", () => {
    const answerMetrics: MessageMetrics = {
      tools_executed: [
        { name: "web_search_mcp", status: "success" },
        { name: "knowledge_base", status: "success" },
      ],
    };
    renderWithPanel(<WorkspacePanel {...baseProps} answerMetrics={answerMetrics} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabOutputs/ }));
    expect(screen.getByText("web_search_mcp")).toBeInTheDocument();
    expect(screen.queryByText("knowledge_base")).not.toBeInTheDocument();
  });

  test("Outputs tab renders tool result detail with expandable args and output", () => {
    const answerMetrics: MessageMetrics = {
      tools_executed: [{ name: "web_search_mcp", status: "success" }],
      tool_results_detail: [
        {
          name: "web_search_mcp",
          status: "success",
          summary: "3 results found",
          args: { query: "weather" },
          output: "Sunny, 21C",
        },
      ],
    };
    renderWithPanel(<WorkspacePanel {...baseProps} answerMetrics={answerMetrics} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabOutputs/ }));
    // Collapsed: name + one-line summary visible, args/output hidden.
    expect(screen.getByText("web_search_mcp")).toBeInTheDocument();
    expect(screen.getByText("3 results found")).toBeInTheDocument();
    expect(screen.queryByText("workspace.toolArgs")).not.toBeInTheDocument();
    // Expand the row → args + output reveal.
    fireEvent.click(screen.getByRole("button", { name: /web_search_mcp/ }));
    expect(screen.getByText("workspace.toolArgs")).toBeInTheDocument();
    expect(screen.getByText("query:")).toBeInTheDocument();
    expect(screen.getByText("weather")).toBeInTheDocument();
    expect(screen.getByText("workspace.toolOutput")).toBeInTheDocument();
    expect(screen.getByText("Sunny, 21C")).toBeInTheDocument();
  });

  test("Inspector tool-calling node deep-links to the Outputs tab when tools ran", () => {
    const nodeTrace: NodeTraceEntry[] = [
      { node: "call_tools_native", sequence: 1, durationMs: 50, status: "success" },
      { node: "respond", sequence: 2, durationMs: 100, status: "success" },
    ];
    const answerMetrics: MessageMetrics = { tools_executed: [{ name: "web_search_mcp", status: "success" }] };
    renderWithPanel(<WorkspacePanel {...baseProps} nodeTrace={nodeTrace} answerMetrics={answerMetrics} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabInspector/ }));
    fireEvent.click(screen.getByRole("button", { name: /workspace\.viewToolResults/ }));
    expect(screen.getByRole("tab", { name: /workspace\.tabOutputs/ })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  test("Inspector tool-calling node is not a deep-link when no tools ran", () => {
    const nodeTrace: NodeTraceEntry[] = [
      { node: "call_tools_native", sequence: 1, durationMs: 50, status: "success" },
      { node: "respond", sequence: 2, durationMs: 100, status: "success" },
    ];
    renderWithPanel(<WorkspacePanel {...baseProps} nodeTrace={nodeTrace} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabInspector/ }));
    expect(screen.queryByRole("button", { name: /workspace\.viewToolResults/ })).not.toBeInTheDocument();
  });

  test("Knowledge tab shows the RAG retrieval summary", () => {
    const answerMetrics: MessageMetrics = { rag_attempted: true, rag_doc_count: 3 };
    renderWithPanel(<WorkspacePanel {...baseProps} answerMetrics={answerMetrics} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabKnowledge/ }));
    expect(screen.getByText("workspace.knowledgeRetrieved")).toBeInTheDocument();
  });

  test("Knowledge tab shows sources with citation index badges", () => {
    const answerMetrics: MessageMetrics = {
      sources: [
        { type: "web", label: "example.com", url: "https://example.com", index: 1 },
        { type: "rag", label: "internal-doc", url: null, index: 2 },
      ],
      rag_doc_count: 1,
    };
    renderWithPanel(<WorkspacePanel {...baseProps} answerMetrics={answerMetrics} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabKnowledge/ }));
    expect(screen.getByText("workspace.sourcesHeading")).toBeInTheDocument();
    // Citation index badges are aria-labelled; query by that to avoid matching the tab count badge.
    const citationBadges = screen.getAllByLabelText("workspace.sourceCitationLabel");
    expect(citationBadges).toHaveLength(2);
    expect(citationBadges[0]).toHaveTextContent("1");
    expect(citationBadges[1]).toHaveTextContent("2");
    expect(screen.getByText("example.com")).toBeInTheDocument();
    expect(screen.getByText("internal-doc")).toBeInTheDocument();
  });

  test("Knowledge tab falls back to positional citation numbers when index is absent", () => {
    const answerMetrics: MessageMetrics = {
      sources: [
        { type: "web", label: "site-a", url: "https://a.com" },
        { type: "web", label: "site-b", url: "https://b.com" },
      ],
    };
    renderWithPanel(<WorkspacePanel {...baseProps} answerMetrics={answerMetrics} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabKnowledge/ }));
    const citationBadges = screen.getAllByLabelText("workspace.sourceCitationLabel");
    expect(citationBadges).toHaveLength(2);
    expect(citationBadges[0]).toHaveTextContent("1");
    expect(citationBadges[1]).toHaveTextContent("2");
  });

  test("Knowledge tab shows Documents and Attachments in context sections", () => {
    const answerMetrics: MessageMetrics = {
      documents_used: [{ id: "d1", filename: "report.md", version: 2 }],
      attachments_used: [{ id: "a1", original_name: "budget.csv" }],
    };
    renderWithPanel(<WorkspacePanel {...baseProps} answerMetrics={answerMetrics} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabKnowledge/ }));
    expect(screen.getByText("workspace.documentsInContext")).toBeInTheDocument();
    expect(screen.getByText("report.md")).toBeInTheDocument();
    expect(screen.getByText("v2")).toBeInTheDocument();
    expect(screen.getByText("workspace.attachmentsInContext")).toBeInTheDocument();
    expect(screen.getByText("budget.csv")).toBeInTheDocument();
  });

  test("Knowledge tab remains in the empty state when only Outputs/Memory evidence is present", () => {
    const answerMetrics: MessageMetrics = {
      tools_executed: [{ name: "calculator_tool", status: "success" }],
    };
    renderWithPanel(<WorkspacePanel {...baseProps} answerMetrics={answerMetrics} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabKnowledge/ }));
    expect(screen.getByText("workspace.knowledgeEmpty")).toBeInTheDocument();
  });

  test("Inspector tab shows the node execution trace with a count badge", () => {
    const nodeTrace: NodeTraceEntry[] = [{ node: "respond", sequence: 1, durationMs: 100, status: "success" }];
    renderWithPanel(<WorkspacePanel {...baseProps} nodeTrace={nodeTrace} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabInspector/ }));
    // Count shows on the active tab.
    expect(screen.getByRole("tab", { name: /workspace\.tabInspector/ })).toHaveTextContent("1");
    expect(screen.getByText("Respond")).toBeInTheDocument();
  });

  test("shows the historical-answer banner on an evidence tab and Jump-to-latest works", () => {
    const onJumpToLatest = jest.fn();
    renderWithPanel(<WorkspacePanel {...baseProps} historicalAnswer onJumpToLatest={onJumpToLatest} />);
    // Default tab is Documents (localStorage cleared) — banner is suppressed there.
    expect(screen.queryByText("workspace.viewingEarlierAnswer")).not.toBeInTheDocument();
    // Switch to an evidence tab → banner appears.
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabKnowledge/ }));
    expect(screen.getByText("workspace.viewingEarlierAnswer")).toBeInTheDocument();
    fireEvent.click(screen.getByText("workspace.jumpToLatest"));
    expect(onJumpToLatest).toHaveBeenCalledTimes(1);
  });

  test("hides the historical banner when not viewing an earlier answer", () => {
    renderWithPanel(<WorkspacePanel {...baseProps} historicalAnswer={false} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabKnowledge/ }));
    expect(screen.queryByText("workspace.viewingEarlierAnswer")).not.toBeInTheDocument();
  });

  test("persists the active tab across remounts", () => {
    const { unmount } = renderWithPanel(<WorkspacePanel {...baseProps} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabMemory/ }));
    unmount();

    renderWithPanel(<WorkspacePanel {...baseProps} />);
    expect(screen.getByRole("tab", { name: /workspace\.tabMemory/ })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });
});

describe("EvidenceChips", () => {
  test("renders nothing when the answer produced no evidence", () => {
    const { container } = render(<EvidenceChips metrics={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });

  test("renders a chip per evidence dimension present", () => {
    render(
      <EvidenceChips
        metrics={{
          memories_used: [{ memory_id: "m1" }],
          tools_executed: [{ name: "web_search_mcp", status: "success" }],
        }}
      />,
    );
    const summary = screen.getByLabelText("workspace.evidenceSummary");
    expect(summary).toBeInTheDocument();
    const chips = summary.querySelectorAll('[data-slot="tooltip-trigger"]');
    expect(chips).toHaveLength(2);
  });

  test("chips invoke onSelect with the mapped workspace tab", () => {
    const onSelect = jest.fn();
    render(<EvidenceChips metrics={{ memories_used: [{ memory_id: "m1" }] }} onSelect={onSelect} />);
    fireEvent.click(screen.getAllByRole("button")[0]);
    expect(onSelect).toHaveBeenCalledWith("memory");
  });

  test("attachments chip routes to the Knowledge tab; docs chip to the Documents tab", () => {
    const onSelect = jest.fn();
    render(
      <EvidenceChips
        metrics={{
          attachments_used: [{ id: "a1", original_name: "budget.csv" }],
          documents_used: [{ id: "d1", filename: "report.md", version: 2 }],
        }}
        onSelect={onSelect}
      />,
    );
    const chips = screen.getAllByRole("button");
    fireEvent.click(chips[0]);
    expect(onSelect).toHaveBeenCalledWith("knowledge");
    fireEvent.click(chips[1]);
    expect(onSelect).toHaveBeenCalledWith("documents");
  });

  test("renders attachment-only answers (added to hasEvidence)", () => {
    render(<EvidenceChips metrics={{ attachments_used: [{ id: "a1", original_name: "x.csv" }] }} />);
    expect(screen.getByLabelText("workspace.evidenceSummary")).toBeInTheDocument();
  });

  test("renders static (non-interactive) chips when onSelect is omitted", () => {
    render(<EvidenceChips metrics={{ memories_used: [{ memory_id: "m1" }] }} />);
    expect(screen.getByLabelText("workspace.evidenceSummary")).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  test("no longer renders a separate map chip (the inline MapWidget is the artifact)", () => {
    render(
      <EvidenceChips
        metrics={{ map_data: { type: "location", zoom: 10, osm_url: "https://osm" } }}
        onSelect={jest.fn()}
      />,
    );
    expect(screen.queryByLabelText("workspace.evidenceSummary")).not.toBeInTheDocument();
  });
});

describe("InspectorTab", () => {
  test("shows an empty state when there is no trace", () => {
    render(<InspectorTab nodeTrace={[]} />);
    expect(screen.getByText("workspace.inspectorEmpty")).toBeInTheDocument();
  });

  test("renders the ordered node trace with humanized labels and timing", () => {
    const trace: NodeTraceEntry[] = [
      { node: "load_tools", sequence: 1, durationMs: 12, status: "success" },
      { node: "respond", sequence: 2, durationMs: 340, status: "success" },
    ];
    render(<InspectorTab nodeTrace={trace} />);
    expect(screen.getByText("Load tools")).toBeInTheDocument();
    expect(screen.getByText("Respond")).toBeInTheDocument();
    expect(screen.getByText("12 ms")).toBeInTheDocument();
    // Both bottom groups render: nodes that didn't run + nodes that run after.
    expect(screen.getByText("workspace.inspectorNotRun")).toBeInTheDocument();
    expect(screen.getByText("workspace.inspectorPostResponse")).toBeInTheDocument();
  });

  test("renders captured post-response nodes with timing, pending ones as badges", () => {
    const trace: NodeTraceEntry[] = [
      { node: "respond", sequence: 1, durationMs: 200, status: "success" },
      // Post-response node whose trace landed after the drain.
      { node: "summarize", sequence: 2, durationMs: 1800, status: "success" },
    ];
    render(<InspectorTab nodeTrace={trace} />);
    // The captured post-response node shows as a full row with its timing…
    expect(screen.getByText("Summarize")).toBeInTheDocument();
    expect(screen.getByText("1800 ms")).toBeInTheDocument();
    // …while the not-yet-captured post-response nodes stay as pending badges.
    expect(screen.getByText("Update memory")).toBeInTheDocument();
    expect(screen.getByText("workspace.inspectorPostResponse")).toBeInTheDocument();
  });

  test("collapses the mutually-exclusive tool-calling strategies into one slot", () => {
    const trace: NodeTraceEntry[] = [
      { node: "load_tools", sequence: 1, durationMs: 5, status: "success" },
      { node: "respond", sequence: 2, durationMs: 200, status: "success" },
    ];
    render(<InspectorTab nodeTrace={trace} />);
    expect(screen.getByText("Call tools")).toBeInTheDocument();
    expect(screen.queryByText("Call tools native")).not.toBeInTheDocument();
    expect(screen.queryByText("Call tools structured")).not.toBeInTheDocument();
  });

  test("conveys node error status to screen readers", () => {
    const trace: NodeTraceEntry[] = [{ node: "respond", sequence: 1, durationMs: 100, status: "error" }];
    render(<InspectorTab nodeTrace={trace} />);
    expect(screen.getByText("workspace.inspectorStatusError")).toBeInTheDocument();
  });

  test("makes a tool-calling node a deep-link only when onOpenOutputs is provided", () => {
    const trace: NodeTraceEntry[] = [{ node: "call_tools_native", sequence: 1, durationMs: 50, status: "success" }];
    const { rerender } = render(<InspectorTab nodeTrace={trace} />);
    // Without the callback the row is plain — no deep-link affordance.
    expect(screen.queryByRole("button", { name: /workspace\.viewToolResults/ })).not.toBeInTheDocument();
    const onOpenOutputs = jest.fn();
    rerender(<InspectorTab nodeTrace={trace} onOpenOutputs={onOpenOutputs} />);
    fireEvent.click(screen.getByRole("button", { name: /workspace\.viewToolResults/ }));
    expect(onOpenOutputs).toHaveBeenCalledTimes(1);
  });
});

describe("ActiveDocumentPane", () => {
  const renderPane = (documents: WorkspaceDocument[] = []) =>
    render(
      <ActiveDocumentProvider documents={documents}>
        <ActiveDocumentPane />
      </ActiveDocumentProvider>,
    );

  test("renders the pane region without crashing when no document is open", () => {
    renderPane();
    expect(
      screen.getByRole("region", { name: "workspace.splitViewTitle" }),
    ).toBeInTheDocument();
  });

  test("the close button is present and clickable", () => {
    renderPane();
    const closeBtn = screen.getByLabelText("workspace.closeSplitView");
    expect(closeBtn).toBeInTheDocument();
    // Clicking while no doc is open should not throw
    fireEvent.click(closeBtn);
  });
});

describe("ActiveDocumentPaneSlot", () => {
  test("renders nothing when no document is open for editing or history", () => {
    const { container } = render(
      <ActiveDocumentProvider documents={[]}>
        <ActiveDocumentPaneSlot />
      </ActiveDocumentProvider>,
    );
    // No editor + no history → the pane is not mounted at all (no region chrome).
    expect(container.firstChild).toBeNull();
    expect(
      screen.queryByRole("region", { name: "workspace.splitViewTitle" }),
    ).not.toBeInTheDocument();
  });
});

describe("DocumentEditorView version history", () => {
  function HistoryHarness({ docId }: { docId: string }) {
    const { loadHistory } = useActiveDocument();
    useEffect(() => {
      void loadHistory(docId);
    }, [docId, loadHistory]);
    return <DocumentEditorView />;
  }

  test("shows the current document version as a non-restorable 'Current' row", async () => {
    // The /versions endpoint only returns superseded snapshots — here, none — so
    // the current version must still appear, sourced from the document record.
    global.fetch = jest
      .fn()
      .mockResolvedValue({ ok: true, status: 200, json: async () => [] } as unknown as Response);
    const d: WorkspaceDocument = {
      id: "doc-9",
      filename: "notes.md",
      mime_type: "text/plain",
      size_bytes: 100,
      version: 3,
      updated_at: "2026-06-13T00:00:00Z",
    };

    render(
      <ActiveDocumentProvider documents={[d]}>
        <HistoryHarness docId="doc-9" />
      </ActiveDocumentProvider>,
    );

    expect(await screen.findByText("workspace.versionCurrent")).toBeInTheDocument();
    expect(screen.getByText("v3")).toBeInTheDocument();
    // The current version is not a restore target.
    expect(screen.queryByText("workspace.restore")).not.toBeInTheDocument();
    expect(screen.queryByText("workspace.noSavedVersions")).not.toBeInTheDocument();
  });
});
