import type { ReactElement } from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { WorkspacePanel } from "../WorkspacePanel";
import { EvidenceChips } from "../EvidenceChips";
import { InspectorTab } from "../InspectorTab";
import { WorkspaceSidebar } from "@/components/WorkspaceSidebar";
import { WorkspacePanelProvider } from "@/context/WorkspacePanelContext";
import { useI18n } from "@/hooks/useI18n";
import type { WorkspaceDocument } from "../types";
import type { MessageMetrics, NodeTraceEntry } from "@/lib/types";

jest.mock("@/hooks/useI18n");

const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;

const doc = (id: string, filename: string): WorkspaceDocument => ({
  id,
  filename,
  mime_type: "text/plain",
  size_bytes: 100,
  version: 1,
});

const baseProps = {
  isOpen: true,
  onToggle: jest.fn(),
  documents: [],
};

// WorkspacePanel/DocumentsTab read internal view state from WorkspacePanelContext.
const renderWithPanel = (ui: ReactElement) => render(<WorkspacePanelProvider>{ui}</WorkspacePanelProvider>);

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

  test("shows the no-results state when search matches nothing", () => {
    renderWithPanel(<WorkspacePanel {...baseProps} documents={[doc("1", "alpha.md")]} />);
    fireEvent.change(screen.getByLabelText("workspace.searchDocuments"), {
      target: { value: "zzz" },
    });
    expect(screen.getByText("workspace.noResults")).toBeInTheDocument();
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
    expect(screen.getByRole("tab", { name: /workspace\.tabMemory/ })).toHaveTextContent("1");
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabMemory/ }));
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

  test("Knowledge tab shows the RAG retrieval summary", () => {
    const answerMetrics: MessageMetrics = { rag_attempted: true, rag_doc_count: 3 };
    renderWithPanel(<WorkspacePanel {...baseProps} answerMetrics={answerMetrics} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabKnowledge/ }));
    expect(screen.getByText("workspace.knowledgeRetrieved")).toBeInTheDocument();
  });

  test("Inspector tab shows the node execution trace with a count badge", () => {
    const nodeTrace: NodeTraceEntry[] = [{ node: "respond", sequence: 1, durationMs: 100, status: "success" }];
    renderWithPanel(<WorkspacePanel {...baseProps} nodeTrace={nodeTrace} />);
    expect(screen.getByRole("tab", { name: /workspace\.tabInspector/ })).toHaveTextContent("1");
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabInspector/ }));
    expect(screen.getByText("Respond")).toBeInTheDocument();
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
    expect(screen.getByTitle("workspace.evidenceMemory")).toBeInTheDocument();
    expect(screen.getByTitle("workspace.evidenceTools")).toBeInTheDocument();
  });

  test("chips invoke onSelect with the mapped workspace tab", () => {
    const onSelect = jest.fn();
    render(<EvidenceChips metrics={{ memories_used: [{ memory_id: "m1" }] }} onSelect={onSelect} />);
    fireEvent.click(screen.getByTitle("workspace.evidenceMemory"));
    expect(onSelect).toHaveBeenCalledWith("memory");
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
});
