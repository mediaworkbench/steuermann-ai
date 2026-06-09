import { render, screen, fireEvent } from "@testing-library/react";
import { axe } from "jest-axe";
import { WorkspaceEvidenceTabs } from "../WorkspaceEvidenceTabs";
import { ConversationEvidenceDrawer } from "../ConversationEvidenceDrawer";
import { fetchConversation } from "@/lib/api";
import { useI18n } from "@/hooks/useI18n";
import type { MessageMetrics, NodeTraceEntry } from "@/lib/types";

jest.mock("@/hooks/useI18n");
jest.mock("@/lib/api", () => ({ fetchConversation: jest.fn() }));

const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;
const mockFetch = fetchConversation as jest.MockedFunction<typeof fetchConversation>;

beforeEach(() => {
  jest.clearAllMocks();
  mockUseI18n.mockReturnValue({
    locale: "en",
    setLocale: jest.fn(),
    t: (key: string) => key,
    formatDate: () => "",
    formatTime: () => "",
    formatDateTime: () => "",
    formatNumber: (value: number) => String(value),
    formatRelativeTime: () => "",
  });
});

describe("WorkspaceEvidenceTabs", () => {
  test("renders four tabs with Knowledge active by default", () => {
    render(<WorkspaceEvidenceTabs metrics={null} />);
    expect(screen.getAllByRole("tab")).toHaveLength(4);
    expect(screen.getByText("workspace.knowledgeEmpty")).toBeInTheDocument();
  });

  test("switches between the evidence tabs", () => {
    render(<WorkspaceEvidenceTabs metrics={null} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabMemory/ }));
    expect(screen.getByText("workspace.memoryEmpty")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabOutputs/ }));
    expect(screen.getByText("workspace.outputsEmpty")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabInspector/ }));
    expect(screen.getByText("workspace.inspectorEmpty")).toBeInTheDocument();
  });

  test("shows a count badge and content for the active tab", () => {
    const metrics: MessageMetrics = { memories_used: [{ memory_id: "m1", text: "likes tea" }] };
    render(<WorkspaceEvidenceTabs metrics={metrics} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabMemory/ }));
    expect(screen.getByRole("tab", { name: /workspace\.tabMemory/ })).toHaveTextContent("1");
    expect(screen.getByText("likes tea")).toBeInTheDocument();
  });

  test("has no accessibility violations", async () => {
    const nodeTrace: NodeTraceEntry[] = [{ node: "respond", sequence: 1, durationMs: 50, status: "success" }];
    const { container } = render(<WorkspaceEvidenceTabs metrics={null} nodeTrace={nodeTrace} />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("ConversationEvidenceDrawer", () => {
  const assistantDetail = {
    conversation: { id: "c1", title: "Tax review" },
    messages: [
      { id: 1, conversation_id: "c1", role: "user", content: "hi", metadata: {}, created_at: null },
      {
        id: 2,
        conversation_id: "c1",
        role: "assistant",
        content: "answer",
        metadata: {
          memories_used: [{ memory_id: "m1", text: "likes tea" }],
          node_trace: [{ node: "respond", sequence: 1, duration_ms: 100, status: "success" }],
        },
        created_at: null,
      },
    ],
  };

  test("loads the conversation and renders the evidence tabs for the latest answer", async () => {
    mockFetch.mockResolvedValue(assistantDetail as any);
    render(<ConversationEvidenceDrawer conversationId="c1" title="Tax review" onClose={jest.fn()} />);

    expect(await screen.findByText("chats.evidenceLatestHint")).toBeInTheDocument();
    expect(screen.getAllByRole("tab")).toHaveLength(4);
    expect(mockFetch).toHaveBeenCalledWith("c1");
  });

  test("shows the empty state when there is no assistant answer", async () => {
    mockFetch.mockResolvedValue({
      conversation: { id: "c2", title: "Empty" },
      messages: [{ id: 1, conversation_id: "c2", role: "user", content: "hi", metadata: {}, created_at: null }],
    } as any);
    render(<ConversationEvidenceDrawer conversationId="c2" title="Empty" onClose={jest.fn()} />);
    expect(await screen.findByText("chats.evidenceEmpty")).toBeInTheDocument();
  });

  test("shows an error state with a retry that refetches", async () => {
    mockFetch.mockResolvedValueOnce(null);
    render(<ConversationEvidenceDrawer conversationId="c3" title="Boom" onClose={jest.fn()} />);
    expect(await screen.findByText("chats.evidenceError")).toBeInTheDocument();

    mockFetch.mockResolvedValueOnce(assistantDetail as any);
    fireEvent.click(screen.getByText("workspace.retry"));
    expect(await screen.findByText("chats.evidenceLatestHint")).toBeInTheDocument();
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  test("the close control calls onClose", async () => {
    mockFetch.mockResolvedValue(assistantDetail as any);
    const onClose = jest.fn();
    render(<ConversationEvidenceDrawer conversationId="c1" title="Tax review" onClose={onClose} />);
    await screen.findByText("chats.evidenceLatestHint");
    fireEvent.click(screen.getByLabelText("chats.closeEvidence"));
    expect(onClose).toHaveBeenCalled();
  });
});
