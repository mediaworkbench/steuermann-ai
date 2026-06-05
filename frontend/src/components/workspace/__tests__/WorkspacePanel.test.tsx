import { render, screen, fireEvent } from "@testing-library/react";
import { WorkspacePanel } from "../WorkspacePanel";
import { WorkspaceSidebar } from "@/components/WorkspaceSidebar";
import { useI18n } from "@/hooks/useI18n";

jest.mock("@/hooks/useI18n");

const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;

const baseProps = {
  isOpen: true,
  onToggle: jest.fn(),
  documents: [],
};

beforeEach(() => {
  jest.clearAllMocks();
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
  test("renders the four workspace tabs with Documents active by default", () => {
    render(<WorkspacePanel {...baseProps} />);
    expect(screen.getAllByRole("tab")).toHaveLength(4);
    // Documents tab is active → its empty state shows.
    expect(screen.getByText("workspace.noDocuments")).toBeInTheDocument();
  });

  test("switches to the Knowledge evidence tab", () => {
    render(<WorkspacePanel {...baseProps} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabKnowledge/ }));
    expect(screen.getByText("workspace.knowledgeEmpty")).toBeInTheDocument();
  });

  test("switches to the Memory and Outputs evidence tabs", () => {
    render(<WorkspacePanel {...baseProps} />);
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabMemory/ }));
    expect(screen.getByText("workspace.memoryEmpty")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: /workspace\.tabOutputs/ }));
    expect(screen.getByText("workspace.outputsEmpty")).toBeInTheDocument();
  });

  test("compat wrapper WorkspaceSidebar renders the modular panel", () => {
    render(<WorkspaceSidebar {...baseProps} />);
    expect(screen.getAllByRole("tab")).toHaveLength(4);
    expect(screen.getByText("workspace.noDocuments")).toBeInTheDocument();
  });
});
