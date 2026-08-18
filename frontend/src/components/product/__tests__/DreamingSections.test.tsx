import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { axe } from "jest-axe";
import { useI18n } from "@/hooks/useI18n";
import {
  fetchDreamConflicts,
  resolveDreamConflict,
  fetchDreamProcedural,
  decideDreamRule,
  fetchDreamAudit,
  undoDreamAction,
  fetchDreamingMetrics,
} from "@/lib/api";
import { DissonanceQueueSection } from "@/components/product/DissonanceQueueSection";
import { ProceduralApprovalsSection } from "@/components/product/ProceduralApprovalsSection";
import { MemoryUndoSection } from "@/components/product/MemoryUndoSection";
import { DreamingMetricsSection } from "@/components/product/DreamingMetricsSection";

jest.mock("@/hooks/useI18n");
jest.mock("sonner", () => ({ toast: { success: jest.fn(), error: jest.fn() } }));
jest.mock("@/lib/api", () => ({
  fetchDreamConflicts: jest.fn(),
  resolveDreamConflict: jest.fn(),
  fetchDreamProcedural: jest.fn(),
  decideDreamRule: jest.fn(),
  fetchDreamAudit: jest.fn(),
  undoDreamAction: jest.fn(),
  fetchDreamingMetrics: jest.fn(),
}));

const mockUseI18n = useI18n as jest.MockedFunction<typeof useI18n>;
const m = {
  conflicts: fetchDreamConflicts as jest.Mock,
  resolve: resolveDreamConflict as jest.Mock,
  procedural: fetchDreamProcedural as jest.Mock,
  decide: decideDreamRule as jest.Mock,
  audit: fetchDreamAudit as jest.Mock,
  undo: undoDreamAction as jest.Mock,
  metrics: fetchDreamingMetrics as jest.Mock,
};

beforeEach(() => {
  jest.clearAllMocks();
  mockUseI18n.mockReturnValue({
    locale: "en",
    setLocale: jest.fn(),
    t: (key: string) => key,
    formatDate: () => "",
    formatTime: () => "",
    formatDateTime: () => "",
    formatNumber: (v: number) => String(v),
    formatRelativeTime: () => "1m ago",
  } as unknown as ReturnType<typeof useI18n>);
});

// --------------------------------------------------------------------------- //
// Dissonance queue
// --------------------------------------------------------------------------- //
describe("DissonanceQueueSection", () => {
  const CONFLICT = {
    conflict_id: "c1",
    semantic_text: "lives in Munich",
    contradiction_text: "moved to Berlin",
    prior_confidence: 0.6,
    new_confidence: 0.3,
    choices: [],
    created_at: null,
  };

  it("renders open conflicts and resolves one", async () => {
    m.conflicts.mockResolvedValue([CONFLICT]);
    m.resolve.mockResolvedValue(undefined);
    render(<DissonanceQueueSection />);

    expect(await screen.findByText("lives in Munich")).toBeInTheDocument();
    expect(screen.getByText("moved to Berlin")).toBeInTheDocument();

    fireEvent.click(screen.getByText("dreaming.acceptNew"));
    await waitFor(() => expect(m.resolve).toHaveBeenCalledWith("c1", "accept_new"));
    await waitFor(() => expect(screen.queryByText("lives in Munich")).not.toBeInTheDocument());
  });

  it("shows the empty state and has no a11y violations", async () => {
    m.conflicts.mockResolvedValue([]);
    const { container } = render(<DissonanceQueueSection />);
    expect(await screen.findByText("dreaming.conflictsEmpty")).toBeInTheDocument();
    expect(await axe(container)).toHaveNoViolations();
  });
});

// --------------------------------------------------------------------------- //
// Procedural approvals
// --------------------------------------------------------------------------- //
describe("ProceduralApprovalsSection", () => {
  const PROPOSED = {
    rule_key: "format.bullets",
    rule_text: "Use bullet lists",
    tier: 1,
    status: "proposed",
    confidence: 0,
    evidence: {},
    updated_at: null,
  };

  it("approves a proposed rule and always shows the Tier-3 lock note", async () => {
    m.procedural.mockResolvedValue([PROPOSED]);
    m.decide.mockResolvedValue(undefined);
    render(<ProceduralApprovalsSection />);

    expect(await screen.findByText("Use bullet lists")).toBeInTheDocument();
    expect(screen.getByText("dreaming.tier3Locked")).toBeInTheDocument();

    fireEvent.click(screen.getByText("dreaming.approve"));
    await waitFor(() => expect(m.decide).toHaveBeenCalledWith("format.bullets", "approve"));
  });

  it("active rules show no approve/reject controls; no a11y violations", async () => {
    m.procedural.mockResolvedValue([{ ...PROPOSED, status: "active" }]);
    const { container } = render(<ProceduralApprovalsSection />);
    expect(await screen.findByText("Use bullet lists")).toBeInTheDocument();
    expect(screen.queryByText("dreaming.approve")).not.toBeInTheDocument();
    expect(await axe(container)).toHaveNoViolations();
  });
});

// --------------------------------------------------------------------------- //
// Undo feed
// --------------------------------------------------------------------------- //
describe("MemoryUndoSection", () => {
  const ITEM = {
    audit_id: "a1",
    cycle: "forgetting",
    action: "delete",
    target_kind: "episodic",
    target_id: "m1",
    before_state: { text: "old memory" },
    after_state: null,
    created_at: null,
    reversible_until: new Date(Date.now() + 5 * 86_400_000).toISOString(),
  };

  it("undoes an action and removes it from the feed", async () => {
    m.audit.mockResolvedValue([ITEM]);
    m.undo.mockResolvedValue(undefined);
    render(<MemoryUndoSection />);

    expect(await screen.findByText("dreaming.actionDelete")).toBeInTheDocument();
    fireEvent.click(screen.getByText("dreaming.undoButton"));
    await waitFor(() => expect(m.undo).toHaveBeenCalledWith("a1"));
    await waitFor(() => expect(screen.queryByText("dreaming.actionDelete")).not.toBeInTheDocument());
  });

  it("renders the empty state with no a11y violations", async () => {
    m.audit.mockResolvedValue([]);
    const { container } = render(<MemoryUndoSection />);
    expect(await screen.findByText("dreaming.undoEmpty")).toBeInTheDocument();
    expect(await axe(container)).toHaveNoViolations();
  });
});

// --------------------------------------------------------------------------- //
// Admin metrics
// --------------------------------------------------------------------------- //
describe("DreamingMetricsSection", () => {
  const METRICS = {
    cycles_run: 12,
    vector_count: 345,
    total_tokens: 6789,
    avg_tokens_per_cycle: 565.8,
    pending_resolutions: { open_conflicts: 2, proposed_procedural: 1, total: 3 },
    deletion_count: 4,
    promotion_count: 1,
    run_status: { ok: 11, partial: 1 },
  };

  it("renders aggregate cards from metrics", async () => {
    m.metrics.mockResolvedValue(METRICS);
    const { container } = render(<DreamingMetricsSection />);

    expect(await screen.findByText("12")).toBeInTheDocument(); // cycles_run
    expect(screen.getByText("345")).toBeInTheDocument(); // vector_count
    expect(screen.getByText("dreaming.cyclesRun")).toBeInTheDocument();
    expect(await axe(container)).toHaveNoViolations();
  });

  it("shows an error state when metrics fail to load", async () => {
    m.metrics.mockResolvedValue(null);
    render(<DreamingMetricsSection />);
    expect(await screen.findByText("dreaming.loadError")).toBeInTheDocument();
  });
});
