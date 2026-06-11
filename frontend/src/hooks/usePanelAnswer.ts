"use client";

import { useState, useCallback, useMemo, useEffect } from "react";
import { useEdgeEffect } from "@/hooks/useEdgeEffect";
import { pickFocusedAnswer } from "@/lib/panelAnswer";
import { useWorkspacePanel } from "@/context/WorkspacePanelContext";
import { useConversationContext } from "@/components/LayoutShell";
import type { Message, MessageMetrics, NodeTraceEntry } from "@/lib/types";
import type { WorkspaceTabId } from "@/components/workspace/types";

interface UsePanelAnswerResult {
  focusedAnswerIndex: number | null;
  lastAssistantIndex: number;
  panelMetrics: MessageMetrics | null;
  panelNodeTrace: NodeTraceEntry[];
  panelIsStreaming: boolean;
  isHistoricalAnswer: boolean;
  handleSelectEvidence: (tab: WorkspaceTabId, index: number) => void;
  handleJumpToLatest: () => void;
}

export function usePanelAnswer(
  messages: Message[],
  isStreaming: boolean,
  nodeTrace: NodeTraceEntry[],
): UsePanelAnswerResult {
  const { activeId, setWorkspaceSidebarOpen } = useConversationContext();
  const { setActiveTab: setWorkspaceTab } = useWorkspacePanel();

  const [focusedAnswerIndex, setFocusedAnswerIndex] = useState<number | null>(null);

  // Reset focus on conversation switch
  useEffect(() => setFocusedAnswerIndex(null), [activeId]);

  // Reset focus at the start of every new turn (rising edge of isStreaming)
  useEdgeEffect(isStreaming, { onRising: () => setFocusedAnswerIndex(null) });

  const lastAssistantIndex = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") return i;
    }
    return -1;
  }, [messages]);

  const latestAnswerMetrics = useMemo<MessageMetrics | null>(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === "assistant" && m.metrics) return m.metrics;
    }
    return null;
  }, [messages]);

  const committedNodeTrace = useMemo<NodeTraceEntry[]>(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === "assistant" && m.nodeTrace && m.nodeTrace.length > 0) return m.nodeTrace;
    }
    return [];
  }, [messages]);

  const inspectorNodeTrace = isStreaming
    ? nodeTrace
    : committedNodeTrace.length > 0
      ? committedNodeTrace
      : nodeTrace;

  const focused = pickFocusedAnswer(messages, focusedAnswerIndex, lastAssistantIndex);
  const panelMetrics = focused.isHistorical ? focused.metrics : latestAnswerMetrics;
  const panelNodeTrace = focused.isHistorical ? focused.nodeTrace : inspectorNodeTrace;
  const panelIsStreaming = isStreaming && !focused.isHistorical;

  const handleSelectEvidence = useCallback(
    (tab: WorkspaceTabId, index: number) => {
      setWorkspaceTab(tab);
      setWorkspaceSidebarOpen(true);
      if (tab !== "documents") {
        setFocusedAnswerIndex(index === lastAssistantIndex ? null : index);
      }
    },
    [setWorkspaceTab, setWorkspaceSidebarOpen, lastAssistantIndex],
  );

  const handleJumpToLatest = useCallback(() => setFocusedAnswerIndex(null), []);

  return {
    focusedAnswerIndex,
    lastAssistantIndex,
    panelMetrics,
    panelNodeTrace,
    panelIsStreaming,
    isHistoricalAnswer: focused.isHistorical,
    handleSelectEvidence,
    handleJumpToLatest,
  };
}
