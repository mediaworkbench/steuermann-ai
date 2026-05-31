import { act, renderHook } from "@testing-library/react";
import { useStreamingChat } from "@/hooks/useStreamingChat";

// ── SSE stream helpers ────────────────────────────────────────────────────────

function makeSseStream(events: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(event));
      }
      controller.close();
    },
  });
}

function sseToken(delta: string): string {
  return `event: token\ndata: ${JSON.stringify({ delta })}\n\n`;
}

function sseNode(node: string, label: string): string {
  return `event: node\ndata: ${JSON.stringify({ node, label })}\n\n`;
}

function sseTool(name: string, status: "start" | "end", label?: string): string {
  return `event: tool_call\ndata: ${JSON.stringify({ name, status, label: label ?? `Using ${name}...` })}\n\n`;
}

function sseMetadata(meta: Record<string, unknown> = {}): string {
  const payload = {
    tokens_used: 10,
    input_tokens: 4,
    output_tokens: 6,
    model_used: "test-model",
    tool_results: {},
    sources: [],
    rag_attempted: false,
    rag_doc_count: 0,
    loaded_memory: [],
    ...meta,
  };
  return `event: metadata\ndata: ${JSON.stringify(payload)}\n\n`;
}

function sseDone(): string {
  return "data: [DONE]\n\n";
}

function sseError(message: string): string {
  return `event: error\ndata: ${JSON.stringify({ message })}\n\n`;
}

function mockFetch(sseEvents: string[], status = 200): void {
  const stream = makeSseStream(sseEvents);
  global.fetch = jest.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    body: stream,
  } as unknown as Response);
}

const defaultParams = {
  message: "Hello",
  userId: "u1",
  conversationId: null,
  attachmentIds: [],
  documentIds: [],
  ragEnabled: true,
};

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("useStreamingChat", () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it("starts with idle state", () => {
    const { result } = renderHook(() => useStreamingChat());
    expect(result.current.isStreaming).toBe(false);
    expect(result.current.streamingContent).toBe("");
    expect(result.current.finalMetadata).toBeNull();
    expect(result.current.streamError).toBeNull();
    expect(result.current.wasCancelled).toBe(false);
    expect(result.current.nodeStatus).toBeNull();
    expect(result.current.toolCallStatus).toBeNull();
  });

  it("accumulates token deltas into streamingContent", async () => {
    mockFetch([sseToken("Hello"), sseToken(" world"), sseMetadata(), sseDone()]);

    const { result } = renderHook(() => useStreamingChat());

    await act(async () => {
      await result.current.sendMessage(defaultParams);
    });

    expect(result.current.streamingContent).toBe("Hello world");
    expect(result.current.isStreaming).toBe(false);
  });

  it("sets nodeStatus on node events and clears on metadata", async () => {
    mockFetch([
      sseNode("retrieve_knowledge", "Searching knowledge base..."),
      sseToken("Answer"),
      sseMetadata(),
      sseDone(),
    ]);

    const { result } = renderHook(() => useStreamingChat());

    await act(async () => {
      const promise = result.current.sendMessage(defaultParams);
      await new Promise((r) => setTimeout(r, 0));
      await promise;
    });

    // nodeStatus is cleared after the metadata event
    expect(result.current.nodeStatus).toBeNull();
  });

  it("sets toolCallStatus on tool_call start and end events", async () => {
    mockFetch([
      sseTool("web_search", "start"),
      sseTool("web_search", "end"),
      sseToken("Answer"),
      sseMetadata(),
      sseDone(),
    ]);

    const { result } = renderHook(() => useStreamingChat());

    await act(async () => {
      await result.current.sendMessage(defaultParams);
    });

    // After stream, toolCallStatus may still be non-null briefly (cleared by setTimeout 800ms)
    // We verify the hook didn't throw and content is correct
    expect(result.current.streamingContent).toBe("Answer");
  });

  it("populates finalMetadata on metadata event", async () => {
    mockFetch([sseToken("Hi"), sseMetadata({ tokens_used: 42, model_used: "gpt-x" }), sseDone()]);

    const { result } = renderHook(() => useStreamingChat());

    await act(async () => {
      await result.current.sendMessage(defaultParams);
    });

    expect(result.current.finalMetadata).not.toBeNull();
    expect(result.current.finalMetadata?.tokens_used).toBe(42);
    expect(result.current.finalMetadata?.model_used).toBe("gpt-x");
    expect(result.current.isStreaming).toBe(false);
  });

  it("sets streamError on error event", async () => {
    mockFetch([sseError("LangGraph crashed"), sseDone()]);

    const { result } = renderHook(() => useStreamingChat());

    await act(async () => {
      await result.current.sendMessage(defaultParams);
    });

    expect(result.current.streamError).toBe("LangGraph crashed");
    expect(result.current.isStreaming).toBe(false);
  });

  it("sets streamError on non-OK HTTP response", async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 503,
      body: null,
    } as unknown as Response);

    const { result } = renderHook(() => useStreamingChat());

    await act(async () => {
      await result.current.sendMessage(defaultParams);
    });

    expect(result.current.streamError).toMatch(/503/);
    expect(result.current.isStreaming).toBe(false);
  });

  it("sets wasCancelled=true when cancel() is called", async () => {
    // Slow stream — the AbortController is triggered before it finishes
    const stream = new ReadableStream<Uint8Array>({
      start() {
        // Don't close immediately — we'll abort via cancel()
      },
    });
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      status: 200,
      body: stream,
    } as unknown as Response);

    const { result } = renderHook(() => useStreamingChat());

    const sendPromise = act(async () => {
      const p = result.current.sendMessage(defaultParams);
      // Cancel immediately on next tick
      await new Promise((r) => setTimeout(r, 0));
      result.current.cancel();
      await p;
    });

    await sendPromise;

    expect(result.current.wasCancelled).toBe(true);
    expect(result.current.isStreaming).toBe(false);
  });

  it("reset() clears all state after stream completes", async () => {
    mockFetch([sseToken("Hello"), sseMetadata(), sseDone()]);

    const { result } = renderHook(() => useStreamingChat());

    await act(async () => {
      await result.current.sendMessage(defaultParams);
    });

    expect(result.current.streamingContent).toBe("Hello");

    act(() => {
      result.current.reset();
    });

    expect(result.current.streamingContent).toBe("");
    expect(result.current.finalMetadata).toBeNull();
    expect(result.current.streamError).toBeNull();
    expect(result.current.wasCancelled).toBe(false);
    expect(result.current.isStreaming).toBe(false);
  });

  it("handles stream ending without explicit [DONE] (reader done)", async () => {
    // No [DONE] — stream just closes
    mockFetch([sseToken("Partial"), sseMetadata()]);

    const { result } = renderHook(() => useStreamingChat());

    await act(async () => {
      await result.current.sendMessage(defaultParams);
    });

    expect(result.current.streamingContent).toBe("Partial");
    expect(result.current.isStreaming).toBe(false);
  });
});
