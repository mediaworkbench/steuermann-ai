import { pickFocusedAnswer } from "@/lib/panelAnswer";
import type { Message } from "@/lib/types";

const user = (content: string): Message => ({ role: "user", content });
const assistant = (content: string, extra: Partial<Message> = {}): Message => ({
  role: "assistant",
  content,
  ...extra,
});

describe("pickFocusedAnswer", () => {
  // messages: [user, assistant#1, user, assistant#2(latest)]
  const messages: Message[] = [
    user("q1"),
    assistant("a1", { metrics: { output_tokens: 10 }, nodeTrace: [{ node: "respond", sequence: 1, durationMs: 5, status: "success" }] }),
    user("q2"),
    assistant("a2", { metrics: { output_tokens: 20 } }),
  ];
  const lastAssistantIndex = 3;

  test("null focus → follow latest", () => {
    expect(pickFocusedAnswer(messages, null, lastAssistantIndex)).toEqual({
      isHistorical: false,
      metrics: null,
      nodeTrace: [],
    });
  });

  test("focus equal to the latest → follow latest", () => {
    expect(pickFocusedAnswer(messages, lastAssistantIndex, lastAssistantIndex).isHistorical).toBe(false);
  });

  test("out-of-bounds index → follow latest (safe fallback)", () => {
    expect(pickFocusedAnswer(messages, 99, lastAssistantIndex).isHistorical).toBe(false);
  });

  test("non-assistant index → follow latest", () => {
    // index 0 is a user message
    expect(pickFocusedAnswer(messages, 0, lastAssistantIndex).isHistorical).toBe(false);
  });

  test("valid earlier assistant → historical with its metrics + trace", () => {
    const res = pickFocusedAnswer(messages, 1, lastAssistantIndex);
    expect(res.isHistorical).toBe(true);
    expect(res.metrics).toEqual({ output_tokens: 10 });
    expect(res.nodeTrace).toHaveLength(1);
  });

  test("earlier assistant without metrics/trace → historical with null/empty", () => {
    const msgs: Message[] = [assistant("a0"), user("q"), assistant("a-latest")];
    const res = pickFocusedAnswer(msgs, 0, 2);
    expect(res).toEqual({ isHistorical: true, metrics: null, nodeTrace: [] });
  });
});
