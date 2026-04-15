import { selectActiveAttachmentIds } from "@/lib/attachments";

describe("selectActiveAttachmentIds", () => {
  it("returns all attachment ids in order", () => {
    const selected = selectActiveAttachmentIds([
      { id: "att-1" },
      { id: "att-2" },
      { id: "att-3" },
    ]);

    expect(selected).toEqual(["att-1", "att-2", "att-3"]);
  });

  it("deduplicates duplicate ids while preserving first-seen order", () => {
    const selected = selectActiveAttachmentIds([
      { id: "att-1" },
      { id: "att-1" },
      { id: "att-2" },
    ]);

    expect(selected).toEqual(["att-1", "att-2"]);
  });

  it("resets selection when switching conversations", () => {
    const conversationASelection = selectActiveAttachmentIds([
      { id: "a-1" },
      { id: "a-2" },
    ]);

    const conversationBSelection = selectActiveAttachmentIds([
      { id: "b-1" },
    ]);

    expect(conversationASelection).toEqual(["a-1", "a-2"]);
    expect(conversationBSelection).toEqual(["b-1"]);
  });
});