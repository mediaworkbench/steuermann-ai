import { looksLikeWriteback, splitWritebackStream } from "@/lib/writeback";

describe("looksLikeWriteback", () => {
  it("returns true for SUMMARY: prefix", () => {
    expect(looksLikeWriteback("SUMMARY:\nImproved.")).toBe(true);
  });

  it("returns true with leading whitespace", () => {
    expect(looksLikeWriteback("  SUMMARY:\nImproved.")).toBe(true);
  });

  it("returns false for plain text", () => {
    expect(looksLikeWriteback("Hello world")).toBe(false);
  });

  it("returns false for empty string", () => {
    expect(looksLikeWriteback("")).toBe(false);
  });

  it("is case-sensitive (lowercase 'summary:' is not a match)", () => {
    expect(looksLikeWriteback("summary:\nLowercase")).toBe(false);
  });
});

describe("splitWritebackStream", () => {
  it("returns empty summary when no SUMMARY: header present", () => {
    const result = splitWritebackStream("Hello world");
    expect(result.summary).toBe("");
    expect(result.inDocument).toBe(false);
  });

  it("returns empty for empty string", () => {
    const result = splitWritebackStream("");
    expect(result.summary).toBe("");
    expect(result.inDocument).toBe(false);
  });

  it("extracts summary before double-newline DOCUMENT: marker", () => {
    const result = splitWritebackStream("SUMMARY:\nImproved clarity.\n\nDOCUMENT:\nNew content here.");
    expect(result.summary).toBe("Improved clarity.");
    expect(result.inDocument).toBe(true);
  });

  it("extracts summary before single-newline DOCUMENT: marker (tolerant)", () => {
    const result = splitWritebackStream("SUMMARY:\nGood summary.\nDOCUMENT:\nContent");
    expect(result.summary).toBe("Good summary.");
    expect(result.inDocument).toBe(true);
  });

  it("handles multiple newlines before DOCUMENT:", () => {
    const result = splitWritebackStream("SUMMARY:\nMultiple newlines.\n\n\nDOCUMENT:\nContent");
    expect(result.summary).toBe("Multiple newlines.");
    expect(result.inDocument).toBe(true);
  });

  it("returns partial summary when DOCUMENT: not yet seen (streaming mid-flight)", () => {
    const result = splitWritebackStream("SUMMARY:\nImproved clar");
    expect(result.summary).toBe("Improved clar");
    expect(result.inDocument).toBe(false);
  });

  it("returns empty summary if SUMMARY: is present but no content yet", () => {
    const result = splitWritebackStream("SUMMARY:\n");
    expect(result.summary).toBe("");
    expect(result.inDocument).toBe(false);
  });

  it("does not leak DOCUMENT: body into summary", () => {
    const result = splitWritebackStream("SUMMARY:\nThe summary.\n\nDOCUMENT:\nLine one\nLine two");
    expect(result.summary).toBe("The summary.");
    expect(result.inDocument).toBe(true);
  });

  it("trims whitespace from the extracted summary", () => {
    const result = splitWritebackStream("SUMMARY:\n  Trimmed summary.  \n\nDOCUMENT:\nContent");
    expect(result.summary).toBe("Trimmed summary.");
  });
});
