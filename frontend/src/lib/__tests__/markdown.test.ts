import { normalizeMath, linkFootnotes } from "../markdown";
import type { Source } from "../types";

describe("normalizeMath", () => {
  describe("currency safety (finance/tax assistant)", () => {
    it("escapes single-dollar currency amounts so they are not parsed as math", () => {
      expect(normalizeMath("costs $5 to $10")).toBe("costs \\$5 to \\$10");
      expect(normalizeMath("$1,000.50 total")).toBe("\\$1,000.50 total");
      expect(normalizeMath("about $5K saved")).toBe("about \\$5K saved");
    });

    it("escapes a currency amount at end of string", () => {
      expect(normalizeMath("pay $42")).toBe("pay \\$42");
    });

    it("does not escape a dollar that is not followed by a number", () => {
      expect(normalizeMath("the $ sign")).toBe("the $ sign");
    });
  });

  describe("math is preserved / normalized", () => {
    it("leaves single-dollar inline math intact (delimiter not followed by a digit)", () => {
      expect(normalizeMath("$x^2$")).toBe("$x^2$");
      expect(normalizeMath("Euler: $e^{i\\pi}+1=0$")).toBe("Euler: $e^{i\\pi}+1=0$");
    });

    it("leaves double-dollar block math intact", () => {
      expect(normalizeMath("$$\\int_0^1 x^2 dx$$")).toBe("$$\\int_0^1 x^2 dx$$");
    });

    it("normalizes \\( \\) inline LaTeX to single dollars", () => {
      expect(normalizeMath("\\(a^2+b^2\\)")).toBe("$a^2+b^2$");
    });

    it("normalizes \\[ \\] block LaTeX to double dollars", () => {
      expect(normalizeMath("\\[ \\sum_{i=1}^n i \\]")).toBe("$$ \\sum_{i=1}^n i $$");
    });
  });

  describe("code is protected", () => {
    it("does not touch a dollar inside a fenced code block", () => {
      const input = "```bash\necho $5\n```";
      expect(normalizeMath(input)).toBe(input);
    });

    it("does not touch \\( inside inline code", () => {
      const input = "use `\\(x\\)` literally";
      expect(normalizeMath(input)).toBe(input);
    });

    it("still transforms math outside a code block in the same string", () => {
      const input = "before `$5` and after $5 done";
      // inline code segment untouched; the bare $5 outside is escaped
      expect(normalizeMath(input)).toBe("before `$5` and after \\$5 done");
    });
  });
});

describe("linkFootnotes", () => {
  const sources: Source[] = [
    { index: 1, label: "Example", type: "web", url: "https://example.com" },
    { index: 2, label: "KB Doc", type: "rag", url: null },
  ];

  it("returns text unchanged when there are no sources", () => {
    expect(linkFootnotes("see [1]", undefined)).toBe("see [1]");
    expect(linkFootnotes("see [1]", [])).toBe("see [1]");
  });

  it("turns a web citation into a superscript markdown link", () => {
    expect(linkFootnotes("see [1]", sources)).toBe("see [¹](https://example.com)");
  });

  it("turns a RAG citation into bold superscript (no link)", () => {
    expect(linkFootnotes("see [2]", sources)).toBe("see **[²]**");
  });

  it("handles grouped citations", () => {
    expect(linkFootnotes("see [1, 2]", sources)).toBe(
      "see [¹](https://example.com) **[²]**",
    );
  });

  it("does not rewrite a [N] index inside code (would corrupt code content)", () => {
    expect(linkFootnotes("use `array[1]` here", sources)).toBe("use `array[1]` here");
    expect(linkFootnotes("```js\nconst x = array[1];\n```", sources)).toBe(
      "```js\nconst x = array[1];\n```",
    );
  });
});
