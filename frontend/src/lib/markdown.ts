/**
 * Pure string preprocessors for assistant markdown. Kept free of React / ESM-only deps
 * (react-markdown, katex, prism-react-renderer) so they can be unit-tested directly without
 * Jest needing to transform those packages. Consumed by `components/MarkdownMessage.tsx`.
 */
import type { Source } from "./types";

/**
 * Apply `fn` only to the parts of `text` that are NOT code — fenced ``` blocks and inline
 * `code` spans are left verbatim. The capturing split keeps the delimiters in the array so the
 * odd indices are the (untouched) code segments. Shared by every preprocessor here so neither
 * math nor footnote rewriting can ever corrupt code content.
 */
function processOutsideCode(text: string, fn: (segment: string) => string): string {
  return text
    .split(/(```[\s\S]*?```|`[^`\n]*`)/g)
    .map((segment, i) => (i % 2 === 1 ? segment : fn(segment)))
    .join("");
}

/**
 * Replace [N] footnote references with clickable markdown links using the sources array.
 * E.g. "[1]" becomes a superscript link if source 1 has a URL, or bold for RAG sources.
 * Citations inside code are left untouched (e.g. `array[1]` stays literal).
 */
export function linkFootnotes(text: string, sources?: Source[]): string {
  if (!sources || sources.length === 0) return text;
  // Build a map from index (1-based from backend) to source
  const indexMap = new Map<number, Source>();
  sources.forEach((s) => {
    if (s.index) indexMap.set(s.index, s);
  });
  // Also fall back to position-based if no index field
  if (indexMap.size === 0) {
    sources.forEach((s, i) => indexMap.set(i + 1, s));
  }

  const SUPERSCRIPT = ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"];
  const toSup = (n: number) =>
    String(n).split("").map((d) => SUPERSCRIPT[parseInt(d)]).join("");

  // Match [N], [N, M], [N, M, O] patterns — outside code only.
  return processOutsideCode(text, (segment) =>
    segment.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_match, nums: string) => {
      const numbers = nums.split(",").map((n: string) => parseInt(n.trim(), 10));
      const parts = numbers.map((n) => {
        const src = indexMap.get(n);
        if (!src) return `[${n}]`;
        if (src.url) return `[${toSup(n)}](${src.url})`;
        return `**[${toSup(n)}]**`;
      });
      return parts.join(" ");
    }),
  );
}

/**
 * Currency-looking dollar amounts ($5, $1,000.50, $5K) — escaped *before* math parsing so
 * remark-math never treats them as inline-math delimiters. Critical for a finance/tax assistant.
 */
const CURRENCY_RE =
  /(?<![\\$])\$(?!\$)(?=\d+(?:,\d{3})*(?:\.\d+)?(?:[KMBkmb])?(?:\s|$|[^a-zA-Z\d]))/g;

/**
 * Normalize the many ways different LLMs emit math into the `$`/`$$` delimiters remark-math
 * understands, while keeping currency literal and never touching code.
 *
 * Order matters: protect code → escape currency → convert \( \) / \[ \] delimiters.
 */
export function normalizeMath(content: string): string {
  // Function replacers avoid `$`-in-replacement-string ambiguity.
  return processOutsideCode(content, (segment) =>
    segment
      .replace(CURRENCY_RE, () => "\\$") // $5 -> \$5  (literal)
      .replace(/\\\[/g, () => "$$") // \[ -> $$  (block math)
      .replace(/\\\]/g, () => "$$")
      .replace(/\\\(/g, () => "$") // \( -> $   (inline math)
      .replace(/\\\)/g, () => "$"),
  );
}
