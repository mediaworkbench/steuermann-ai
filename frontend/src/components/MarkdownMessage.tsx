"use client";

import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Highlight, themes } from "prism-react-renderer";
import { Check, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { useTheme } from "@/hooks/useTheme";
import { linkFootnotes, normalizeMath } from "@/lib/markdown";
import type { Source } from "@/lib/types";

/** Syntax-highlighted, copyable code block (prism-react-renderer, synchronous). */
function CodeBlock({ code, language }: { code: string; language: string }) {
  const { effectiveTheme } = useTheme();
  const [copied, setCopied] = useState(false);
  const theme = effectiveTheme === "dark" ? themes.oneDark : themes.oneLight;

  const handleCopy = () => {
    navigator.clipboard
      .writeText(code)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => {});
  };

  return (
    <div className="my-2 overflow-hidden rounded-lg border border-border">
      <div className="flex items-center justify-between border-b border-border bg-surface-muted px-3 py-1">
        <span className="select-none font-mono text-xs text-muted-foreground">{language}</span>
        <Tooltip>
          <TooltipTrigger render={
            <Button
              type="button"
              onClick={handleCopy}
              variant="ghost"
              size="sm"
              className={`p-1 rounded transition-colors cursor-pointer ${
                copied
                  ? "text-primary"
                  : "text-muted-foreground hover:bg-primary/10 hover:text-primary"
              }`}
              aria-label="Copy code"
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </Button>
          } />
          <TooltipContent>Copy code</TooltipContent>
        </Tooltip>
      </div>
      <Highlight code={code} language={language} theme={theme}>
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <pre className={`${className} text-sm p-3 overflow-x-auto`} style={style}>
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })}>
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token })} />
                ))}
              </div>
            ))}
          </pre>
        )}
      </Highlight>
    </div>
  );
}

/** Render markdown content with math (KaTeX), code highlighting, and footnote linking. */
export function MarkdownMessage({ content, sources }: { content: string; sources?: Source[] }) {
  const processed = useMemo(
    () => linkFootnotes(normalizeMath(content), sources),
    [content, sources],
  );
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[[rehypeKatex, { strict: false }]]}
      components={{
        a: ({ href, children, ...props }) => (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
              className="break-all text-primary underline hover:text-primary/80"
            {...props}
          >
            {children}
          </a>
        ),
        p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
        h1: ({ children }) => <h1 className="text-xl font-bold mb-2 mt-4 first:mt-0">{children}</h1>,
        h2: ({ children }) => <h2 className="text-lg font-bold mb-2 mt-3 first:mt-0">{children}</h2>,
        h3: ({ children }) => <h3 className="text-base font-semibold mb-1.5 mt-3 first:mt-0">{children}</h3>,
        ul: ({ children }) => <ul className="list-disc pl-5 mb-3 space-y-1">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal pl-5 mb-3 space-y-1">{children}</ol>,
        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
        em: ({ children }) => <em className="italic">{children}</em>,
        code: ({ className, children }) => {
          const text = Array.isArray(children) ? children.join("") : String(children ?? "");
          const isBlock = (className?.includes("language-") ?? false) || text.includes("\n");
          if (isBlock) {
            const match = /language-(\w+)/.exec(className ?? "");
            return <CodeBlock code={text.replace(/\n$/, "")} language={match ? match[1] : "text"} />;
          }
          return (
            <code className="rounded bg-surface-muted px-1.5 py-0.5 font-mono text-sm">{children}</code>
          );
        },
        // CodeBlock renders its own <pre>; keep this transparent so we never nest a block in <pre>.
        pre: ({ children }) => <>{children}</>,
        blockquote: ({ children }) => (
          <blockquote className="my-2 border-l-3 border-primary/40 pl-3 italic text-muted-foreground">
            {children}
          </blockquote>
        ),
        table: ({ children }) => (
          <div className="overflow-x-auto my-3">
            <table className="min-w-full rounded border border-border text-sm">{children}</table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border-b border-border bg-surface-muted px-3 py-1.5 text-left font-semibold">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border-b border-border/60 px-3 py-1.5">{children}</td>
        ),
        hr: () => <hr className="my-4 border-border" />,
      }}
    >
      {processed}
    </ReactMarkdown>
  );
}
