"use client";

import { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Highlight, themes } from "prism-react-renderer";
import { Icon } from "./Icon";
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
    <div className="my-2 rounded-lg border border-evergreen/10 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-1 bg-evergreen/5 border-b border-evergreen/10">
        <span className="text-xs font-mono text-evergreen/50 select-none">{language}</span>
        <button
          onClick={handleCopy}
          className={`p-1 rounded transition-colors cursor-pointer ${
            copied
              ? "text-pacific-blue"
              : "text-evergreen/40 hover:text-pacific-blue hover:bg-pacific-blue/10"
          }`}
          aria-label="Copy code"
          title="Copy code"
        >
          <Icon name={copied ? "check" : "content_copy"} size={14} />
        </button>
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
            className="text-pacific-blue underline hover:text-pacific-blue/80 break-all"
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
            <code className="bg-evergreen/5 rounded px-1.5 py-0.5 text-sm font-mono">{children}</code>
          );
        },
        // CodeBlock renders its own <pre>; keep this transparent so we never nest a block in <pre>.
        pre: ({ children }) => <>{children}</>,
        blockquote: ({ children }) => (
          <blockquote className="border-l-3 border-pacific-blue/40 pl-3 my-2 text-evergreen/70 italic">
            {children}
          </blockquote>
        ),
        table: ({ children }) => (
          <div className="overflow-x-auto my-3">
            <table className="min-w-full text-sm border border-evergreen/10 rounded">{children}</table>
          </div>
        ),
        th: ({ children }) => (
          <th className="px-3 py-1.5 text-left font-semibold bg-light-cyan/20 border-b border-evergreen/10">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="px-3 py-1.5 border-b border-evergreen/5">{children}</td>
        ),
        hr: () => <hr className="my-4 border-evergreen/10" />,
      }}
    >
      {processed}
    </ReactMarkdown>
  );
}
