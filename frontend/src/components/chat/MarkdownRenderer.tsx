"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation } from "@/lib/api";

function cleanLLMOutput(text: string): string {
  return text
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/?p>/gi, "\n")
    .replace(/<\/?strong>/gi, "**")
    .replace(/<\/?em>/gi, "*")
    .replace(/<\/?(?:div|span|table|tr|td|th|thead|tbody)[^>]*>/gi, "")
    .replace(/【\d+†[^】]*】/g, "")
    .replace(/\[\d+†[^\]]*\]/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

const PROSE =
  "prose prose-invert prose-sm max-w-none prose-headings:font-display prose-headings:text-text-primary prose-headings:font-semibold prose-headings:tracking-tight prose-p:text-text-primary prose-p:leading-relaxed prose-strong:text-text-primary prose-a:text-accent-bright prose-a:underline prose-a:underline-offset-2 hover:prose-a:text-accent prose-code:font-mono prose-code:text-accent-bright prose-code:bg-surface-inset prose-code:px-1 prose-code:py-0.5 prose-code:text-xs prose-pre:bg-surface-inset prose-pre:border prose-pre:border-border prose-th:border prose-th:border-border prose-th:px-3 prose-th:py-2 prose-th:bg-surface-inset prose-th:text-text-primary prose-td:border prose-td:border-border prose-td:px-3 prose-td:py-2 prose-li:text-text-primary prose-blockquote:border-l-accent prose-blockquote:text-text-secondary prose-blockquote:not-italic";

export default function MarkdownRenderer({
  content,
  citations,
}: {
  content: string;
  citations?: Citation[];
}) {
  const cleaned = cleanLLMOutput(content);

  return (
    <div className={PROSE}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="my-3 overflow-x-auto border border-border">
              <table className="w-full text-sm">{children}</table>
            </div>
          ),
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
          code: ({ className, children, ...props }) => {
            const isInline = !className;
            if (isInline) {
              return (
                <code className="font-mono text-xs" {...props}>
                  {children}
                </code>
              );
            }
            return (
              <pre className="my-3 overflow-x-auto border border-border p-4">
                <code className="font-mono text-xs text-text-secondary" {...props}>
                  {children}
                </code>
              </pre>
            );
          },
          // Inline citation markers like [1] get styled
          p: ({ children }) => {
            return <p className="leading-relaxed">{children}</p>;
          },
        }}
      >
        {cleaned}
      </ReactMarkdown>
      {citations && citations.length > 0 && (
        <div className="mt-3 hidden border-t border-border pt-2">
          {/* Reserved for inline citation expansion */}
        </div>
      )}
    </div>
  );
}
