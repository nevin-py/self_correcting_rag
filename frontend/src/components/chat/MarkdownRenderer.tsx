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
  "prose prose-invert max-w-none " +
  // Readability: body copy 15px/1.7, sans — the gothic display font never
  // touches message content, including markdown headings inside an answer.
  "prose-p:text-[15px] prose-p:leading-[1.7] prose-p:text-text-primary " +
  "prose-headings:font-body prose-headings:text-text-primary prose-headings:font-semibold " +
  "prose-h1:text-[19px] prose-h2:text-[17px] prose-h3:text-[16px] " +
  "prose-strong:text-text-primary prose-strong:font-semibold " +
  "prose-a:text-accent-bright prose-a:no-underline hover:prose-a:underline " +
  "prose-li:text-[15px] prose-li:leading-[1.65] prose-ul:my-2 prose-ol:my-2 " +
  "prose-blockquote:border-l-2 prose-blockquote:border-accent prose-blockquote:text-text-secondary ";


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
              <div className="my-3 overflow-hidden rounded-md border border-border bg-surface-inset">
                <div className="h-px w-full border-t border-accent/70" aria-hidden="true" />
                <pre className="overflow-x-auto p-3">
                  <code className="font-mono text-[13.5px] leading-relaxed text-text-secondary" {...props}>
                    {children}
                  </code>
                </pre>
              </div>
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
