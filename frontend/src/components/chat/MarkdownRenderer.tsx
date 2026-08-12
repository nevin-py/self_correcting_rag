"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function cleanLLMOutput(text: string): string {
  return text
    // Strip HTML tags that LLMs sometimes produce
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/?p>/gi, "\n")
    .replace(/<\/?strong>/gi, "**")
    .replace(/<\/?em>/gi, "*")
    .replace(/<\/?(?:div|span|table|tr|td|th|thead|tbody)[^>]*>/gi, "")
    // Strip citation markers like 【1†L1-L4】 or [1†L1-L4]
    .replace(/【\d+†[^】]*】/g, "")
    .replace(/\[\d+†[^\]]*\]/g, "")
    // Clean up excessive newlines
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

const PROSE_CLASSES = [
  "prose prose-invert prose-sm max-w-none",
  "prose-headings:text-[var(--arctic)] prose-headings:font-display",
  "prose-p:text-[var(--slopes)] prose-p:leading-relaxed",
  "prose-strong:text-[var(--arctic)]",
  "prose-a:text-[var(--glacier)] prose-a:no-underline hover:prose-a:underline",
  "prose-code:text-[var(--glacier)] prose-code:bg-[var(--midnight)] prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-code:font-mono",
  "prose-pre:bg-[var(--midnight)] prose-pre:border prose-pre:border-[var(--apres-ski)]/20",
  "prose-th:text-[var(--arctic)] prose-th:font-medium prose-th:border prose-th:border-[var(--apres-ski)]/20 prose-th:px-3 prose-th:py-2 prose-th:bg-[var(--midnight)]",
  "prose-td:text-[var(--slopes)] prose-td:border prose-td:border-[var(--apres-ski)]/20 prose-td:px-3 prose-td:py-2",
  "prose-li:text-[var(--slopes)]",
  "prose-blockquote:border-l-[var(--glacier)] prose-blockquote:text-[var(--apres-ski)]",
].join(" ");

export default function MarkdownRenderer({ content }: { content: string }) {
  const cleaned = cleanLLMOutput(content);

  return (
    <div className={PROSE_CLASSES}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="overflow-x-auto my-3">
              <table className="w-full text-sm">{children}</table>
            </div>
          ),
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer"
              className="text-[var(--glacier)] hover:underline">{children}</a>
          ),
          code: ({ className, children, ...props }) => {
            const isInline = !className;
            if (isInline) {
              return <code className="bg-[var(--midnight)] text-[var(--glacier)] px-1.5 py-0.5 rounded text-xs font-mono" {...props}>{children}</code>;
            }
            return (
              <pre className="bg-[var(--midnight)] border border-[var(--apres-ski)]/20 rounded-[var(--radius-sm)] p-4 overflow-x-auto my-3">
                <code className="text-xs font-mono text-[var(--slopes)]" {...props}>{children}</code>
              </pre>
            );
          },
        }}
      >
        {cleaned}
      </ReactMarkdown>
    </div>
  );
}
